use std::{
    cell::{Cell, RefCell},
    cmp::Ordering,
};

use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, WorkgroupCount},
    rvec, wgc, DType, InvariantError, KernelElement, MetaOperation, OpGuards, OpMetadata,
    Operation, OperationError, RVec, Shape, StorageView, Strides, Tensor,
};

// Defines a matrix multiplication operation.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct MatmulSpec {
    a_dt: DType,
    b_dt: DType,
    a_shape: Shape,
    b_shape: Shape,
    c_shape: Shape,
    a_stack: usize,
    b_stack: usize,
    c_stack: usize,
    trans_b: bool,
    stack_shape: Shape, //N-D matmul is handled by stacking the first N-2 dimensions
}

impl MatmulSpec {
    //TODO: parameterize these
    pub const TILE_DIM: usize = 32;
    pub const ROW_PER_THREAD: usize = 4;

    pub fn new(A: &Tensor, B: &Tensor, C: &Tensor, trans_b: bool) -> Self {
        let mut a_shape = A.shape().clone();
        let mut b_shape = B.shape().clone();
        let mut c_shape = C.shape().clone();
        let a_dt = A.dt();
        let b_dt = B.dt();

        if (a_shape.rank() < 2) || (b_shape.rank() < 2) {
            panic!("MatMul: inputs must be at least 2D");
        }

        match a_shape.rank().cmp(&b_shape.rank()) {
            Ordering::Less => {
                a_shape.left_pad_to(1, b_shape.rank());
            }
            Ordering::Greater => {
                b_shape.left_pad_to(1, a_shape.rank());
            }
            _ => {}
        };

        let _b_rank = b_shape.rank();

        let stack_dims = c_shape.rank() - 2;
        let stack_shape = c_shape.slice(0..stack_dims);

        let a_stack = a_shape.drain(0..stack_dims).product();
        let b_stack = b_shape.drain(0..stack_dims).product();
        let c_stack = c_shape.drain(0..stack_dims).product();

        if a_stack != 1 && b_stack != 1 {
            //Here we want all of the stacks to be equal
            //OR A or B to be 1
            assert!(a_stack == b_stack && b_stack == c_stack);
        }

        if a_shape.rank() == 1 {
            a_shape.insert(0, 1);
        }

        if b_shape.rank() == 1 {
            b_shape.insert(0, 1);
        }

        log::debug!(
            "MatMul stacking: left {} right {} stack_dims={} stack_count={}",
            a_shape,
            b_shape,
            stack_dims,
            stack_shape.numel(),
        );
        Self {
            a_dt,
            b_dt,
            a_shape,
            b_shape,
            c_shape,
            a_stack,
            b_stack,
            c_stack,
            trans_b,
            stack_shape,
        }
    }

    pub fn select_kernel_element(&self) -> KernelElement {
        log::debug!(
            "select_kernel: m={} n={} k={}",
            self.m(),
            self.n(),
            self.k()
        );

        let checks = [
            self.k(),
            self.n(),
            self.a_shape.numel(),
            self.b_shape.numel(),
            self.c_shape.numel(),
        ];

        if checks.iter().all(|&x| x % 4 == 0) {
            KernelElement::Vec4
        } else if checks.iter().all(|&x| x % 2 == 0) {
            KernelElement::Vec2
        } else {
            KernelElement::Scalar
        }
    }

    pub fn m(&self) -> usize {
        self.a_shape[0]
    }

    pub fn k(&self) -> usize {
        self.a_shape[1]
    }

    pub fn n(&self) -> usize {
        if self.trans_b {
            self.b_shape[0]
        } else {
            self.b_shape[1]
        }
    }

    pub fn a_stack(&self) -> usize {
        self.a_stack
    }

    pub fn b_stack(&self) -> usize {
        self.b_stack
    }

    pub fn c_stack(&self) -> usize {
        self.c_stack
    }

    pub fn stacks(&self) -> usize {
        self.stack_shape.numel()
    }

    pub fn b_dt(&self) -> DType {
        self.b_dt
    }

    pub fn stacked_shapes(&self) -> (Shape, Shape, Shape) {
        let mut a_shape = self.a_shape.clone();
        let mut b_shape = self.b_shape.clone();
        let mut c_shape = self.c_shape.clone();
        a_shape.insert(0, self.stacks());
        b_shape.insert(0, self.stacks());
        c_shape.insert(0, self.stacks());
        (a_shape, b_shape, c_shape)
    }

    pub fn tile_fit(&self) -> (bool, bool, bool) {
        let a_fit = self.m() % Self::TILE_DIM == 0;
        let b_fit = self.n() % Self::TILE_DIM == 0;
        let out_fit = self.k() % Self::TILE_DIM == 0;
        (a_fit, b_fit, out_fit)
    }
}

#[derive(new, Debug, Clone)]
pub struct Matmul {
    lhs: Tensor,
    rhs: Tensor,
    trans_b: bool,
    spec: RefCell<Option<MatmulSpec>>,
}

impl Matmul {
    pub fn compute_c_shape(a: &Tensor, b: &Tensor, trans_b: bool) -> anyhow::Result<Shape> {
        let (mut ashape, mut bshape) = (a.shape().clone(), b.shape().clone());

        let implicit_m = ashape.rank() < 2;
        let implicit_n = bshape.rank() < 2;
        if implicit_m {
            ashape.insert(0, 1);
        }
        if implicit_n {
            bshape.insert(!trans_b as usize, 1);
        }

        let equalize_rank = |shape: &mut Shape, target_rank: usize| {
            while shape.rank() < target_rank {
                shape.insert(0, 1);
            }
        };
        equalize_rank(&mut ashape, bshape.rank());
        equalize_rank(&mut bshape, ashape.rank());

        let arank = ashape.rank();
        let brank = bshape.rank();
        let (a_prefix, b_prefix) = (&ashape[..arank - 2], &bshape[..brank - 2]);
        let c_broadcasted_prefix = Shape::multi_broadcast(&[&a_prefix.into(), &b_prefix.into()])
            .ok_or_else(|| {
                anyhow::anyhow!("Matmul broadcasting: a: {:?} b: {:?}", ashape, bshape)
            })?;

        let (m, ka) = (ashape[arank - 2], ashape[arank - 1]);
        let (mut kb, mut n) = (bshape[brank - 2], bshape[brank - 1]);
        if trans_b {
            std::mem::swap(&mut kb, &mut n);
        }

        if ka != kb {
            anyhow::bail!("Matmul broadcasting: a: {:?} b: {:?}", ashape, bshape);
        }

        let mut c_shape_final = c_broadcasted_prefix;
        if ashape.rank() >= 2 {
            c_shape_final.push(m);
        }
        if bshape.rank() >= 2 {
            c_shape_final.push(n);
        }

        Ok(c_shape_final)
    }

    pub fn compute_spec(&self, dst: &Tensor) {
        let spec = MatmulSpec::new(&self.lhs, &self.rhs, dst, self.trans_b);
        self.spec.replace(Some(spec));
    }
}

/*
#[allow(clippy::too_many_arguments)]
#[derive(Debug, Clone, ShaderType)]
pub struct MatmulMeta {
    M: u32,
    N: u32,
    K: u32,
    MD2: u32,
    ND2: u32,
    KD2: u32,
    MD4: u32,
    ND4: u32,
    KD4: u32,
    A_OFFSET: u32, //batch offset
    B_OFFSET: u32,
    C_OFFSET: u32,
}
*/

#[allow(clippy::too_many_arguments)]
#[derive(Debug, Clone, ShaderType)]
pub struct MatmulMeta {
    aShape: glam::IVec3,
    bShape: glam::IVec3,
    outShape: glam::IVec3,
    outShapeStrides: glam::IVec3,
    dimInner: u32,
}

impl OpMetadata for MatmulMeta {}

impl Operation for Matmul {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let c_shape = Matmul::compute_c_shape(&self.lhs, &self.rhs, self.trans_b).unwrap();
        let c_strides = Strides::from(&c_shape);
        Ok(StorageView::new(c_shape, self.lhs.dt(), c_strides))
    }
}

impl OpGuards for Matmul {
    fn check_shapes(&self) {
        let c_shape = Matmul::compute_c_shape(&self.lhs, &self.rhs, self.trans_b);
        assert!(c_shape.is_ok());
    }

    fn check_dtypes(&self) {
        let allowed_pairs = [(DType::F32, DType::F32), (DType::F32, DType::WQ8)];
        if !allowed_pairs.contains(&(self.lhs.dt(), self.rhs.dt())) {
            panic!(
                "Failed to validate DTypes: {:?}, {:?}",
                self.lhs.dt(),
                self.rhs.dt()
            );
        }
    }
}

impl MetaOperation for Matmul {
    type Meta = MatmulMeta;

    fn update(&self, dst: &Tensor) -> Result<(), OperationError> {
        self.compute_spec(dst);
        Ok(())
    }

    fn kernel_key(&self) -> String {
        let ref_spec = self.spec.borrow();
        let spec = ref_spec.as_ref().unwrap();
        let (a_fit, b_fit, out_fit) = spec.tile_fit();

        let selected = if self.trans_b {
            let old = match (self.lhs.dt(), self.rhs.dt(), self.trans_b) {
                (DType::F32, DType::F32, false) => "sgemm",
                (DType::F32, DType::WQ8, false) => "qgemm",
                (DType::F32, DType::F32, true) => "sgemm_bt",
                (DType::F32, DType::WQ8, true) => "qgemm_bt",
                _ => panic!(
                    "Unsupported matmul: {:?}, {:?}, transb:{:?}",
                    self.lhs.dt(),
                    self.rhs.dt(),
                    self.trans_b
                ),
            };
            old.to_string()
        } else {
            let key_base = match (self.lhs.dt(), self.rhs.dt()) {
                (DType::F32, DType::F32) => "sgemm",
                (DType::F32, DType::WQ8) => "qgemm",
                _ => panic!(
                    "Unsupported matmul: {:?}, {:?}",
                    self.lhs.dt(),
                    self.rhs.dt(),
                ),
            };
            format!(
                "{}_A_FIT{}_B_FIT{}_OUT_FIT{}",
                key_base, a_fit, b_fit, out_fit
            )
        };
        println!("SPEC: {:?}", spec);
        println!("SELECTED MATMUL: {}", selected);
        selected
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.lhs, &self.rhs]
    }

    fn kernel_element(&self, _: &Tensor) -> KernelElement {
        let ref_spec = self.spec.borrow();
        let spec = ref_spec.as_ref().unwrap();
        spec.select_kernel_element()
    }

    fn calculate_dispatch(&self, _: &Tensor) -> Result<WorkgroupCount, OperationError> {
        let ref_spec = self.spec.borrow();
        let spec = ref_spec.as_ref().unwrap();

        let group_x = WorkgroupCount::div_ceil(spec.m(), 32) as _;
        let group_y = WorkgroupCount::div_ceil(spec.n(), 32) as _;

        let dispatch = wgc![group_x, group_y, spec.stacks() as _];
        println!("DISPATCH: {:?}", dispatch);

        Ok(dispatch)
    }

    fn storage_bind_group_layout(
        &self,
        _inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        let (A, B) = (&self.lhs, &self.rhs);
        let layout = match (A.dt(), B.dt()) {
            (DType::F32, DType::F32) => BindGroupLayoutDescriptor::binary(),
            (DType::F32, DType::WQ8) => BindGroupLayoutDescriptor::ternary(),
            _ => return Err(InvariantError::UnsupportedDType(B.dt()).into()),
        };
        Ok(layout)
    }

    fn metadata(&self, _: &Tensor, ke: &KernelElement) -> Result<Self::Meta, OperationError> {
        let ref_spec = self.spec.borrow();
        let spec = ref_spec.as_ref().unwrap();

        let mut out_shape = spec.c_shape.clone();
        out_shape.insert(0, spec.stacks());

        let outShapeStrides = Strides::from(&out_shape).to_vec();
        let outShapeStrides = glam::IVec3::new(
            outShapeStrides[0] as _,
            outShapeStrides[1] as _,
            outShapeStrides[2] as _,
        );

        let meta = MatmulMeta {
            aShape: glam::IVec3::new(spec.a_stack() as _, spec.m() as _, spec.k() as _),
            bShape: glam::IVec3::new(spec.b_stack() as _, spec.k() as _, spec.n() as _),
            outShape: glam::IVec3::new(spec.c_stack() as _, spec.m() as _, spec.n() as _),
            outShapeStrides: outShapeStrides.into(),
            dimInner: spec.k() as _,
        };

        println!("METADATA: {:?}", meta);

        Ok(meta)
    }
}

#[cfg(test)]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::test_util::run_py_prg;

    use crate::{shape, Device, DeviceRequest, Quantization, Quantizer};

    use super::*;

    fn matmul_harness() -> anyhow::Result<(Tensor, Tensor)> {
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let a = Tensor::randn::<f32>(shape![6, 1500, 64], cpu_device.clone());
        let b = Tensor::randn::<f32>(shape![6, 64, 1500], cpu_device.clone());
        Ok((a, b))
    }

    fn ground_truth(a: &Tensor, b: &Tensor) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
def matmul(a, b):
    return torch.matmul(torch.from_numpy(a), torch.from_numpy(b)).numpy()"#;
        run_py_prg(prg.to_string(), &[a, b], &[])
    }

    #[derive(Arbitrary, Debug)]
    struct SGEMMProblem {
        #[strategy(1..=4usize)]
        B: usize,
        #[strategy(1..=512usize)]
        M: usize,
        #[strategy(1..=512usize)]
        K: usize,
        #[strategy(1..=512usize)]
        N: usize,
    }

    #[proptest(cases = 8)]
    fn test_sgemm(prob: SGEMMProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let SGEMMProblem { B, M, K, N } = prob;
        println!("Running sgemm: B={} M={} K={} N={}", B, M, K, N);
        run_matmul_trial(&device, prob).unwrap();
    }

    fn run_matmul_trial(device: &Device, prob: SGEMMProblem) -> anyhow::Result<()> {
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let SGEMMProblem { B, M, K, N } = prob;
        let a = Tensor::randn::<f32>(shape![B, M, K], cpu_device.clone());
        let b = Tensor::randn::<f32>(shape![B, K, N], cpu_device.clone());
        let ground = ground_truth(&a, &b)?;

        let a_gpu = a.to(device)?;
        let b_gpu = b.to(device)?;
        let c_gpu = a_gpu.matmul(b_gpu, false)?.resolve()?;

        let d_gpu = c_gpu.to(&Device::CPU)?;
        println!("RATCHET SGEMM\n{:?}\n", d_gpu);
        println!("PYTORCH FP32:\n{:?}", ground);
        ground.all_close(&d_gpu, 1e-4, 1e-4)?;
        Ok(())
    }

    #[test]
    fn debug_sgemm() {
        let _ = env_logger::builder().is_test(true).try_init();
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let prob = SGEMMProblem {
            B: 1,
            M: 254,
            K: 254,
            N: 254,
        };
        run_matmul_trial(&device, prob).unwrap();
    }

    #[test]
    fn test_qgemm() -> anyhow::Result<()> {
        let (a, b) = matmul_harness()?;
        let ground = ground_truth(&a, &b)?;

        let quantizer = Quantizer::new(Quantization::SInt8);
        let bq = quantizer.sint8_quantize(b);
        let device = Device::request_device(DeviceRequest::GPU)?;
        let a_gpu = a.to(&device)?;
        let b_gpu = bq.to(&device)?;
        let c_gpu = a_gpu.matmul(b_gpu, false)?.resolve()?;
        let ours = c_gpu.to(&Device::CPU)?;

        println!("RATCHET WQ8\n{:?}\n", ours);
        println!("PYTORCH FP32:\n{:?}", ground);

        ground.all_close(&ours, 1e1, 1e-1)?;

        Ok(())
    }
}
