use std::cmp::Ordering;

use encase::ShaderType;

use crate::{
    gguf::{GGUFDType, Q8_0},
    gpu::{BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, wgs, DType, InvariantError, KernelElement, KernelKey, KernelSource, MetaOperation,
    OpGuards, OpMetadata, Operation, OperationError, RVec, Shape, StorageView, Strides, Tensor,
    WorkgroupSize, Workload, GEMM, GEMV,
};

//https://link.springer.com/chapter/10.1007/978-3-642-29737-3_42
#[derive(Debug, Clone)]
pub enum GEMVHeuristic {
    VeryTall,
    Tall,
    Square,
    Fat,
}

impl GEMVHeuristic {
    pub fn as_workgroup_size(&self) -> (usize, usize) {
        match self {
            GEMVHeuristic::Fat => (4, 256),
            _ => (8, 8),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct GEMMSpec {
    a_dt: DType,
    b_dt: DType,
    a_shape: Shape,
    b_shape: Shape,
    c_shape: Shape,
    a_stack: usize,
    b_stack: usize,
    c_stack: usize,
    trans_a: bool,
    trans_b: bool,
    trans_out: bool,
    stack_shape: Shape, //N-D matmul is handled by stacking the first N-2 dimensions
    pub heuristic: GEMVHeuristic,
}

impl GEMMSpec {
    pub const TILE_DIM: usize = 32;
    pub const ROW_PER_THREAD: usize = 4;

    pub fn new(
        A: &Tensor,
        B: &Tensor,
        C: &Tensor,
        trans_a: bool,
        trans_b: bool,
        trans_out: bool,
    ) -> Self {
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

        let heuristic = match (a_shape[0], a_shape[1]) {
            (arows, acols) if arows > acols * 4 => GEMVHeuristic::VeryTall,
            (arows, acols) if arows > acols * 2 => GEMVHeuristic::Tall,
            (arows, acols) if acols > arows * 2 => GEMVHeuristic::Fat,
            _ => GEMVHeuristic::Square,
        };

        Self {
            a_dt,
            b_dt,
            a_shape,
            b_shape,
            c_shape,
            a_stack,
            b_stack,
            c_stack,
            trans_a,
            trans_b,
            trans_out,
            stack_shape,
            heuristic,
        }
    }

    pub fn select_kernel_element(&self) -> KernelElement {
        if self.trans_a || self.trans_b || self.trans_out || self.b_shape.is_vector() {
            //We cannot support transposed with vectorized kernels
            //If GEMV we use Scalar
            return KernelElement::Scalar;
        }

        let checks = [
            self.dim_inner(),
            self.c_shape[1],
            self.a_shape.numel(),
            self.b_shape.numel(),
            self.c_shape.numel(),
        ];

        if checks.iter().all(|&x| x % 4 == 0) {
            KernelElement::Vec4
        } else {
            KernelElement::Scalar
        }
    }

    pub fn lhs_shape(&self) -> &Shape {
        &self.a_shape
    }

    pub fn rhs_shape(&self) -> &Shape {
        &self.b_shape
    }

    pub fn c_shape(&self) -> &Shape {
        &self.c_shape
    }

    pub fn dim_a_outer(&self) -> usize {
        self.c_shape[0]
    }
    pub fn dim_b_outer(&self) -> usize {
        self.c_shape[1]
    }

    pub fn dim_inner(&self) -> usize {
        if self.trans_a {
            self.a_shape[0]
        } else {
            self.a_shape[1]
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

    pub fn is_gemv(&self) -> bool {
        self.b_shape.is_vector() && !self.trans_a
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
        let dimAOuter = self.dim_a_outer();
        let dimBOuter = self.dim_b_outer();
        let dimInner = self.dim_inner();

        let a_fit = dimAOuter % Self::TILE_DIM == 0;
        let b_fit = dimBOuter % Self::TILE_DIM == 0;
        let out_fit = dimInner % Self::TILE_DIM == 0;
        (a_fit, b_fit, out_fit)
    }
}

#[derive(Debug, Clone)]
pub struct Matmul {
    pub(crate) lhs: Tensor,
    pub(crate) rhs: Tensor,
    pub(crate) bias: Option<Tensor>,
    pub(crate) trans_lhs: bool,
    pub(crate) trans_rhs: bool,
    pub(crate) trans_out: bool,
}

impl Matmul {
    pub fn new(
        lhs: Tensor,
        rhs: Tensor,
        bias: Option<Tensor>,
        trans_lhs: bool,
        trans_rhs: bool,
        trans_out: bool,
    ) -> Self {
        if !bias.as_ref().map_or(true, |b| b.shape().is_vector()) {
            panic!("Bias must be a vector: {:?}", bias);
        }

        if matches!(lhs.dt(), DType::GGUF(GGUFDType::Q8_0(_))) && trans_lhs {
            panic!("Transposed quantized inputs are not supported");
        }

        Self {
            lhs,
            rhs,
            bias,
            trans_lhs,
            trans_rhs,
            trans_out,
        }
    }

    pub fn compute_c_shape(
        a: &Tensor,
        b: &Tensor,
        trans_lhs: bool,
        trans_rhs: bool,
        trans_out: bool,
    ) -> anyhow::Result<Shape> {
        let (mut ashape, mut bshape) = (a.shape().clone(), b.shape().clone());

        let implicit_m = ashape.rank() < 2;
        let implicit_n = bshape.rank() < 2;
        if implicit_m {
            ashape.insert(trans_lhs as usize, 1);
        }
        if implicit_n {
            bshape.insert(!trans_rhs as usize, 1);
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
        let mut c_broadcasted_prefix =
            Shape::multi_broadcast(&[&a_prefix.into(), &b_prefix.into()]).ok_or_else(|| {
                anyhow::anyhow!(
                    "Matmul broadcasting: a: {:?} b: {:?} trans_a: {:?} trans_b: {:?}",
                    ashape,
                    bshape,
                    trans_lhs,
                    trans_rhs
                )
            })?;

        let (mut m, mut ka) = (ashape[arank - 2], ashape[arank - 1]);
        let (mut kb, mut n) = (bshape[brank - 2], bshape[brank - 1]);

        if trans_lhs {
            std::mem::swap(&mut m, &mut ka);
        }

        if trans_rhs {
            std::mem::swap(&mut kb, &mut n);
        }

        if ka != kb {
            anyhow::bail!("Matmul broadcasting: ka != kb: {} != {}", ka, kb);
        }

        let mut c_shape_final = c_broadcasted_prefix.clone();
        if trans_out {
            c_broadcasted_prefix.push(n);
            c_broadcasted_prefix.push(m);
            if !implicit_n {
                c_shape_final.push(n);
            }
            if !implicit_m {
                c_shape_final.push(m);
            }
        } else {
            c_broadcasted_prefix.push(m);
            c_broadcasted_prefix.push(n);
            if !implicit_m {
                c_shape_final.push(m);
            }
            if !implicit_n {
                c_shape_final.push(n);
            }
        }

        Ok(c_shape_final)
    }

    pub fn compute_spec(&self, dst: &Tensor) -> GEMMSpec {
        GEMMSpec::new(
            &self.lhs,
            &self.rhs,
            dst,
            self.trans_lhs,
            self.trans_rhs,
            self.trans_out,
        )
    }
}

#[allow(clippy::too_many_arguments)]
#[derive(Debug, Clone, ShaderType)]
pub struct MatmulMeta {
    aShape: glam::IVec3,
    aStrides: glam::IVec3,
    bShape: glam::IVec3,
    bStrides: glam::IVec3,
    outShape: glam::IVec3,
    outStrides: glam::IVec3,
    dimAOuter: i32,
    dimBOuter: i32,
    dimInner: i32,
}

impl OpMetadata for MatmulMeta {}

impl Operation for Matmul {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let c_shape = Matmul::compute_c_shape(
            &self.lhs,
            &self.rhs,
            self.trans_lhs,
            self.trans_rhs,
            self.trans_out,
        )
        .unwrap();
        let c_strides = Strides::from(&c_shape);
        Ok(StorageView::new(c_shape, self.rhs.dt(), c_strides))
    }
}

impl OpGuards for Matmul {
    fn check_shapes(&self) {
        let c_shape = Matmul::compute_c_shape(
            &self.lhs,
            &self.rhs,
            self.trans_lhs,
            self.trans_rhs,
            self.trans_out,
        );
        assert!(c_shape.is_ok());
    }

    fn check_dtypes(&self) {
        let allowed_pairs = [
            (DType::F32, DType::F32),
            (DType::F16, DType::F16),
            (DType::GGUF(GGUFDType::Q8_0(Q8_0)), DType::F32),
        ];
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
    fn kernel_name(&self) -> String {
        "GEMM".to_string()
    }

    fn kernel_key(
        &self,
        workgroup_size: &WorkgroupSize,
        inplace: bool,
        dst: &Tensor,
        kernel_element: &KernelElement,
    ) -> KernelKey {
        let spec = self.compute_spec(dst);
        let kernel_stem = if spec.is_gemv() { "gemv" } else { "gemm" };
        let (a_fit, b_fit, out_fit) = spec.tile_fit();
        let bias_key = if self.bias.is_some() { "bias" } else { "" };

        let additional = format!(
            "{}_{}_{}_{}_{}_{}_{}",
            if a_fit { "" } else { "a_checked" },
            if b_fit { "" } else { "b_checked" },
            if out_fit { "" } else { "out_checked" },
            if self.trans_lhs { "trans_a" } else { "" },
            if self.trans_rhs { "trans_b" } else { "" },
            if self.trans_out { "trans_out" } else { "" },
            bias_key
        );

        KernelKey::new(
            kernel_stem,
            &self.srcs(),
            dst,
            workgroup_size,
            inplace,
            kernel_element,
            Some(&additional),
        )
    }

    fn srcs(&self) -> RVec<&Tensor> {
        if let Some(bias) = &self.bias {
            rvec![&self.lhs, &self.rhs, bias]
        } else {
            rvec![&self.lhs, &self.rhs]
        }
    }

    fn kernel_element(&self, dst: &Tensor) -> KernelElement {
        let spec = self.compute_spec(dst);
        spec.select_kernel_element()
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError> {
        let spec = self.compute_spec(dst);

        if spec.rhs_shape().is_vector() && !self.trans_lhs {
            let (TX, TY) = spec.heuristic.as_workgroup_size();
            let group_x = WorkgroupCount::div_ceil(spec.lhs_shape()[0], TX);

            Ok(Workload {
                workgroup_count: wgc![group_x as _, 1, spec.stacks() as _],
                workgroup_size: wgs![TX as _, TY as _, 1],
            })
        } else {
            let TILE_DIM = 32;
            let a_shape = spec.lhs_shape();
            let b_shape = spec.rhs_shape();

            let dimA = if self.trans_lhs {
                a_shape[1]
            } else {
                a_shape[0]
            };

            let dimB = if self.trans_rhs {
                b_shape[0]
            } else {
                b_shape[1]
            };

            let group_x = WorkgroupCount::div_ceil(dimB as _, TILE_DIM);
            let group_y = WorkgroupCount::div_ceil(dimA, TILE_DIM);
            let workgroup_count = wgc![group_x as _, group_y as _, spec.stacks() as _];

            Ok(Workload {
                workgroup_count,
                workgroup_size: wgs![8, 8, 1],
            })
        }
    }

    fn storage_bind_group_layout(
        &self,
        _inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        let (A, B, bias) = (&self.lhs, &self.rhs, &self.bias);
        let layout = match (A.dt(), B.dt(), bias.is_some()) {
            (DType::F32, DType::F32, false) => BindGroupLayoutDescriptor::binary(),
            (DType::F32, DType::F32, true) => BindGroupLayoutDescriptor::ternary(),
            (DType::F16, DType::F16, false) => BindGroupLayoutDescriptor::binary(),
            (DType::F16, DType::F16, true) => BindGroupLayoutDescriptor::ternary(),
            (DType::GGUF(_), DType::F32, false) => BindGroupLayoutDescriptor::ternary(),
            (DType::GGUF(_), DType::F32, true) => BindGroupLayoutDescriptor::nthary(4),
            _ => return Err(InvariantError::UnsupportedDType(B.dt()).into()),
        };
        Ok(layout)
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        dst: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        let spec = self.compute_spec(dst);

        let mut a_shape = spec.a_shape.clone();
        a_shape.insert(0, spec.a_stack());
        let aStrides = Strides::from(&a_shape);

        let mut b_shape = spec.b_shape.clone();
        b_shape.insert(0, spec.b_stack());
        let bStrides = Strides::from(&b_shape);

        let mut out_shape = spec.c_shape.clone();
        out_shape.insert(0, spec.stacks());
        let outStrides = Strides::from(&out_shape);

        let dimAOuter = spec.dim_a_outer() as i32;
        let dimBOuter = spec.dim_b_outer() as i32;
        let dimInner = spec.dim_inner() as i32;

        let meta = MatmulMeta {
            aShape: a_shape.into(),
            aStrides: aStrides.into(),
            bShape: b_shape.into(),
            bStrides: bStrides.into(),
            outShape: out_shape.into(),
            outStrides: outStrides.into(),
            dimAOuter,
            dimBOuter,
            dimInner,
        };
        Ok(uniform.write(&meta)?)
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let spec = self.compute_spec(dst);
        if spec.is_gemv() {
            let gemv: GEMV = self.clone().into();
            gemv.build_kernel(inplace, dst, workgroup_size, spec)
        } else {
            let gemm: GEMM = self.clone().into();
            gemm.build_kernel(inplace, dst, workgroup_size, spec)
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::test_util::run_py_prg;

    use crate::{shape, Device, DeviceRequest, Quantization, Quantizer};

    use super::*;

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    fn ground_truth(
        a: &Tensor,
        b: &Tensor,
        bias: Option<&Tensor>,
        trans_lhs: bool,
        trans_rhs: bool,
        trans_out: bool,
    ) -> anyhow::Result<Tensor> {
        let a_op = if trans_lhs {
            "torch.permute(torch.from_numpy(a), [0, 2, 1])"
        } else {
            "torch.from_numpy(a)"
        };

        let b_op = if trans_rhs {
            "torch.permute(torch.from_numpy(b), [0, 2, 1])"
        } else {
            "torch.from_numpy(b)"
        };

        let inner = if bias.is_some() {
            format!(
                "torch.add(torch.matmul({}, {}), torch.from_numpy(bias))",
                a_op, b_op
            )
        } else {
            format!("torch.matmul({}, {})", a_op, b_op)
        };

        let result_op = if trans_out {
            format!(
                "np.ascontiguousarray(torch.permute({}, [0, 2, 1]).numpy())",
                inner
            )
        } else {
            format!("{}.numpy()", inner)
        };

        let prg = format!(
            r#"
import torch
import numpy as np
def matmul(a, b{}):
    return {}"#,
            if bias.is_some() { ", bias" } else { "" },
            result_op
        );

        let args = if let Some(bias) = bias {
            vec![a, b, bias]
        } else {
            vec![a, b]
        };

        run_py_prg(prg.to_string(), &args, &[], a.dt())
    }

    #[derive(Arbitrary, Clone, Debug)]
    enum TransKind {
        LHS,
        LHSAndOut,
        RHS,
        RHSAndOut,
        Out,
    }

    impl From<TransKind> for (bool, bool, bool) {
        fn from(val: TransKind) -> Self {
            match val {
                TransKind::LHS => (true, false, false),
                TransKind::LHSAndOut => (true, false, true),
                TransKind::RHS => (false, true, false),
                TransKind::RHSAndOut => (false, true, true),
                TransKind::Out => (false, false, true),
            }
        }
    }

    #[derive(Arbitrary, Debug)]
    struct SGEMMProblem {
        #[strategy(1..=3usize)]
        B: usize,
        #[strategy(1..=256usize)]
        M: usize,
        #[strategy(1..=256usize)]
        K: usize,
        #[strategy(1..=256usize)]
        N: usize,
        has_bias: bool,
        transpose: TransKind,
    }

    #[proptest(cases = 64)]
    fn test_sgemm(prob: SGEMMProblem) {
        let device = GPU_DEVICE.with(|d| d.clone());
        let SGEMMProblem {
            B,
            M,
            K,
            N,
            has_bias,
            ref transpose,
        } = prob;
        println!(
            "Running sgemm: B={} M={} N={} K={} has_bias={} transpose={:?}",
            B, M, N, K, has_bias, transpose
        );
        run_matmul_trial(&device, prob).unwrap();
    }

    fn run_matmul_trial(device: &Device, prob: SGEMMProblem) -> anyhow::Result<()> {
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let SGEMMProblem {
            B,
            M,
            K,
            N,
            mut has_bias,
            ref transpose,
        } = prob;

        let (trans_lhs, trans_rhs, trans_out) = transpose.clone().into();
        if trans_out {
            has_bias = false;
        }

        let a_shape = if trans_lhs {
            shape![B, K, M]
        } else {
            shape![B, M, K]
        };

        let b_shape = if trans_rhs {
            shape![B, N, K]
        } else {
            shape![B, K, N]
        };

        let bias = if has_bias {
            Some(Tensor::randn::<f32>(shape![N], cpu_device.clone()))
        } else {
            None
        };
        println!("A shape: {:?}", a_shape);
        println!("B shape: {:?}", b_shape);
        println!("Bias: {:?}", bias.as_ref().map(|b| b.shape()));

        let a = Tensor::randn::<f32>(a_shape, cpu_device.clone());
        let b = Tensor::randn::<f32>(b_shape, cpu_device.clone());
        let ground = ground_truth(&a, &b, bias.as_ref(), trans_lhs, trans_rhs, trans_out)?;
        println!("Ground shape: {:?}", ground.shape());

        let a_gpu = a.to(device)?;
        let b_gpu = b.to(device)?;
        let bias_gpu = bias.as_ref().map(|b| b.to(device)).transpose()?;
        let c_gpu = a_gpu
            .gemm(b_gpu, bias_gpu, trans_lhs, trans_rhs, trans_out)?
            .resolve()?;

        let d_gpu = c_gpu.to(&Device::CPU)?;
        println!("RATCHET SGEMM\n{:?}\n", d_gpu);
        println!("PYTORCH FP32:\n{:?}", ground);
        ground.all_close(&d_gpu, 1e-4, 1e-4)?;
        Ok(())
    }

    #[test]
    fn test_qgemm() -> anyhow::Result<()> {
        let device = GPU_DEVICE.with(|d| d.clone());
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let a = Tensor::randn::<f32>(shape![6, 1500, 64], cpu_device.clone());
        let b = Tensor::randn::<f32>(shape![6, 64, 1500], cpu_device.clone());
        let ground = ground_truth(&a, &b, None, false, false, false)?;

        let quantizer = Quantizer::new(Quantization::SInt8);
        let aq = quantizer.sint8_quantize(a);
        let a_gpu = aq.to(&device)?;
        let b_gpu = b.to(&device)?;
        let c_gpu = a_gpu.matmul(b_gpu, false, false)?.resolve()?;
        let ours = c_gpu.to(&Device::CPU)?;

        println!("RATCHET QUANT\n{:?}\n", ours);
        println!("PYTORCH FP32:\n{:?}", ground);

        ground.all_close(&ours, 1e1, 1e-1)?;

        Ok(())
    }

    /*
         *Running sgemm: B=2 M=144 N=48 K=20 has_bias=true transpose=RHS
    A shape: [2x144x20]
    B shape: [2x48x20]
    Bias: Some([48])
    Ground shape: [2x144x48]
    RATCHET SGEMM
    Tensor { id: T33, shape: [2x144x48], dt: F32, op: Const, storage: Some("[-6.192138, 5.2859936, 15.683312, 10.934153, -9.986342, 0.1486624, 2.7134056, 4.9786434, -8.2089205, -0.54307663, -11.632709, 1.3128983, 9.759326, -1.2242645, 2.088686, 2.0799613, 1.219845, -9.854678, 9.500938, 4.369473, -4.144409, 0.84077555, -0.49583006, 0.3056485, -0.26782158, -1.2021586, -0.24937314, -0.0077694342, 1.2242687, -0.9283801, 0.5450683, -0.17495869, -0.81706333, -2.2411058, 0.5305886, -0.6256241, -1.0573441, -2.4417546, 0.5115216, 0.5458218, -0.50746536, -1.6070161, 0.89670783, 0.29370007, -0.5909658, 0.17349091, -0.13022095, 0.45880267, -1.9249877, 3.6802177, -2.3275533, 6.5063677, 2.6182313, 8.986089, 1.8303605, 0.7365761, -0.10881066, -3.256226, 4.3280907, 0.3745324, -0.16945879, 0.44012117, 9.3545265, 4.2577667] ... [-0.81706333, -2.2411058, 0.5305886, -0.6256241, -1.0573441, -2.4417546, 0.5115216, 0.5458218, -0.50746536, -1.6070161, 0.89670783, 0.29370007, -0.5909658, 0.17349091, -0.13022095, 0.45880267, 0.56590044, -1.025188, 0.97912395, 0.5011347, -0.4318299, 0.24531859, -0.4338185, -0.8967798, 3.026456, -0.8987544, -2.186475, 0.5457448, -0.15398388, -2.3198574, 0.18968439, -0.71898377, 0.72424525, -0.7855232, -1.3228257, 0.0014403614, -4.144409, 0.84077555, -0.49583006, 0.3056485, -0.26782158, -1.2021586, -0.24937314, -0.0077694342, 1.2242687, -0.9283801, 0.5450683, -0.17495869, -0.81706333, -2.2411058, 0.5305886, -0.6256241, -1.0573441, -2.4417546, 0.5115216, 0.5458218, -0.50746536, -1.6070161, 0.89670783, 0.29370007, -0.5909658, 0.17349091, -0.13022095, 0.45880267]") }

    PYTORCH FP32:
    Tensor { id: T28, shape: [2x144x48], dt: F32, op: Const, storage: Some("[2.0746732, -1.5678979, 0.39557886, 4.873418, -5.257111, -6.3939295, 8.211104, 6.1635447, 3.3015025, 0.20391542, 0.17722225, 0.28534043, 1.9187872, 1.7748964, -6.3771772, -1.4675636, 7.4864717, 3.4863303, 0.9503579, -2.802917, -7.1844664, 3.8913846, -8.901279, 0.5297334, 6.483171, -5.097629, 1.2435904, -7.8922114, -3.2591894, 2.5727148, -0.12980556, 1.755307, -0.31242073, -2.7295804, 2.7777462, -3.05021, 2.8549528, -3.4246676, -0.28530437, 0.2476727, -3.6427648, -0.7217546, 1.603257, 2.6147892, 1.109592, -1.7375972, 2.4050972, -0.766258, 2.5675044, 4.0622864, 1.5349252, -0.76998216, 3.5102494, -8.597033, 2.0128114, -7.589532, 10.014074, -5.1309433, -11.969616, 8.625012, 0.32394457, 2.8099582, -6.788409, -1.0087457] ... [-2.9889612, -3.714323, -1.8129258, 4.151071, -4.8124704, -5.3405666, 0.98362005, 0.14711213, -4.696717, 3.4926739, 0.22852325, -7.337511, 1.2199502, 8.830381, -1.1150132, 6.558776, 4.4053016, 0.09638333, -7.4392586, 2.98313, 0.45246473, 0.48678058, 2.6349897, 3.1246035, -0.65857816, 6.773018, -1.9281363, -0.630875, 1.8154215, -6.872533, -3.241759, -1.5071366, -2.0275948, 4.775486, -0.2407763, 6.930494, -9.384127, -0.643037, -6.331457, 3.1488688, 4.353603, 0.7344209, 1.5611992, -6.466379, -0.15513253, 0.49996656, 4.6645308, 3.0692585, 2.4736497, 1.500803, 0.17227167, 2.825548, -4.346708, 1.3864045, -4.4530373, 2.0878115, -4.9559193, -1.8951969, 7.9504848, 3.2946389, -2.9681816, 0.2985428, 2.533373, 4.552261]") }
         */

    #[test]
    fn debug_generated_gemm() -> anyhow::Result<()> {
        let prob = SGEMMProblem {
            B: 2,
            M: 144,
            N: 48,
            K: 20,
            has_bias: true,
            transpose: TransKind::RHS,
        };

        let SGEMMProblem {
            B,
            M,
            K,
            N,
            has_bias,
            ref transpose,
        } = prob;

        println!(
            "Running sgemm: B={} M={} N={} K={} has_bias={} transpose={:?}",
            B, M, N, K, has_bias, transpose
        );
        let device = GPU_DEVICE.with(|d| d.clone());
        run_matmul_trial(&device, prob)
    }

    #[test]
    fn debug_gemm() -> anyhow::Result<()> {
        let _ = env_logger::builder().is_test(true).try_init();
        let device = GPU_DEVICE.with(|d| d.clone());
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let a = Tensor::randn::<f32>(shape![2, 222, 252], cpu_device.clone());
        let b = Tensor::randn::<f32>(shape![1, 222, 238], cpu_device.clone());
        let bias = Some(Tensor::randn::<f32>(shape![238], cpu_device.clone()));

        let TRANS_LHS = true;
        let TRANS_RHS = false;
        let TRANS_OUT = false;
        let QUANT = false;

        let ground = ground_truth(&a, &b, bias.as_ref(), TRANS_LHS, TRANS_RHS, TRANS_OUT)?;

        let a_gpu = if QUANT {
            let quantizer = Quantizer::new(Quantization::SInt8);
            let aq = quantizer.sint8_quantize(a);
            aq.to(&device)?
        } else {
            a.to(&device)?
        };

        let b_gpu = b.to(&device)?;
        let bias_gpu = bias.as_ref().map(|b| b.to(&device)).transpose()?;
        let c_gpu = a_gpu
            .gemm(b_gpu, bias_gpu, TRANS_LHS, TRANS_RHS, TRANS_OUT)?
            .resolve()?;
        let ours = c_gpu.to(&Device::CPU)?;

        println!("RATCHET\n{:?}\n", ours.to_ndarray_view::<f32>());
        println!("PYTORCH:\n{:?}", ground.to_ndarray_view::<f32>());

        ground.all_close(&ours, 1e-3, 1e-3)?;
        Ok(())
    }
}
