mod gemm;
mod quantized;
mod subgroup_gemv;
mod workgroup_gemv;

pub use gemm::*;
pub use quantized::*;
pub use subgroup_gemv::*;
pub use workgroup_gemv::*;

use std::{cmp::Ordering, mem};

use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform},
    rvec, DType, Device, GPUOperation, Kernel, KernelElement, KernelKey, KernelMetadata,
    KernelRenderable, KernelSource, OpGuards, Operation, OperationError, RVec, Shape, StorageView,
    Strides, Tensor, WorkgroupSize, Workload, Q4_KF, Q4_KH, Q8_0F, Q8_0H,
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
    pub fn new(arows: usize, acols: usize) -> Self {
        match (arows, acols) {
            (arows, acols) if arows > acols * 4 => GEMVHeuristic::VeryTall,
            (arows, acols) if arows > acols * 2 => GEMVHeuristic::Tall,
            (arows, acols) if acols > arows * 2 => GEMVHeuristic::Fat,
            _ => GEMVHeuristic::Square,
        }
    }

    pub fn as_workgroup_size(&self) -> (usize, usize) {
        match self {
            GEMVHeuristic::Fat => (4, 256),
            _ => (8, 8),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct MatmulSpec {
    lhs_dt: DType,
    rhs_dt: DType,
    raw_lhs_shape: Shape,
    raw_rhs_shape: Shape,
    lhs_shape: Shape,
    rhs_shape: Shape,
    dst_shape: Shape,
    lhs_strides: Strides,
    rhs_strides: Strides,
    dst_strides: Strides,
    lhs_stack: usize,
    rhs_stack: usize,
    dst_stack: usize,
    trans_lhs: bool,
    trans_rhs: bool,
    trans_dst: bool,
    stack_shape: Shape, //N-D matmul is handled by stacking the first N-2 dimensions
    pub heuristic: GEMVHeuristic,
}

impl MatmulSpec {
    pub const ROW_PER_THREAD: usize = 4;
    pub const TILE_DIM: usize = 32;

    pub fn new(
        LHS: &Tensor,
        RHS: &Tensor,
        trans_lhs: bool,
        trans_rhs: bool,
        trans_dst: bool,
    ) -> Self {
        let raw_lhs_shape = LHS.shape().clone();
        let raw_rhs_shape = RHS.shape().clone();
        let mut lhs_shape = raw_lhs_shape.clone();
        let mut rhs_shape = raw_rhs_shape.clone();

        let mut trans_lhs = trans_lhs;
        let mut trans_rhs = trans_rhs;

        let lhs_dt = LHS.dt();
        let rhs_dt = RHS.dt();

        if (lhs_shape.rank() < 2) || (rhs_shape.rank() < 2) {
            panic!("MatMul: inputs must be at least 2D");
        }

        match lhs_shape.rank().cmp(&rhs_shape.rank()) {
            Ordering::Less => {
                lhs_shape.left_pad_to(1, rhs_shape.rank());
            }
            Ordering::Greater => {
                rhs_shape.left_pad_to(1, lhs_shape.rank());
            }
            _ => {}
        };

        let mut dst_shape =
            Matmul::compute_dst_shape(&lhs_shape, &rhs_shape, trans_lhs, trans_rhs, trans_dst)
                .unwrap();

        let stack_dims = dst_shape.rank() - 2;
        let stack_shape = dst_shape.slice(0..stack_dims);

        let lhs_stack = lhs_shape.drain(0..stack_dims).product();
        let rhs_stack = rhs_shape.drain(0..stack_dims).product();
        let dst_stack = dst_shape.drain(0..stack_dims).product();

        if lhs_stack != 1 && rhs_stack != 1 {
            //Here we want all of the stacks to be equal
            //OR A or B to be 1
            assert!(lhs_stack == rhs_stack && rhs_stack == dst_stack);
        }

        if lhs_shape.rank() == 1 {
            lhs_shape.insert(0, 1);
        }

        if rhs_shape.rank() == 1 {
            rhs_shape.insert(0, 1);
        }

        let mut lhs_strides = Strides::from(&lhs_shape);
        let mut rhs_strides = Strides::from(&rhs_shape);
        let dst_strides = Strides::from(&dst_shape);

        let is_cpu = matches!(LHS.device(), Device::CPU);

        if is_cpu {
            // The (a b)T => bT aT rule means that if we have to transpose dst we can simply transpose the inputs and swap them.
            // However two transposes cancel each other out, in which case we can just skip transposing the input altogether.
            // This is just the xor operator (^).
            if trans_lhs ^ trans_dst {
                lhs_shape.transpose();
                lhs_strides.transpose();
            }
            if trans_rhs ^ trans_dst {
                rhs_shape.transpose();
                rhs_strides.transpose();
            }
            if trans_dst {
                // (a b)T => bT aT
                // aT bT has already been applied correctly above, so we can just swap.
                mem::swap(&mut lhs_shape, &mut rhs_shape);
                // strides and transposes must follow their shapes
                mem::swap(&mut lhs_strides, &mut rhs_strides);
                mem::swap(&mut trans_lhs, &mut trans_rhs);
            }
        }

        log::debug!(
            "MatmulSpec stacking: lhs={lhs_shape:?} rhs={rhs_shape:?} stack_dims={stack_dims} stack_count={}",
            stack_shape.numel(),
        );

        let heuristic = GEMVHeuristic::new(rhs_shape[0], rhs_shape[1]);

        Self {
            lhs_dt,
            rhs_dt,
            raw_lhs_shape,
            raw_rhs_shape,
            lhs_shape,
            rhs_shape,
            dst_shape,
            lhs_strides,
            rhs_strides,
            dst_strides,
            lhs_stack,
            rhs_stack,
            dst_stack,
            trans_lhs,
            trans_rhs,
            trans_dst,
            stack_shape,
            heuristic,
        }
    }

    pub fn select_kernel_element(&self) -> KernelElement {
        if self.trans_lhs || self.trans_rhs || self.trans_dst || self.rhs_shape.is_vector() {
            //We cannot support transposed with vectorized kernels
            //If GEMV we use Scalar
            return KernelElement::Scalar;
        }

        let checks = [
            self.dim_inner(),
            self.dst_shape[1],
            self.lhs_shape.numel(),
            self.rhs_shape.numel(),
            self.dst_shape.numel(),
        ];

        if checks.iter().all(|&x| x % 4 == 0) {
            KernelElement::Vec4
        } else {
            KernelElement::Scalar
        }
    }

    pub fn raw_lhs_shape(&self) -> &Shape {
        &self.raw_lhs_shape
    }

    pub fn raw_rhs_shape(&self) -> &Shape {
        &self.raw_rhs_shape
    }

    pub fn lhs_shape(&self) -> &Shape {
        &self.lhs_shape
    }

    pub fn lhs_shape_batched(&self) -> Shape {
        let mut lhs_shape = self.lhs_shape.clone();
        lhs_shape.insert(0, self.lhs_stack());
        lhs_shape
    }

    pub fn rhs_shape_batched(&self) -> Shape {
        let mut rhs_shape = self.rhs_shape.clone();
        rhs_shape.insert(0, self.rhs_stack());
        rhs_shape
    }

    pub fn dst_shape_batched(&self) -> Shape {
        let mut dst_shape = self.dst_shape.clone();
        dst_shape.insert(0, self.stacks());
        dst_shape
    }

    pub fn rhs_shape(&self) -> &Shape {
        &self.rhs_shape
    }

    pub fn dst_shape(&self) -> &Shape {
        &self.dst_shape
    }

    pub fn lhs_strides(&self) -> &Strides {
        &self.lhs_strides
    }

    pub fn rhs_strides(&self) -> &Strides {
        &self.rhs_strides
    }

    pub fn dst_strides(&self) -> &Strides {
        &self.dst_strides
    }

    pub fn dim_lhs_outer(&self) -> usize {
        self.dst_shape[0]
    }
    pub fn dim_rhs_outer(&self) -> usize {
        self.dst_shape[1]
    }

    pub fn new_dim_lhs_outer(&self) -> usize {
        if self.trans_dst {
            self.dst_shape[1]
        } else {
            self.dst_shape[0]
        }
    }

    pub fn new_dim_rhs_outer(&self) -> usize {
        if self.trans_rhs {
            self.rhs_shape[1]
        } else {
            self.rhs_shape[0]
        }
    }

    // Returns M dimension before transpose
    pub fn raw_m(&self) -> usize {
        self.raw_lhs_shape[0]
    }

    // Returns N dimension before transpose
    pub fn raw_n(&self) -> usize {
        self.raw_rhs_shape[1]
    }

    /// Returns K dimension before transpose
    pub fn raw_k(&self) -> usize {
        assert_eq!(self.raw_lhs_shape[1], self.raw_rhs_shape[0]);
        self.raw_rhs_shape[0]
    }

    /// Returns M dimension after transpose
    pub fn m(&self) -> usize {
        self.lhs_shape[0]
    }

    /// Returns N dimension after transpose
    pub fn n(&self) -> usize {
        self.rhs_shape[1]
    }

    /// Returns K dimension after transpose
    pub fn k(&self) -> usize {
        assert_eq!(self.lhs_shape[1], self.rhs_shape[0]);
        self.rhs_shape[0]
    }

    pub fn dim_inner(&self) -> usize {
        if self.trans_lhs {
            self.lhs_shape[0]
        } else {
            self.lhs_shape[1]
        }
    }

    pub fn lhs_stack(&self) -> usize {
        self.lhs_stack
    }

    pub fn rhs_stack(&self) -> usize {
        self.rhs_stack
    }

    pub fn dst_stack(&self) -> usize {
        self.dst_stack
    }

    pub fn trans_lhs(&self) -> bool {
        self.trans_lhs
    }

    pub fn trans_rhs(&self) -> bool {
        self.trans_rhs
    }

    pub fn trans_dst(&self) -> bool {
        self.trans_dst
    }

    pub fn stacks(&self) -> usize {
        self.stack_shape.numel()
    }

    pub fn rhs_dt(&self) -> DType {
        self.rhs_dt
    }

    pub fn is_gemv(&self) -> bool {
        self.rhs_shape.is_vector() && !self.trans_lhs
    }

    pub fn stacked_shapes(&self) -> (Shape, Shape, Shape) {
        let mut lhs_shape = self.lhs_shape.clone();
        let mut rhs_shape = self.rhs_shape.clone();
        let mut dst_shape = self.dst_shape.clone();
        lhs_shape.insert(0, self.stacks());
        rhs_shape.insert(0, self.stacks());
        dst_shape.insert(0, self.stacks());
        (lhs_shape, rhs_shape, dst_shape)
    }

    pub fn tile_fit(&self) -> (bool, bool, bool) {
        let dim_lhs_outer = self.dim_lhs_outer();
        let dim_rhs_outer = self.dim_rhs_outer();
        let dim_inner = self.dim_inner();

        let lhs_fit = dim_lhs_outer % Self::TILE_DIM == 0;
        let rhs_fit = dim_rhs_outer % Self::TILE_DIM == 0;
        let dst_fit = dim_inner % Self::TILE_DIM == 0;
        (lhs_fit, rhs_fit, dst_fit)
    }
}

#[derive(derive_new::new, Debug, Clone)]
pub struct Matmul {
    pub(crate) lhs: Tensor,
    pub(crate) rhs: Tensor,
    pub(crate) bias: Option<Tensor>,
    pub(crate) trans_lhs: bool,
    pub(crate) trans_rhs: bool,
    pub(crate) trans_dst: bool,
}

impl Matmul {
    pub fn compute_dst_shape(
        lhs_shape: &Shape,
        rhs_shape: &Shape,
        trans_lhs: bool,
        trans_rhs: bool,
        trans_dst: bool,
    ) -> anyhow::Result<Shape> {
        let mut lhs_shape = lhs_shape.clone();
        let mut rhs_shape = rhs_shape.clone();

        let implicit_m = lhs_shape.rank() < 2;
        let implicit_n = rhs_shape.rank() < 2;
        if implicit_m {
            lhs_shape.insert(trans_lhs as usize, 1);
        }
        if implicit_n {
            rhs_shape.insert(!trans_rhs as usize, 1);
        }

        let equalize_rank = |shape: &mut Shape, target_rank: usize| {
            while shape.rank() < target_rank {
                shape.insert(0, 1);
            }
        };
        equalize_rank(&mut lhs_shape, rhs_shape.rank());
        equalize_rank(&mut rhs_shape, lhs_shape.rank());

        let lhs_rank = lhs_shape.rank();
        let rhs_rank = rhs_shape.rank();
        let (lhs_prefix, rhs_prefix) = (&lhs_shape[..lhs_rank - 2], &rhs_shape[..rhs_rank - 2]);
        let dst_broadcasted_prefix =
            Shape::multi_broadcast(&[&lhs_prefix.into(), &rhs_prefix.into()]).ok_or_else(|| {
                anyhow::anyhow!(
                    "Matmul broadcasting: a: {:?} b: {:?} trans_a: {:?} trans_b: {:?}",
                    lhs_shape,
                    rhs_shape,
                    trans_lhs,
                    trans_rhs
                )
            })?;

        let (mut m, mut kl) = (lhs_shape[lhs_rank - 2], lhs_shape[lhs_rank - 1]);
        let (mut kr, mut n) = (rhs_shape[rhs_rank - 2], rhs_shape[rhs_rank - 1]);

        if trans_lhs {
            std::mem::swap(&mut m, &mut kl);
        }

        if trans_rhs {
            std::mem::swap(&mut kr, &mut n);
        }

        if kl != kr {
            anyhow::bail!("Matmul broadcasting: ka != kb: {} != {}", kl, kr);
        }

        let mut dst_shape_final = dst_broadcasted_prefix.clone();
        if trans_dst {
            if !implicit_n {
                dst_shape_final.push(n);
            }
            if !implicit_m {
                dst_shape_final.push(m);
            }
        } else {
            if !implicit_m {
                dst_shape_final.push(m);
            }
            if !implicit_n {
                dst_shape_final.push(n);
            }
        }

        Ok(dst_shape_final)
    }

    pub fn compute_spec(&self) -> MatmulSpec {
        MatmulSpec::new(
            &self.lhs,
            &self.rhs,
            self.trans_lhs,
            self.trans_rhs,
            self.trans_dst,
        )
    }
}

impl Operation for Matmul {
    fn name(&self) -> &'static str {
        "Matmul"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let dst_shape = Matmul::compute_dst_shape(
            self.lhs.shape(),
            self.rhs.shape(),
            self.trans_lhs,
            self.trans_rhs,
            self.trans_dst,
        )
        .unwrap();
        let dst_strides = Strides::from(&dst_shape);
        Ok(StorageView::new(dst_shape, self.rhs.dt(), dst_strides))
    }

    fn srcs(&self) -> RVec<&Tensor> {
        if let Some(bias) = &self.bias {
            rvec![&self.lhs, &self.rhs, bias]
        } else {
            rvec![&self.lhs, &self.rhs]
        }
    }
}

impl OpGuards for Matmul {
    fn check_shapes(&self) {
        let dst_shape = Matmul::compute_dst_shape(
            self.lhs.shape(),
            self.rhs.shape(),
            self.trans_lhs,
            self.trans_rhs,
            self.trans_dst,
        );
        assert!(dst_shape.is_ok());
    }

    fn check_dtypes(&self) {
        let allowed_pairs = [
            (DType::F32, DType::F32),
            (DType::F16, DType::F16),
            (DType::Q8_0F(Q8_0F::default()), DType::F32),
            (DType::Q8_0H(Q8_0H::default()), DType::F16),
            (DType::Q4_KF(Q4_KF::default()), DType::F32),
            (DType::Q4_KH(Q4_KH::default()), DType::F16),
        ];

        if !allowed_pairs.contains(&(self.lhs.dt(), self.rhs.dt())) {
            panic!(
                "DType mismatch: lhs: {:?}, rhs: {:?}",
                self.lhs.dt(),
                self.rhs.dt()
            );
        }

        if let Some(bias) = &self.bias {
            if bias.dt() != self.rhs.dt() {
                panic!(
                    "DType mismatch: bias: {:?}, rhs: {:?}",
                    bias.dt(),
                    self.rhs.dt()
                );
            }
        }
    }
}

/// Encapsulate all metadata structs for underlying kernels
pub enum MatmulMeta {
    GEMMMeta(GEMMMeta),
    SubgroupGEMVMeta(SubgroupGEMVMeta),
    WorkgroupGEMVMeta(WorkgroupGEMVMeta),
    Quantized(QuantizedMeta),
}

impl KernelMetadata for MatmulMeta {
    fn render_meta(&self) -> crate::WgslFragment {
        match self {
            MatmulMeta::GEMMMeta(meta) => meta.render_meta(),
            MatmulMeta::SubgroupGEMVMeta(meta) => meta.render_meta(),
            MatmulMeta::WorkgroupGEMVMeta(meta) => meta.render_meta(),
            MatmulMeta::Quantized(meta) => meta.render_meta(),
        }
    }

    fn write(&self, uniform: &mut CpuUniform) -> Result<u64, OperationError> {
        match self {
            MatmulMeta::GEMMMeta(meta) => meta.write(uniform),
            MatmulMeta::SubgroupGEMVMeta(meta) => meta.write(uniform),
            MatmulMeta::WorkgroupGEMVMeta(meta) => meta.write(uniform),
            MatmulMeta::Quantized(meta) => meta.write(uniform),
        }
    }
}

pub enum MatmulKernels {
    GEMM(GEMM),
    SubgroupGEMV(SubgroupGEMV),
    WorkgroupGEMV(WorkgroupGEMV),
    Quantized(Quantized),
}

impl KernelRenderable for MatmulKernels {
    fn register_bindings<P: crate::WgslPrimitive>(
        &self,
        builder: &mut crate::WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError> {
        match self {
            MatmulKernels::GEMM(kernel) => kernel.register_bindings::<P>(builder, inplace),
            MatmulKernels::SubgroupGEMV(kernel) => kernel.register_bindings::<P>(builder, inplace),
            MatmulKernels::WorkgroupGEMV(kernel) => kernel.register_bindings::<P>(builder, inplace),
            MatmulKernels::Quantized(kernel) => kernel.register_bindings::<P>(builder, inplace),
        }
    }

    fn render<P: crate::WgslPrimitive>(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        match self {
            MatmulKernels::GEMM(k) => k.render::<P>(inplace, dst, workgroup_size),
            MatmulKernels::SubgroupGEMV(k) => k.render::<P>(inplace, dst, workgroup_size),
            MatmulKernels::WorkgroupGEMV(k) => k.render::<P>(inplace, dst, workgroup_size),
            MatmulKernels::Quantized(k) => k.render::<P>(inplace, dst, workgroup_size),
        }
    }
}

/// Defer down to kernel level
impl Kernel for MatmulKernels {
    type Metadata = MatmulMeta;

    fn kernel_key(
        &self,
        workgroup_size: &WorkgroupSize,
        inplace: bool,
        srcs: &[&Tensor],
        dst: &Tensor,
        kernel_element: &KernelElement,
    ) -> KernelKey {
        match self {
            MatmulKernels::GEMM(kernel) => {
                kernel.kernel_key(workgroup_size, inplace, srcs, dst, kernel_element)
            }
            MatmulKernels::SubgroupGEMV(kernel) => {
                kernel.kernel_key(workgroup_size, inplace, srcs, dst, kernel_element)
            }
            MatmulKernels::WorkgroupGEMV(kernel) => {
                kernel.kernel_key(workgroup_size, inplace, srcs, dst, kernel_element)
            }
            MatmulKernels::Quantized(kernel) => {
                kernel.kernel_key(workgroup_size, inplace, srcs, dst, kernel_element)
            }
        }
    }

    fn kernel_name(&self) -> String {
        match self {
            MatmulKernels::GEMM(kernel) => kernel.kernel_name(),
            MatmulKernels::SubgroupGEMV(kernel) => kernel.kernel_name(),
            MatmulKernels::WorkgroupGEMV(kernel) => kernel.kernel_name(),
            MatmulKernels::Quantized(kernel) => kernel.kernel_name(),
        }
    }

    fn metadata(
        &self,
        dst: &Tensor,
        kernel_element: &KernelElement,
    ) -> Result<Self::Metadata, OperationError> {
        match self {
            MatmulKernels::GEMM(k) => Ok(MatmulMeta::GEMMMeta(k.metadata(dst, kernel_element)?)),
            MatmulKernels::SubgroupGEMV(k) => Ok(MatmulMeta::SubgroupGEMVMeta(
                k.metadata(dst, kernel_element)?,
            )),
            MatmulKernels::WorkgroupGEMV(k) => Ok(MatmulMeta::WorkgroupGEMVMeta(
                k.metadata(dst, kernel_element)?,
            )),
            MatmulKernels::Quantized(k) => {
                Ok(MatmulMeta::Quantized(k.metadata(dst, kernel_element)?))
            }
        }
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError> {
        match self {
            MatmulKernels::GEMM(kernel) => kernel.calculate_dispatch(dst),
            MatmulKernels::SubgroupGEMV(kernel) => kernel.calculate_dispatch(dst),
            MatmulKernels::WorkgroupGEMV(kernel) => kernel.calculate_dispatch(dst),
            MatmulKernels::Quantized(kernel) => kernel.calculate_dispatch(dst),
        }
    }

    fn kernel_element(&self, dst: &Tensor) -> KernelElement {
        match self {
            MatmulKernels::GEMM(kernel) => kernel.kernel_element(dst),
            MatmulKernels::SubgroupGEMV(kernel) => kernel.kernel_element(dst),
            MatmulKernels::WorkgroupGEMV(kernel) => kernel.kernel_element(dst),
            MatmulKernels::Quantized(kernel) => kernel.kernel_element(dst),
        }
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        match self {
            MatmulKernels::GEMM(kernel) => kernel.build_kernel(inplace, dst, workgroup_size),
            MatmulKernels::SubgroupGEMV(k) => k.build_kernel(inplace, dst, workgroup_size),
            MatmulKernels::WorkgroupGEMV(k) => k.build_kernel(inplace, dst, workgroup_size),
            MatmulKernels::Quantized(kernel) => kernel.build_kernel(inplace, dst, workgroup_size),
        }
    }

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        match self {
            MatmulKernels::GEMM(kernel) => kernel.storage_bind_group_layout(inplace),
            MatmulKernels::SubgroupGEMV(kernel) => kernel.storage_bind_group_layout(inplace),
            MatmulKernels::WorkgroupGEMV(kernel) => kernel.storage_bind_group_layout(inplace),
            MatmulKernels::Quantized(kernel) => kernel.storage_bind_group_layout(inplace),
        }
    }
}

impl GPUOperation for Matmul {
    type KernelEnum = MatmulKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        if !self.bias.as_ref().map_or(true, |b| b.shape().is_vector()) {
            panic!("Bias must be a vector: {:?}", self.bias);
        }

        if self.lhs.dt().is_quantized() && self.trans_lhs {
            panic!("Transposed quantized inputs are not supported");
        }

        let is_gemv = self.rhs.shape().is_vector() && !self.trans_lhs;
        let is_q4 = self.lhs.dt().is_q4();
        let supports_subgroup = self
            .lhs
            .device()
            .try_gpu()
            .unwrap()
            .compute_features()
            .SUBGROUP;

        let spec = self.compute_spec();

        match (is_gemv, is_q4, supports_subgroup) {
            (true, false, true) => {
                MatmulKernels::SubgroupGEMV(SubgroupGEMV::from_matmul(self, spec))
            }
            (true, false, false) => {
                MatmulKernels::WorkgroupGEMV(WorkgroupGEMV::from_matmul(self, spec))
            }
            (false, true, _) => MatmulKernels::Quantized(Quantized::from_matmul(self, spec)),
            (false, false, _) => MatmulKernels::GEMM(GEMM::from_matmul(self, spec)),
            _ => todo!(),
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::test_util::run_py_prg;

    use crate::{shape, Device, DeviceRequest, Quantization, Quantizer};

    use super::*;

    fn ground_truth(
        a: &Tensor,
        b: &Tensor,
        bias: Option<&Tensor>,
        trans_lhs: bool,
        trans_rhs: bool,
        trans_dst: bool,
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

        let result_op = if trans_dst {
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
        None,
        LHS,
        LHSAndDST,
        RHS,
        RHSAndDST,
        DST,
    }

    impl From<TransKind> for (bool, bool, bool) {
        fn from(val: TransKind) -> Self {
            match val {
                TransKind::None => (false, false, false),
                TransKind::LHS => (true, false, false),
                TransKind::LHSAndDST => (true, false, true),
                TransKind::RHS => (false, true, false),
                TransKind::RHSAndDST => (false, true, true),
                TransKind::DST => (false, false, true),
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
    fn test_sgemm_gpu(prob: SGEMMProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
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

    #[proptest(cases = 64)]
    fn test_sgemm_cpu(prob: SGEMMProblem) {
        let device = Device::request_device(DeviceRequest::CPU).unwrap();
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

        let (trans_lhs, trans_rhs, trans_dst) = transpose.clone().into();
        if trans_dst {
            has_bias = false;
        }
        has_bias = false;

        let lhs_shape = if trans_lhs {
            shape![B, K, M]
        } else {
            shape![B, M, K]
        };

        let rhs_shape = if trans_rhs {
            shape![B, N, K]
        } else {
            shape![B, K, N]
        };

        let bias = if has_bias {
            Some(Tensor::randn::<f32>(shape![N], cpu_device.clone()))
        } else {
            None
        };

        println!("LHS shape: {:?}", lhs_shape);
        println!("RHS shape: {:?}", rhs_shape);
        println!("Bias: {:?}", bias.as_ref().map(|b| b.shape()));

        let a = Tensor::randn::<f32>(lhs_shape, cpu_device.clone());
        let b = Tensor::randn::<f32>(rhs_shape, cpu_device.clone());
        let ground = ground_truth(&a, &b, bias.as_ref(), trans_lhs, trans_rhs, trans_dst)?;
        println!("Ground shape: {:?}", ground.shape());

        let a_gpu = a.to(device)?;
        let b_gpu = b.to(device)?;
        let bias_gpu = bias.as_ref().map(|b| b.to(device)).transpose()?;
        let c_gpu = a_gpu
            .gemm(b_gpu, bias_gpu, trans_lhs, trans_rhs, trans_dst)?
            .resolve()?;

        let d_gpu = c_gpu.to(&Device::CPU)?;
        println!("RATCHET SGEMM\n{:?}\n", d_gpu);
        println!("PYTORCH FP32:\n{:?}", ground);

        ground.all_close(&d_gpu, 1e-4, 1e-4)?;
        Ok(())
    }

    #[test]
    fn test_qgemm() -> anyhow::Result<()> {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
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

    #[test]
    fn debug_gemm() -> anyhow::Result<()> {
        let _ = env_logger::builder().is_test(true).try_init();

        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let a = Tensor::randn::<f32>(shape![2, 175, 241], cpu_device.clone());
        let b = Tensor::randn::<f32>(shape![2, 241, 182], cpu_device.clone());
        let bias = Some(Tensor::randn::<f32>(shape![182], cpu_device.clone()));

        let TRANS_LHS = false;
        let TRANS_RHS = false;
        let TRANS_DST = false;
        let QUANT = false;

        let ground = ground_truth(&a, &b, bias.as_ref(), TRANS_LHS, TRANS_RHS, TRANS_DST)?;

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
            .gemm(b_gpu, bias_gpu, TRANS_LHS, TRANS_RHS, TRANS_DST)?
            .resolve()?;
        let ours = c_gpu.to(&Device::CPU)?;

        println!("RATCHET\n{:?}\n", ours.to_ndarray_view::<f32>());
        println!("PYTORCH:\n{:?}", ground.to_ndarray_view::<f32>());

        ground.all_close(&ours, 1e-3, 1e-3)?;
        Ok(())
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn debug_gemv() -> anyhow::Result<()> {
        let _ = env_logger::builder().is_test(true).try_init();

        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let a = Tensor::randn::<f32>(shape![1, 51865, 384], cpu_device.clone());
        let b = Tensor::randn::<f32>(shape![1, 1, 384], cpu_device.clone());

        let TRANS_LHS = false;
        let TRANS_RHS = true;
        let TRANS_DST = true;
        let QUANT = false;

        let ground = ground_truth(&a, &b, None, TRANS_LHS, TRANS_RHS, TRANS_DST)?;

        let a_gpu = if QUANT {
            let quantizer = Quantizer::new(Quantization::SInt8);
            let aq = quantizer.sint8_quantize(a);
            aq.to(&device)?
        } else {
            a.to(&device)?
        };

        let b_gpu = b.to(&device)?;
        let c_gpu = a_gpu
            .gemm(b_gpu, None, TRANS_LHS, TRANS_RHS, TRANS_DST)?
            .resolve()?;
        let ours = c_gpu.to(&Device::CPU)?;

        println!("RATCHET\n{:?}\n", ours.to_ndarray_view::<f32>());
        println!("PYTORCH:\n{:?}", ground.to_ndarray_view::<f32>());

        ground.all_close(&ours, 1e-3, 1e-3)?;
        Ok(())
    }
}
