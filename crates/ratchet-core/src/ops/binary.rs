use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use ratchet_macros::WgslMetadata;

use crate::{
    gpu::{dtype::WgslDType, BindGroupLayoutDescriptor, CpuUniform},
    rvec, Array, BindingMode, BuiltIn, DType, InvariantError, KernelElement, KernelSource,
    MetaOperation, OpGuards, Operation, OperationError, RVec, Scalar, Shape, StorageView, Strides,
    Tensor, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
};
#[cfg(test)]
use test_strategy::Arbitrary;

#[cfg_attr(test, derive(Arbitrary))]
#[derive(Debug, Clone)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinaryOp {
    pub fn kernel_name(&self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
        }
    }

    pub fn kernel_operator(&self) -> &'static str {
        match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
        }
    }
}

#[derive(new, Debug, Clone)]
pub struct Binary {
    lhs: Tensor,
    rhs: Tensor,
    op: BinaryOp,
}

impl Binary {
    pub fn op(&self) -> &BinaryOp {
        &self.op
    }

    fn build_binary<P: WgslPrimitive>(
        &self,
        inplace: bool,
        _: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let device = self.lhs.device().try_gpu().unwrap();
        let mut kernel_builder = WgslKernelBuilder::new(
            workgroup_size.clone(),
            rvec![
                BuiltIn::WorkgroupId,
                BuiltIn::LocalInvocationIndex,
                BuiltIn::NumWorkgroups
            ],
            device.compute_features().clone(),
        );

        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        kernel_builder.write_metadata::<BinaryMeta>();

        let N = (P::W as u32).render();

        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let index = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (index >= metadata.numel / 'N) {
                return;
            }
        });

        let op = self.op.kernel_operator();
        let apply = if inplace {
            wgsl! {
                let val = A[index];
                A[index] = val 'op B[index];
            }
        } else {
            wgsl! { Y[index] = A[index] 'op B[index]; }
        };
        kernel_builder.write_main(apply);
        Ok(kernel_builder.build()?)
    }

    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError> {
        if inplace {
            builder.register_storage("A", BindingMode::ReadWrite, Array::<P>::default());
            builder.register_storage("B", BindingMode::ReadOnly, Array::<P>::default());
        } else {
            builder.register_storage("A", BindingMode::ReadOnly, Array::<P>::default());
            builder.register_storage("B", BindingMode::ReadOnly, Array::<P>::default());
            builder.register_storage("Y", BindingMode::ReadWrite, Array::<P>::default());
        }
        builder.register_uniform();
        Ok(())
    }
}

#[derive(Debug, ShaderType, WgslMetadata)]
pub struct BinaryMeta {
    numel: u32,
}

impl OpGuards for Binary {
    fn check_shapes(&self) {
        let shapes = [self.lhs.shape(), self.rhs.shape()];
        let broadcasted = Shape::multi_broadcast(&shapes);
        assert!(broadcasted.is_some());
    }

    fn check_dtypes(&self) {
        assert_eq!(self.lhs.dt(), self.rhs.dt());
    }
}

impl Operation for Binary {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let lhs = &self.lhs;
        let rhs = &self.rhs;
        let shapes = &[lhs.shape(), rhs.shape()];
        if lhs.is_scalar() || rhs.is_scalar() {
            let other = if lhs.is_scalar() { rhs } else { lhs };
            return Ok(other.storage_view().clone());
        }
        let broadcasted = Shape::multi_broadcast(shapes);
        if broadcasted.is_none() {
            let failed = shapes.iter().map(|s| (*s).clone()).collect::<Vec<_>>();
            return Err(InvariantError::BroadcastingFailed(failed).into());
        }
        let broadcasted = broadcasted.unwrap();
        let ostrides = Strides::from(&broadcasted);
        Ok(StorageView::new(broadcasted, lhs.dt(), ostrides))
    }
}

impl MetaOperation for Binary {
    fn kernel_name(&self) -> String {
        self.op.kernel_name().to_string()
    }

    fn supports_inplace(&self) -> bool {
        true
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.lhs, &self.rhs]
    }

    fn kernel_element(&self, dst: &Tensor) -> KernelElement {
        let numel = dst.shape().numel();

        if numel % 4 == 0 {
            KernelElement::Vec4
        } else if numel % 2 == 0 {
            KernelElement::Vec2
        } else {
            KernelElement::Scalar
        }
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError> {
        Ok(Workload::std(dst.shape().numel(), self.kernel_element(dst)))
    }

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if inplace {
            Ok(BindGroupLayoutDescriptor::binary_inplace())
        } else {
            Ok(BindGroupLayoutDescriptor::binary())
        }
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        dst: &Tensor,
        _kernel_element: &KernelElement,
    ) -> Result<u64, OperationError> {
        let numel = dst.shape().numel() as _;
        let meta = BinaryMeta { numel };
        Ok(uniform.write(&meta)?)
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = self.kernel_element(dst);
        match (self.lhs.dt(), &kernel_element) {
            (DType::F32, KernelElement::Scalar) => {
                self.build_binary::<Scalar<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec2) => {
                self.build_binary::<Vec2<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec4) => {
                self.build_binary::<Vec4<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Scalar) => {
                self.build_binary::<Scalar<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec2) => {
                self.build_binary::<Vec2<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec4) => {
                self.build_binary::<Vec4<f16>>(inplace, dst, workgroup_size)
            }
            _ => Err(OperationError::CompileError(format!(
                "Unsupported dtype {:?} or kernel element {:?}",
                self.lhs.dt(),
                kernel_element
            ))),
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use crate::{test_util::run_py_prg, BinaryOp, Device, DeviceRequest, Shape, Tensor};
    use test_strategy::{proptest, Arbitrary};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct BinaryProblem {
        op: BinaryOp,
        #[any(vec![1..=4, 1..=4, 1..=1, 1..=256])]
        shape: Shape,
    }

    fn ground_truth(a: &Tensor, b: &Tensor, op: &BinaryOp) -> anyhow::Result<Tensor> {
        let kn = op.kernel_name();
        let prg = format!(
            r#"
import torch
def {}(a, b):
    return torch.{}(torch.from_numpy(a), torch.from_numpy(b)).numpy()
"#,
            kn, kn
        );
        run_py_prg(prg.to_string(), &[a, b], &[], a.dt())
    }

    fn run_binary_trial(prob: BinaryProblem) -> anyhow::Result<()> {
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let BinaryProblem { op, shape } = prob;
        let a = Tensor::randn::<f32>(shape.clone(), cpu_device.clone());
        let b = Tensor::randn::<f32>(shape, cpu_device.clone());
        let ground = ground_truth(&a, &b, &op)?;
        let device = GPU_DEVICE.with(|d| d.clone());

        let a_gpu = a.to(&device)?;
        let b_gpu = b.to(&device)?;
        let c_gpu = match op {
            BinaryOp::Add => a_gpu.add(b_gpu)?,
            BinaryOp::Sub => a_gpu.sub(b_gpu)?,
            BinaryOp::Mul => a_gpu.mul(b_gpu)?,
            BinaryOp::Div => a_gpu.div(b_gpu)?,
        }
        .resolve()?;

        let d_gpu = c_gpu.to(&Device::CPU)?;
        ground.all_close(&d_gpu, 1e-4, 1e-4)?;
        Ok(())
    }

    #[proptest(cases = 8)]
    fn test_binary(prob: BinaryProblem) {
        run_binary_trial(prob).unwrap();
    }
}
