use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use ratchet_macros::WgslMetadata;

use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform},
    rvec, Array, BindingMode, BuiltIn, DType, KernelElement, KernelSource, MetaOperation, OpGuards,
    Operation, OperationError, RVec, Scalar, StorageView, Strides, Tensor, Vec2, Vec4,
    WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
};

#[derive(new, Debug, Clone)]
pub struct Cast {
    input: Tensor,
    dst_dt: DType,
}

//Kernel cannot implement standard cast as it has multiple type bounds
impl Cast {
    fn register_bindings<SP: WgslPrimitive, DP: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        builder.register_storage("X", BindingMode::ReadOnly, Array::<SP>::default());
        builder.register_storage("Y", BindingMode::ReadWrite, Array::<DP>::default());
        builder.register_uniform();
        Ok(())
    }

    fn build_cast<SP: WgslPrimitive, DP: WgslPrimitive>(
        &self,
        inplace: bool,
        _: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let device = self.input.device().try_gpu().unwrap();
        let mut kernel_builder = WgslKernelBuilder::new(
            workgroup_size.clone(),
            rvec![
                BuiltIn::WorkgroupId,
                BuiltIn::LocalInvocationIndex,
                BuiltIn::NumWorkgroups
            ],
            device.compute_features().clone(),
        );

        self.register_bindings::<SP, DP>(&mut kernel_builder, inplace)?;
        kernel_builder.write_metadata::<CastMeta>();

        let n = SP::W;
        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let index = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (index >= metadata.numel / 'n) {
                return;
            }
        });

        let dst_accessor = DP::render_type();
        kernel_builder.write_main(wgsl! {
            Y[index] = 'dst_accessor(X[index]);
        });

        Ok(kernel_builder.build()?)
    }
}

#[derive(Debug, ShaderType, WgslMetadata)]
pub struct CastMeta {
    numel: u32,
}

impl OpGuards for Cast {
    fn check_shapes(&self) {}

    fn check_dtypes(&self) {}
}

impl Operation for Cast {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let shape = self.input.shape().clone();
        let strides = Strides::from(&shape);
        Ok(StorageView::new(shape, self.dst_dt, strides))
    }
}

impl MetaOperation for Cast {
    fn kernel_name(&self) -> String {
        "cast".to_string()
    }

    fn supports_inplace(&self) -> bool {
        //Really annoying that this can't be inplace
        //Buffer binding stops this from being possible
        false
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input]
    }

    fn kernel_element(&self, _: &Tensor) -> KernelElement {
        let numel = self.input.shape().numel();
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
        _: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        Ok(BindGroupLayoutDescriptor::unary())
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        _: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        let numel = self.input.shape().numel() as u32;
        Ok(uniform.write(&CastMeta { numel })?)
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = self.kernel_element(dst);
        match (self.input.dt(), self.dst_dt, &kernel_element) {
            (DType::F32, DType::F16, KernelElement::Scalar) => {
                self.build_cast::<Scalar<f32>, Scalar<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F32, DType::F16, KernelElement::Vec2) => {
                self.build_cast::<Vec2<f32>, Vec2<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F32, DType::F16, KernelElement::Vec4) => {
                self.build_cast::<Vec4<f32>, Vec4<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, DType::F32, KernelElement::Scalar) => {
                self.build_cast::<Scalar<f16>, Scalar<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F16, DType::F32, KernelElement::Vec2) => {
                self.build_cast::<Vec2<f16>, Vec2<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F16, DType::F32, KernelElement::Vec4) => {
                self.build_cast::<Vec4<f16>, Vec4<f32>>(inplace, dst, workgroup_size)
            }
            (DType::U32, DType::F32, KernelElement::Scalar) => {
                self.build_cast::<Scalar<u32>, Scalar<f32>>(inplace, dst, workgroup_size)
            }
            (DType::U32, DType::F32, KernelElement::Vec2) => {
                self.build_cast::<Vec2<u32>, Vec2<f32>>(inplace, dst, workgroup_size)
            }
            (DType::U32, DType::F32, KernelElement::Vec4) => {
                self.build_cast::<Vec4<u32>, Vec4<f32>>(inplace, dst, workgroup_size)
            }
            (DType::I32, DType::F32, KernelElement::Scalar) => {
                self.build_cast::<Scalar<i32>, Scalar<f32>>(inplace, dst, workgroup_size)
            }
            (DType::I32, DType::F32, KernelElement::Vec2) => {
                self.build_cast::<Vec2<i32>, Vec2<f32>>(inplace, dst, workgroup_size)
            }
            (DType::I32, DType::F32, KernelElement::Vec4) => {
                self.build_cast::<Vec4<i32>, Vec4<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, DType::U32, KernelElement::Scalar) => {
                self.build_cast::<Scalar<f32>, Scalar<u32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, DType::U32, KernelElement::Vec2) => {
                self.build_cast::<Vec2<f32>, Vec2<u32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, DType::U32, KernelElement::Vec4) => {
                self.build_cast::<Vec4<f32>, Vec4<u32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, DType::I32, KernelElement::Scalar) => {
                self.build_cast::<Scalar<f32>, Scalar<i32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, DType::I32, KernelElement::Vec2) => {
                self.build_cast::<Vec2<f32>, Vec2<i32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, DType::I32, KernelElement::Vec4) => {
                self.build_cast::<Vec4<f32>, Vec4<i32>>(inplace, dst, workgroup_size)
            }
            _ => unimplemented!(
                "Cannot cast from {:?} to {:?}",
                self.input.dt(),
                self.dst_dt
            ),
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use half::f16;
    use test_strategy::{proptest, Arbitrary};

    use crate::{shape, test_util::run_py_prg, DType, Device, DeviceRequest, Tensor};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct CastProblem {
        dst_dt: DType,
        #[strategy(1..=2usize)]
        B: usize,
        #[strategy(1..=128usize)]
        M: usize,
        #[strategy(1..=128usize)]
        N: usize,
    }

    fn ground_truth(input: &Tensor, dst_dt: DType) -> anyhow::Result<Tensor> {
        let prg = format!(
            r#"
import torch
def cast(a):
    return torch.from_numpy(a).to({}).numpy()
"#,
            dst_dt.as_torch()
        );

        run_py_prg(prg.to_string(), &[input], &[], dst_dt)
    }

    fn run_cast_trial(prob: CastProblem) -> anyhow::Result<()> {
        let device = GPU_DEVICE.with(|d| d.clone());
        let CastProblem { dst_dt, B, M, N } = prob;
        let input = Tensor::randn::<f32>(shape![B, M, N], Device::CPU);
        let ground = ground_truth(&input, dst_dt)?;

        let input_gpu = input.to(&device)?;
        let casted = input_gpu.cast(dst_dt)?.resolve()?;

        let casted_cpu = casted.to(&Device::CPU)?;
        match dst_dt {
            DType::F16 => {
                ground.all_close::<f16>(&casted_cpu, f16::from_f32(1e-4), f16::from_f32(1e-4))?
            }
            DType::F32 => ground.all_close::<f32>(&casted_cpu, 1e-4, 1e-4)?,
            _ => return Ok(()), //all_close doesn't support integers
        }
        Ok(())
    }

    #[proptest(cases = 256)]
    fn test_type_cast(prob: CastProblem) {
        run_cast_trial(prob).unwrap();
    }
}
