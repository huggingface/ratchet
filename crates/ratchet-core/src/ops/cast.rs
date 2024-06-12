use std::borrow::Cow;

use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use ratchet_macros::WgslMetadata;

use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, wgs, Array, BindingMode, BuiltIn, DType, KernelElement, KernelSource, MetaOperation,
    OpGuards, Operation, OperationError, RVec, Scalar, StorageView, Tensor, Vec2, Vec4,
    WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
};

#[derive(new, Debug, Clone)]
pub struct Cast {
    input: Tensor,
    dst_dt: DType,
}

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

        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let index = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (index >= metadata.numel / 'n) {
                return;
            }
        });

        kernel_builder.write_main(wgsl! {
            Y[index] = 'func(X[index]);
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
        Ok(self.input.storage_view().clone())
    }
}

impl MetaOperation for Cast {
    fn kernel_name(&self) -> String {
        format!("cast_{}_to_{}", self.input.dt(), self.dst_dt)
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

    fn calculate_dispatch(&self, _dst: &Tensor) -> Result<Workload, OperationError> {
        let workgroup_size = wgs![8, 8, 1];

        let numel = self.input.shape().numel();
        let x_groups = WorkgroupCount::div_ceil(numel as _, workgroup_size.product() as _);
        let (x_groups, y_groups) = if x_groups > WorkgroupCount::MAX_WGS_PER_DIM {
            let y_groups = WorkgroupCount::div_ceil(x_groups, WorkgroupCount::MAX_WGS_PER_DIM);
            (WorkgroupCount::MAX_WGS_PER_DIM, y_groups)
        } else {
            (x_groups, 1)
        };

        Ok(Workload {
            workgroup_count: wgc![x_groups as _, y_groups as _, 1],
            workgroup_size,
        })
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
        let a = &self.input;
        let numel = a.shape().numel() as u32;
        let meta = CastMeta { numel };
        Ok(uniform.write(&meta)?)
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
            _ => Err(OperationError::CompileError(format!(
                "Unsupported dtype {:?} or kernel element {:?}",
                self.input.dt(),
                kernel_element
            ))),
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {}
