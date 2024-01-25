use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{
        BindGroupLayoutDescriptor, ComputePipelineDescriptor, CpuUniform, PipelineLayoutDescriptor,
        WgpuDevice,
    },
    rvec, wgc, CompiledOp, Enforcer, KernelElement, OpMetadata, Operation, OperationError, RVec,
    StorageView, Tensor,
};

#[derive(new, Debug, Clone)]
pub struct Softmax {
    input: Tensor,
    dim: usize,
}

#[derive(Debug, derive_new::new, ShaderType)]
pub struct SoftmaxMeta {
    M: u32,
    N: u32,
    ND4: u32,
}

impl OpMetadata for SoftmaxMeta {}

impl Operation for Softmax {
    type Meta = SoftmaxMeta;

    fn name(&self) -> &'static str {
        "Softmax"
    }

    fn supports_inplace(&self) -> bool {
        true
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input]
    }

    fn compile(
        &self,
        dst: &Tensor,
        uniform: &mut CpuUniform,
        device: &WgpuDevice,
        can_inplace: bool,
    ) -> Result<CompiledOp, OperationError> {
        let input = &self.input;
        let M = input.shape()[self.dim - 1] as u32;
        let N = input.shape()[self.dim] as u32;
        let offset = uniform.write(&SoftmaxMeta { M, N, ND4: N / 4 })?;
        let stacks = input.shape().slice(0..self.dim - 1).numel();
        let workgroup_count = wgc![M as _, stacks as _, 1];

        let kernel_element = if N % 4 == 0 {
            KernelElement::Vec4
        } else if N % 2 == 0 {
            KernelElement::Vec2
        } else {
            KernelElement::Scalar
        };

        let storage_layout_desc = if can_inplace {
            BindGroupLayoutDescriptor::unary_inplace()
        } else {
            BindGroupLayoutDescriptor::unary()
        };
        let storage_layout = device.get_or_create_bind_group_layout(&storage_layout_desc)?;
        let uniform_layout =
            device.get_or_create_bind_group_layout(&BindGroupLayoutDescriptor::uniform())?;
        let pipeline_layout = device.get_or_create_pipeline_layout(&PipelineLayoutDescriptor {
            entries: rvec![storage_layout, uniform_layout],
        })?;

        let pipeline_handle =
            device.get_or_create_compute_pipeline(&ComputePipelineDescriptor {
                pipeline_layout,
                kernel_name: "softmax",
                kernel_element,
            })?;

        let storage_bind_groups = CompiledOp::create_storage_bind_groups(
            &self.srcs(),
            dst,
            rvec![storage_layout],
            device,
        );

        Ok(CompiledOp::new(
            pipeline_handle,
            workgroup_count,
            storage_bind_groups,
            offset as _,
        ))
    }

    fn infer_output(&self, srcs: &[&Tensor]) -> Result<StorageView, OperationError> {
        Ok(srcs[0].view().clone())
    }

    fn check_invariants(srcs: &[&Tensor]) -> Result<(), OperationError> {
        Enforcer::check_input_arity(srcs, 1)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    #[test]
    pub fn softmax() -> anyhow::Result<()> {
        Ok(())
    }
}
