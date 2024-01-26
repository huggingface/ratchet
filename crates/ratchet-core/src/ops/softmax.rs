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
            can_inplace,
        );

        Ok(CompiledOp::new(
            pipeline_handle,
            workgroup_count,
            storage_bind_groups,
            offset as _,
        ))
    }

    fn infer_output(&self, srcs: &[&Tensor]) -> Result<StorageView, OperationError> {
        //TODO: FIX
        Ok(srcs[0].view().clone())
    }

    fn check_invariants(srcs: &[&Tensor]) -> Result<(), OperationError> {
        Enforcer::check_input_arity(srcs, 1)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::test_util::run_py_prg;
    use crate::{shape, Device, DeviceRequest, Tensor};

    fn ground_truth(a: &Tensor) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
import torch.nn.functional as F
def softmax(a):
    return F.softmax(torch.from_numpy(a), dim=-1).numpy()
"#;
        run_py_prg(prg.to_string(), &[a])
    }

    #[test]
    pub fn softmax() -> anyhow::Result<()> {
        let gpu_device = Device::request_device(DeviceRequest::GPU)?;
        let a = Tensor::randn::<f32>(shape![64, 64], Device::CPU);
        let ground = ground_truth(&a)?;

        let a_gpu = a.to(&gpu_device)?;
        let b = a_gpu.softmax(1)?;
        b.resolve()?;
        let ours = b.to(&Device::CPU)?;
        println!("GROUND: \n{:?}", ground);
        println!("OURS: \n{:?}", ours);
        ground.all_close(&ours, 1e-6, 1e-6)?;

        Ok(())
    }
}
