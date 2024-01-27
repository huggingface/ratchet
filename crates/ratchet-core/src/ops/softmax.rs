use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{
        BindGroupLayoutDescriptor, WorkgroupCount,
    },
    rvec, wgc, Enforcer, KernelElement, OpMetadata, Operation, OperationError, RVec,
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

    fn supports_inplace(&self) -> bool {
        true
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input]
    }

    fn infer_output(&self, srcs: &[&Tensor]) -> Result<StorageView, OperationError> {
        Ok(srcs[0].view().clone())
    }

    fn check_invariants(srcs: &[&Tensor]) -> Result<(), OperationError> {
        Enforcer::check_input_arity(srcs, 1)?;
        Ok(())
    }

    fn kernel_name(&self) -> &'static str {
        "softmax"
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        let input = &self.input;
        let N = input.shape()[self.dim] as u32;
        if N % 4 == 0 {
            KernelElement::Vec4
        } else if N % 2 == 0 {
            KernelElement::Vec2
        } else {
            KernelElement::Scalar
        }
    }

    fn calculate_dispatch(&self) -> Result<WorkgroupCount, OperationError> {
        let input = &self.input;
        let stacks = input.shape().slice(0..self.dim - 1).numel();
        let M = input.shape()[self.dim - 1] as u32;
        Ok(wgc![M as _, stacks as _, 1])
    }

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if inplace {
            Ok(BindGroupLayoutDescriptor::unary_inplace())
        } else {
            Ok(BindGroupLayoutDescriptor::unary())
        }
    }

    fn metadata(
        &self,
        _dst: &Tensor,
        _kernel_element: &KernelElement,
    ) -> Result<Self::Meta, OperationError> {
        let input = &self.input;
        let M = input.shape()[self.dim - 1] as u32;
        let N = input.shape()[self.dim] as u32;
        let ND4 = N / 4;
        Ok(SoftmaxMeta { M, N, ND4 })
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
