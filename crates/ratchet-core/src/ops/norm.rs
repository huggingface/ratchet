//TODO: move to custom Op
use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, WorkgroupCount},
    rvec, wgc, Enforcer, KernelElement, MetaOperation, OpMetadata, Operation, OperationError, RVec,
    StorageView, Tensor,
};

#[derive(new, Debug, Clone)]
pub struct LayerNorm {
    scale: Tensor,
    bias: Option<Tensor>,
    eps: f32,
}

impl Operation for LayerNorm {
    fn check_invariants(srcs: &[&Tensor]) -> Result<(), OperationError> {
        Enforcer::check_input_arity_range(srcs, 2..=3)?;
        Ok(())
    }

    fn infer_output(&self, srcs: &[&Tensor]) -> Result<StorageView, OperationError> {
        Ok(srcs[0].view().clone())
    }
}

#[derive(Debug, Clone)]
pub enum NormOp {
    LayerNorm(LayerNorm),
}

#[derive(new, Debug, Clone)]
pub struct Norm {
    input: Tensor,
    op: NormOp,
}

impl Norm {
    pub fn name(&self) -> &'static str {
        "norm"
    }
}

#[derive(Debug, derive_new::new, ShaderType)]
pub struct NormMeta {
    M: u32,
    N: u32,
    ND2: u32,
    ND4: u32,
    eps: f32,
}

impl OpMetadata for NormMeta {}

impl Operation for Norm {
    fn infer_output(&self, srcs: &[&Tensor]) -> Result<StorageView, OperationError> {
        Ok(srcs[0].view().clone())
    }

    fn check_invariants(srcs: &[&Tensor]) -> Result<(), OperationError> {
        Enforcer::check_input_arity(srcs, 1)?;
        Ok(())
    }
}

impl MetaOperation for Norm {
    type Meta = NormMeta;

    fn srcs(&self) -> RVec<&Tensor> {
        match &self.op {
            NormOp::LayerNorm(LayerNorm { scale, bias, .. }) => match bias {
                Some(bias) => rvec![&self.input, scale, bias],
                None => rvec![&self.input, scale],
            },
        }
    }

    //TODO: enable inplace here

    fn kernel_name(&self) -> &'static str {
        match self.op {
            NormOp::LayerNorm(_) => "layernorm",
        }
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        let input = &self.input;
        let rank = input.rank();
        let N = input.shape()[rank - 1] as u32;
        if N % 4 == 0 {
            KernelElement::Vec4
        } else if N % 2 == 0 {
            KernelElement::Vec2
        } else {
            KernelElement::Scalar
        }
    }

    fn calculate_dispatch(&self, _dst: &Tensor) -> Result<WorkgroupCount, OperationError> {
        let input = &self.input;
        let rank = input.rank();

        let M = input.shape()[rank - 2] as u32;
        let stacks = input.shape().slice(0..rank - 2).numel();
        Ok(wgc![M as _, stacks as _, 1])
    }

    fn storage_bind_group_layout(
        &self,
        _inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        Ok(BindGroupLayoutDescriptor::ternary())
    }

    fn metadata(
        &self,
        _dst: &Tensor,
        _kernel_element: &KernelElement,
    ) -> Result<Self::Meta, OperationError> {
        let input = &self.input;
        let rank = input.rank();
        let M = input.shape()[rank - 2] as u32;
        let N = input.shape()[rank - 1] as u32;
        let ND2 = N / 2;
        let ND4 = N / 4;
        let eps = match &self.op {
            NormOp::LayerNorm(LayerNorm { eps, .. }) => *eps,
        };
        Ok(NormMeta::new(M, N, ND2, ND4, eps))
    }
}

#[cfg(test)]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::test_util::run_py_prg;
    use crate::{shape, Device, DeviceRequest, Tensor};

    fn ground_truth(input: &Tensor, scale: &Tensor, bias: &Tensor) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
import torch.nn.functional as F

def layernorm(input, scale, bias):
    (input, scale, bias) = (torch.from_numpy(input), torch.from_numpy(scale), torch.from_numpy(bias))
    return F.layer_norm(input, (input.shape[-1],), weight=scale, bias=bias).numpy()
"#;
        run_py_prg(prg.to_string(), &[input, scale, bias])
    }

    fn run_norm_trial(device: &Device, problem: NormProblem) -> anyhow::Result<()> {
        let NormProblem { B, M, N } = problem;
        let input = Tensor::randn::<f32>(shape![B, M, N], Device::CPU);
        let scale = Tensor::randn::<f32>(shape![N], Device::CPU);
        let bias = Tensor::randn::<f32>(shape![N], Device::CPU);
        let ground = ground_truth(&input, &scale, &bias)?;

        let input_gpu = input.to(device)?;
        let scale_gpu = scale.to(device)?;
        let bias_gpu = bias.to(device)?;

        let result = input_gpu.layer_norm(&scale_gpu, Some(&bias_gpu), 1e-5)?;
        result.resolve()?;

        let ours = result.to(&Device::CPU)?;
        ground.all_close(&ours, 1e-4, 1e-4)?;
        Ok(())
    }

    #[derive(Arbitrary, Debug)]
    struct NormProblem {
        #[strategy(1..=4usize)]
        B: usize,
        #[strategy(1..=512usize)]
        M: usize,
        #[strategy(1..=512usize)]
        N: usize,
    }

    #[proptest(cases = 32)]
    fn test_norm(prob: NormProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let NormProblem { B, M, N } = prob;
        println!("B = {}, M = {}, N = {}", B, M, N);
        run_norm_trial(&device, prob).unwrap();
    }
}
