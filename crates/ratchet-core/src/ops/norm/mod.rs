//TODO: move to custom Op

mod groupnorm;

use encase::ShaderType;
pub use groupnorm::GroupNorm;

use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, DType, KernelElement, MetaOperation, OpGuards, OpMetadata, Operation,
    OperationError, RVec, StorageView, Tensor,
};
use derive_new::new;

#[derive(new, Debug, Clone)]
pub struct Norm {
    pub(crate) input: Tensor,
    pub(crate) scale: Tensor,
    pub(crate) bias: Option<Tensor>,
    pub(crate) eps: f32,
}
impl OpGuards for Norm {
    fn check_shapes(&self) {
        assert!(self.input.rank() >= 2);
    }

    fn check_dtypes(&self) {
        assert!(self.input.dt() == DType::F32);
        assert!(self.scale.dt() == DType::F32);
        if self.bias.is_some() {
            assert!(self.bias.as_ref().unwrap().dt() == DType::F32);
        }
    }
}

impl Operation for Norm {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        Ok(self.input.storage_view().clone())
    }
}

#[derive(Debug, Clone)]
pub enum NormOp {
    LayerNorm(Norm),
    RMSNorm(Norm),
    GroupNorm(GroupNorm),
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

impl MetaOperation for NormOp {
    fn kernel_name(&self) -> String {
        match self {
            NormOp::LayerNorm(_) => "layernorm".to_string(),
            NormOp::RMSNorm(_) => "rmsnorm".to_string(),
            NormOp::GroupNorm(_) => "groupnorm".to_string(),
        }
    }

    fn srcs(&self) -> RVec<&Tensor> {
        match self {
            NormOp::LayerNorm(Norm {
                input, scale, bias, ..
            }) => match bias {
                Some(bias) => rvec![input, scale, bias],
                None => rvec![input, scale],
            },
            NormOp::RMSNorm(Norm { input, scale, .. }) => rvec![input, scale],
            NormOp::GroupNorm(GroupNorm {
                norm: Norm {
                    input, scale, bias, ..
                },
                ..
            }) => match bias {
                Some(bias) => rvec![input, scale, bias],
                None => rvec![input, scale],
            },
        }
    }

    fn kernel_key(&self, _: bool, dst: &Tensor) -> String {
        let op_key = match self {
            NormOp::LayerNorm(_) => "layernorm",
            NormOp::RMSNorm(_) => "rmsnorm",
            NormOp::GroupNorm(_) => "groupnorm",
        };
        format!("{}_{}", op_key, self.kernel_element(dst).as_str())
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        let input = self.srcs()[0];
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
        match self {
            NormOp::LayerNorm(_) | NormOp::RMSNorm(_) => {
                let input = self.srcs()[0];
                let rank = input.rank();

                let M = input.shape()[rank - 2] as u32;
                let stacks = input.shape().slice(0..rank - 2).numel();
                Ok(wgc![M as _, stacks as _, 1])
            }
            NormOp::GroupNorm(GroupNorm { num_groups, .. }) => {
                let input = self.srcs()[0];
                let rank = input.rank();
                let M = *num_groups;
                let stacks = input.shape().slice(0..rank - 2).numel();
                Ok(wgc![M as _, stacks as _, 1])
            }
        }
    }

    fn storage_bind_group_layout(
        &self,
        _inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        match self {
            NormOp::LayerNorm(l) => match l.bias {
                Some(_) => Ok(BindGroupLayoutDescriptor::ternary()),
                None => Ok(BindGroupLayoutDescriptor::binary()),
            },
            NormOp::RMSNorm(_) => Ok(BindGroupLayoutDescriptor::binary()),
            NormOp::GroupNorm(l) => match l.norm.bias {
                Some(_) => Ok(BindGroupLayoutDescriptor::ternary()),
                None => Ok(BindGroupLayoutDescriptor::binary()),
            },
        }
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        _: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        let input = self.srcs()[0];
        let rank = input.rank();
        match self {
            NormOp::RMSNorm(n) | NormOp::LayerNorm(n) => {
                let M = input.shape()[rank - 2] as u32;
                let N = input.shape()[rank - 1] as u32;
                let ND2 = N / 2;
                let ND4 = N / 4;
                let meta = NormMeta::new(M, N, ND2, ND4, n.eps);
                Ok(uniform.write(&meta)?)
            }
            NormOp::GroupNorm(GroupNorm {
                norm: Norm { eps, .. },
                num_groups,
            }) => {
                let img_size = input.shape()[rank - 1] as u32;
                let channels = input.shape()[1] as u32;
                let M = *num_groups as u32;
                let N = (channels / *num_groups as u32) * img_size;
                let ND2 = N / 2;
                let ND4 = N / 4;
                let meta = NormMeta::new(M, N, ND2, ND4, *eps);
                Ok(uniform.write(&meta)?)
            }
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::test_util::run_py_prg;
    use crate::{rvec, shape, Device, DeviceRequest, Tensor};

    fn ground_truth(
        var: NormVariant,
        input: &Tensor,
        scale: &Tensor,
        bias: Option<&Tensor>,
    ) -> anyhow::Result<Tensor> {
        let ln_prg = r#"
import torch
import torch.nn.functional as F

def layer_norm(input, scale, bias):
    (input, scale, bias) = (torch.from_numpy(input), torch.from_numpy(scale), torch.from_numpy(bias))
    return F.layer_norm(input, (input.shape[-1],), weight=scale, bias=bias).numpy()
"#;

        let rms_prg = r#"
import torch
def manual_rms_norm(input, scale):
    (input, scale) = (torch.from_numpy(input), torch.from_numpy(scale))
    variance = input.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    input = input * torch.rsqrt(variance + 1e-5)
    return (scale * input).numpy()
"#;

        let prg = match var {
            NormVariant::LayerNorm => ln_prg,
            NormVariant::RMSNorm => rms_prg,
        };

        let inputs = match bias {
            Some(bias) => rvec![input, scale, bias],
            None => rvec![input, scale],
        };

        run_py_prg(prg.to_string(), &inputs, &[])
    }

    fn run_norm_trial(device: &Device, problem: NormProblem) -> anyhow::Result<()> {
        let NormProblem { var, B, M, N } = problem;
        let input = Tensor::randn::<f32>(shape![B, M, N], Device::CPU);
        let scale = Tensor::randn::<f32>(shape![N], Device::CPU);

        let bias = match var {
            NormVariant::LayerNorm => Some(Tensor::randn::<f32>(shape![N], Device::CPU)),
            NormVariant::RMSNorm => None,
        };

        let ground = match var {
            NormVariant::LayerNorm => ground_truth(var, &input, &scale, bias.as_ref())?,
            NormVariant::RMSNorm => ground_truth(var, &input, &scale, bias.as_ref())?,
        };

        let input_gpu = input.to(device)?;
        let scale_gpu = scale.to(device)?;
        let bias_gpu = bias.map(|b| b.to(device)).transpose()?;

        let result = match var {
            NormVariant::LayerNorm => input_gpu.layer_norm(scale_gpu, bias_gpu, 1e-5)?.resolve()?,
            NormVariant::RMSNorm => input_gpu.rms_norm(scale_gpu, 1e-5)?.resolve()?,
        };

        let ours = result.to(&Device::CPU)?;
        ground.all_close(&ours, 1e-4, 1e-4)?;
        Ok(())
    }

    #[derive(Arbitrary, Debug, Copy, Clone)]
    pub enum NormVariant {
        LayerNorm,
        RMSNorm,
    }

    #[derive(Arbitrary, Debug)]
    struct NormProblem {
        var: NormVariant,
        #[strategy(1..=3usize)]
        B: usize,
        #[strategy(1..=256usize)]
        M: usize,
        #[strategy(1..=256usize)]
        N: usize,
    }

    #[proptest(cases = 64)]
    fn test_norm(prob: NormProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        println!("prob = {:#?}", prob);
        run_norm_trial(&device, prob).unwrap();
    }
}
