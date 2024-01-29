use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, WorkgroupCount},
    rvec, wgc, Enforcer, KernelElement, OpMetadata, Operation, OperationError, RVec, StorageView,
    Tensor,
};

#[cfg(test)]
use test_strategy::Arbitrary;

#[cfg_attr(test, derive(Arbitrary))]
#[derive(Debug, Clone)]
pub enum UnaryOp {
    Gelu,
    Tanh,
    Exp,
    Log,
    Sin,
    Cos,
    Abs,
    Sqrt,
    Relu,
    Floor,
    Ceil,
}

impl UnaryOp {
    pub fn kernel_name(&self) -> &'static str {
        match self {
            UnaryOp::Gelu => "gelu",
            UnaryOp::Tanh => "tanh",
            UnaryOp::Exp => "exp",
            UnaryOp::Log => "log",
            UnaryOp::Sin => "sin",
            UnaryOp::Cos => "cos",
            UnaryOp::Abs => "abs",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Relu => "relu",
            UnaryOp::Floor => "floor",
            UnaryOp::Ceil => "ceil",
        }
    }
}

#[derive(new, Debug, Clone)]
pub struct Unary {
    input: Tensor,
    op: UnaryOp,
}

impl Unary {
    pub fn op(&self) -> &UnaryOp {
        &self.op
    }
}

#[derive(Debug, ShaderType)]
pub struct UnaryMeta {
    numel: u32,
}

impl OpMetadata for UnaryMeta {}

impl Operation for Unary {
    type Meta = UnaryMeta;

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input]
    }

    fn supports_inplace(&self) -> bool {
        true
    }

    fn infer_output(&self, srcs: &[&Tensor]) -> Result<StorageView, OperationError> {
        Ok(srcs[0].view().clone())
    }

    fn check_invariants(srcs: &[&Tensor]) -> Result<(), OperationError> {
        Enforcer::check_input_arity(srcs, 1)?;
        Ok(())
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        let a_rank = &self.input.shape().rank();
        let N = &self.input.shape()[a_rank - 1];

        if N % 4 == 0 {
            KernelElement::Vec4
        } else if N % 2 == 0 {
            KernelElement::Vec2
        } else {
            KernelElement::Scalar
        }
    }

    fn calculate_dispatch(&self) -> Result<WorkgroupCount, OperationError> {
        let numel = self.input.shape().numel();
        let wgcx = WorkgroupCount::div_ceil(numel as _, 64);
        Ok(wgc![wgcx as _, 1, 1])
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

    fn kernel_name(&self) -> &'static str {
        self.op.kernel_name()
    }

    fn metadata(
        &self,
        _dst: &Tensor,
        _kernel_element: &KernelElement,
    ) -> Result<Self::Meta, OperationError> {
        let a = &self.input;
        let numel = a.shape().numel() as u32;
        Ok(UnaryMeta { numel })
    }
}

#[cfg(test)]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::{shape, test_util::run_py_prg, Device, DeviceRequest, Tensor, UnaryOp};

    #[derive(Arbitrary, Debug)]
    struct UnaryProblem {
        op: UnaryOp,
        #[strategy(1..=4usize)]
        B: usize,
        #[strategy(1..=512usize)]
        M: usize,
        #[strategy(1..=512usize)]
        N: usize,
    }

    fn ground_truth(a: &Tensor, op: &UnaryOp, args: &str) -> anyhow::Result<Tensor> {
        let kn = op.kernel_name();
        let func_prg = format!(
            r#"
import torch
import torch.nn.functional as F

def {}(a):
    return F.{}(torch.from_numpy(a), {}).numpy()
"#,
            kn, kn, args,
        );

        let imp_prg = format!(
            r#"
import torch

def {}(a):
    return torch.{}(torch.from_numpy(a), {}).numpy()
"#,
            kn, kn, args,
        );

        let prg = match op {
            UnaryOp::Gelu => func_prg,
            _ => imp_prg,
        };

        run_py_prg(prg.to_string(), &[a])
    }

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    fn run_unary_trial(prob: UnaryProblem) -> anyhow::Result<()> {
        let device = GPU_DEVICE.with(|d| d.clone());
        let UnaryProblem { op, B, M, N } = prob;
        println!("op: {:?}, B: {}, M: {}, N: {}", op, B, M, N);
        let a = Tensor::randn::<f32>(shape![B, M], Device::CPU);

        let args = match op {
            UnaryOp::Gelu => "approximate=\"tanh\"",
            _ => "",
        };
        let ground = ground_truth(&a, &op, args)?;

        let a_gpu = a.to(&device)?;
        let c_gpu = match op {
            UnaryOp::Gelu => a_gpu.gelu()?,
            UnaryOp::Tanh => a_gpu.tanh()?,
            UnaryOp::Exp => a_gpu.exp()?,
            UnaryOp::Log => a_gpu.log()?,
            UnaryOp::Sin => a_gpu.sin()?,
            UnaryOp::Cos => a_gpu.cos()?,
            UnaryOp::Abs => a_gpu.abs()?,
            UnaryOp::Sqrt => a_gpu.sqrt()?,
            UnaryOp::Relu => a_gpu.relu()?,
            UnaryOp::Floor => a_gpu.floor()?,
            UnaryOp::Ceil => a_gpu.ceil()?,
        };
        c_gpu.resolve()?;

        let d_gpu = c_gpu.to(&Device::CPU)?;
        ground.all_close(&d_gpu, 5e-3, 5e-3)?;
        Ok(())
    }

    #[proptest(cases = 128)]
    fn test_unary(prob: UnaryProblem) {
        run_unary_trial(prob).unwrap();
    }
}
