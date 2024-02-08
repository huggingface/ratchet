use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, WorkgroupCount},
    rvec, wgc, Enforcer, KernelElement, MetaOperation, OpMetadata, Operation, OperationError, RVec,
    StorageView, Tensor,
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
}

#[derive(new, Debug, Clone)]
pub struct Binary {
    lhs: Tensor,
    rhs: Tensor,
    op: BinaryOp,
}

impl Binary {
    pub fn name(&self) -> &'static str {
        self.op.kernel_name()
    }

    pub fn op(&self) -> &BinaryOp {
        &self.op
    }
}

#[derive(Debug, ShaderType)]
pub struct BinaryMeta {
    M: u32,
    N: u32,
}

impl OpMetadata for BinaryMeta {}

impl Operation for Binary {
    fn infer_output(&self, srcs: &[&Tensor]) -> Result<StorageView, OperationError> {
        //TODO: THIS IS WRONG
        if srcs.len() != 2 {
            panic!("Binary operation expects 2 inputs");
        }
        if srcs[0].shape() != srcs[1].shape() {
            panic!("Binary operation expects inputs of the same shape");
        }
        Ok(srcs[0].storage_view().clone())
    }

    fn check_invariants(srcs: &[&Tensor]) -> Result<(), OperationError> {
        Enforcer::check_input_arity(srcs, 2)?;
        Enforcer::check_dtype_match(srcs)?;
        Ok(())
    }
}

impl MetaOperation for Binary {
    type Meta = BinaryMeta;

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.lhs, &self.rhs]
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        let a_rank = &self.lhs.shape().rank();
        let N = &self.lhs.shape()[a_rank - 1];

        if N % 4 == 0 {
            KernelElement::Vec4
        } else if N % 2 == 0 {
            KernelElement::Vec2
        } else {
            KernelElement::Scalar
        }
    }

    fn calculate_dispatch(&self, _dst: &Tensor) -> Result<WorkgroupCount, OperationError> {
        let a = &self.lhs;
        let a_rank = a.shape().rank();
        let M = a.shape()[a_rank - 2];
        let a_prefix = &a.shape()[..(a_rank - 2)];
        let stacks = a_prefix.iter().product::<usize>();

        let wgcx = WorkgroupCount::div_ceil(M as _, 64);
        Ok(wgc![wgcx as _, stacks as _, 1])
    }

    fn storage_bind_group_layout(
        &self,
        _inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        /*
        if inplace {
            Ok(BindGroupLayoutDescriptor::binary_inplace())
        } else {
            Ok(BindGroupLayoutDescriptor::binary())
        }
        */
        Ok(BindGroupLayoutDescriptor::binary())
    }

    fn kernel_name(&self) -> &'static str {
        self.op.kernel_name()
    }

    fn metadata(
        &self,
        _dst: &Tensor,
        _kernel_element: &KernelElement,
    ) -> Result<Self::Meta, OperationError> {
        let a = &self.lhs;
        let a_rank = a.shape().rank();
        let [M, N] = [a.shape()[a_rank - 2] as u32, a.shape()[a_rank - 1] as u32];
        Ok(BinaryMeta { M, N })
    }
}

#[cfg(test)]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::{shape, test_util::run_py_prg, BinaryOp, Device, DeviceRequest, Tensor};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct BinaryProblem {
        op: BinaryOp,
        #[strategy(1..=4usize)]
        B: usize,
        #[strategy(1..=512usize)]
        M: usize,
        //#[strategy(1..=512usize)]
        //N: usize,
        //TODO: add N support
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
        run_py_prg(prg.to_string(), &[a, b])
    }

    fn run_binary_trial(prob: BinaryProblem) -> anyhow::Result<()> {
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let BinaryProblem { op, B, M } = prob;
        println!("op: {:?}, B: {}, M: {}", op, B, M);
        let a = Tensor::randn::<f32>(shape![B, M], cpu_device.clone());
        let b = Tensor::randn::<f32>(shape![1], cpu_device.clone());
        let ground = ground_truth(&a, &b, &op)?;
        let device = GPU_DEVICE.with(|d| d.clone());

        let a_gpu = a.to(&device)?;
        let b_gpu = b.to(&device)?;
        let c_gpu = match op {
            BinaryOp::Add => a_gpu.add(&b_gpu)?,
            BinaryOp::Sub => a_gpu.sub(&b_gpu)?,
            BinaryOp::Mul => a_gpu.mul(&b_gpu)?,
            BinaryOp::Div => a_gpu.div(&b_gpu)?,
        };
        c_gpu.resolve()?;

        let d_gpu = c_gpu.to(&Device::CPU)?;
        ground.all_close(&d_gpu, 1e-4, 1e-4)?;
        Ok(())
    }

    #[proptest(cases = 8)]
    fn test_binary(prob: BinaryProblem) {
        run_binary_trial(prob).unwrap();
    }
}
