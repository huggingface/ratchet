mod permute;
pub use permute::Permute;

use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, WorkgroupCount},
    rvec, wgc, Enforcer, InvariantError, KernelElement, MetaOperation, OpMetadata, Operation,
    OperationError, RVec, StorageView, Strides, Tensor,
};

#[cfg(test)]
use test_strategy::Arbitrary;

#[cfg_attr(test, derive(Arbitrary))]
#[derive(Debug, Clone)]
pub enum ReindexOp {
    Permute(Permute),
}

impl ReindexOp {
    pub fn kernel_name(&self) -> &'static str {
        match self {
            ReindexOp::Permute(_) => "permute",
        }
    }
}

#[derive(new, Debug, Clone)]
pub struct Reindex {
    input: Tensor,
    op: ReindexOp,
}

impl Reindex {
    pub fn op(&self) -> &ReindexOp {
        &self.op
    }
}

#[derive(Debug, ShaderType)]
pub struct ReindexMeta {
    src_stride: [u32; 4],
    dst_stride: [u32; 4],
    src_numel: u32,
    dst_numel: u32,
    // "Optional" fields below
    permute: [u32; 4],
}

impl OpMetadata for ReindexMeta {}

impl MetaOperation for Reindex {
    type Meta = ReindexMeta;

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input]
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        KernelElement::Scalar
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
        Ok(BindGroupLayoutDescriptor::unary())
    }

    fn kernel_name(&self) -> &'static str {
        self.op.kernel_name()
    }

    fn metadata(
        &self,
        _dst: &Tensor,
        _kernel_element: &KernelElement,
    ) -> Result<Self::Meta, OperationError> {
        //TODO: this is garbage
        let src_stride = self.input.strides().try_into().unwrap();
        let dst_stride = self.input.strides().try_into().unwrap();
        let src_numel = self.input.shape().numel() as u32;
        let dst_numel = self.input.shape().numel() as u32;
        let permute = match &self.op {
            ReindexOp::Permute(p) => p.dims.iter().map(|&d| d as u32).collect::<Vec<_>>(),
        };
        let pp = permute.as_slice().try_into().unwrap();
        Ok(ReindexMeta {
            src_stride,
            dst_stride,
            src_numel,
            dst_numel,
            permute: pp,
        })
    }
}

#[cfg(test)]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::{shape, test_util::run_py_prg, BinaryOp, Device, DeviceRequest, ReindexOp, Tensor};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct ReindexProblem {
        op: ReindexOp,
        #[strategy(1..=4usize)]
        B: usize,
        #[strategy(1..=512usize)]
        M: usize,
        #[strategy(1..=512usize)]
        N: usize,
    }

    fn ground_truth(a: &Tensor, op: &ReindexOp, args: &str) -> anyhow::Result<Tensor> {
        let kn = op.kernel_name();
        let prg = format!(
            r#"
import torch
def {}(a):
    return torch.{}(torch.from_numpy(a), {}).numpy()
"#,
            kn, kn, args
        );
        run_py_prg(prg.to_string(), &[a])
    }

    fn run_reindex_trial(prob: ReindexProblem) -> anyhow::Result<()> {
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let ReindexProblem { op, B, M, N } = prob;
        println!("op: {:?}, B: {}, M: {}, N: {}", op, B, M, N);
        let a = Tensor::randn::<f32>(shape![B, M, N], cpu_device.clone());
        let device = GPU_DEVICE.with(|d| d.clone());

        let a_gpu = a.to(&device)?;
        let (ours, ground) = match op {
            ReindexOp::Permute(ref p) => {
                let arg_str = format!("{:?}", p.dims);
                println!("arg_str: {}", arg_str);
                let ground = ground_truth(&a, &op, arg_str.as_str())?;
                (a.permute(&p.dims)?, ground)
            }
        };
        ours.resolve()?;
        let d_gpu = ours.to(&Device::CPU)?;
        ground.all_close(&d_gpu, 1e-4, 1e-4)?;
        Ok(())
    }

    #[proptest(cases = 1)]
    fn test_reindex(prob: ReindexProblem) {
        run_reindex_trial(prob).unwrap();
    }
}
