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
mod tests {}
