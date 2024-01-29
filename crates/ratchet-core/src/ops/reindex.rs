use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, WorkgroupCount},
    rvec, wgc, Enforcer, InvariantError, Kernel, KernelElement, OpMetadata, Operation,
    OperationError, RVec, StorageView, Strides, Tensor,
};

#[cfg(test)]
use test_strategy::Arbitrary;

#[cfg_attr(test, derive(Arbitrary))]
#[derive(new, Debug, Clone)]
pub struct Permute {
    pub dims: Vec<usize>,
}

impl Operation for Permute {
    fn infer_output(&self, srcs: &[&Tensor]) -> Result<StorageView, OperationError> {
        let input_shape = srcs[0].shape();
        if input_shape.rank() != self.dims.len() {
            return Err(InvariantError::RankMismatch {
                accepted: input_shape.rank()..=input_shape.rank(),
                actual: self.dims.len(),
            })?;
        }

        let mut output_shape = input_shape.clone();
        for i in 0..input_shape.rank() {
            output_shape[i] = input_shape[self.dims[i]].clone();
        }
        let strides = Strides::from(&output_shape);
        Ok(StorageView::new(output_shape, srcs[0].dt(), strides))
    }

    fn check_invariants(srcs: &[&Tensor]) -> Result<(), OperationError> {
        Enforcer::check_input_arity(srcs, 1)?;
        Ok(())
    }
}

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
    numel: u32,
}

impl OpMetadata for ReindexMeta {}

impl Kernel for Reindex {
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
        let a = &self.input;
        let numel = a.shape().numel() as u32;
        Ok(ReindexMeta { numel })
    }
}

#[cfg(test)]
mod tests {}
