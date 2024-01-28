use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, WorkgroupCount},
    rvec, wgc, Enforcer, KernelElement, OpMetadata, Operation, OperationError, RVec, StorageView,
    Tensor,
};

#[derive(Debug, Clone)]
pub enum UnaryOp {
    Gelu,
    Tanh,
}

impl UnaryOp {
    pub fn kernel_name(&self) -> &'static str {
        match self {
            UnaryOp::Gelu => "gelu",
            UnaryOp::Tanh => "tanh",
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
    M: u32,
    N: u32,
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
        let a_rank = a.shape().rank();
        let [M, N] = [a.shape()[a_rank - 2] as u32, a.shape()[a_rank - 1] as u32];
        Ok(UnaryMeta { M, N })
    }
}

#[cfg(test)]
mod tests {}
