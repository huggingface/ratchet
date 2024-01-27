use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{
        BindGroupLayoutDescriptor, WorkgroupCount,
    },
    rvec, wgc, Enforcer, KernelElement, OpMetadata, Operation, OperationError, RVec,
    StorageView, Tensor,
};

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
    type Meta = BinaryMeta;

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.lhs, &self.rhs]
    }

    fn infer_output(&self, srcs: &[&Tensor]) -> Result<StorageView, OperationError> {
        Ok(srcs[0].view().clone())
    }

    fn check_invariants(srcs: &[&Tensor]) -> Result<(), OperationError> {
        Enforcer::check_input_arity(srcs, 2)?;
        Enforcer::check_dtype_match(srcs)?;
        Ok(())
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

    fn calculate_dispatch(&self) -> Result<WorkgroupCount, OperationError> {
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
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if inplace {
            Ok(BindGroupLayoutDescriptor::binary_inplace())
        } else {
            Ok(BindGroupLayoutDescriptor::binary())
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
        let a = &self.lhs;
        let a_rank = a.shape().rank();
        let [M, N] = [a.shape()[a_rank - 2] as u32, a.shape()[a_rank - 1] as u32];
        Ok(BinaryMeta { M, N })
    }
}
