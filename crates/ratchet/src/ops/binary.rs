use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, BindGroupLayoutHandle, CpuUniform, WgpuDevice},
    rvec, CompiledOp, Device, OpMetadata, Operation, OperationError, RVec, Tensor,
};

#[derive(Debug)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, new)]
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

    fn name(&self) -> &'static str {
        "Binary"
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.lhs, &self.rhs]
    }

    fn storage_layout(&self, device: &WgpuDevice) -> BindGroupLayoutHandle {
        device
            .allocate_bind_group_layout(&BindGroupLayoutDescriptor::binary())
            .unwrap()
    }

    fn compile(&self, device: &Device, uniform: &CpuUniform) -> Result<CompiledOp, OperationError> {
        //Here we need to create the CompiledOp
        todo!()
    }
}
