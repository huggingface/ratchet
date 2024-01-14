use std::fmt::Debug;

use encase::internal::WriteInto;
use encase::ShaderType;

use crate::gpu::{BindGroupLayoutHandle, CpuUniform, WgpuDevice, UNIFORM_ALIGN};
use crate::{Binary, CompiledOp, Device, RVec, Tensor};

#[derive(Debug)]
pub enum UnaryOp {
    Gelu,
}

#[derive(Debug)]
pub enum LazyOp {
    Empty,
    Binary(Binary),
    Unary(Tensor, UnaryOp),
    Const,
}

impl LazyOp {
    pub fn compile(&self) -> CompiledOp {
        todo!()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OperationError {
    #[error("Failed to compile operation: {0}")]
    CompileError(String),
}

///A trait for types that are written into uniform buffers, these
///hold the metadata for a shader.
pub trait OpMetadata: Sized + ShaderType + WriteInto {
    const __IS_VALID_META: () = {
        assert!(std::mem::size_of::<Self>() <= UNIFORM_ALIGN);
    };

    fn n_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

//Every Operation in the CFG should implement this trait
pub trait Operation: Debug + 'static {
    ///Meta is a struct containing all data written into our uniform buffer.
    ///Typically contains shapes or strides.
    type Meta: OpMetadata;

    fn name(&self) -> &'static str;

    fn srcs(&self) -> RVec<&Tensor>;

    fn compile(&self, device: &Device, uniform: &CpuUniform) -> Result<CompiledOp, OperationError>;

    fn storage_layout(&self, device: &WgpuDevice) -> BindGroupLayoutHandle;
}
