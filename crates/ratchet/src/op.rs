use std::fmt::Debug;

use encase::internal::WriteInto;
use encase::ShaderType;
use wgpu::DynamicOffset;

use crate::gpu::{
    BindGroupLayoutHandle, ComputePipelineHandle, CpuUniform, PoolError, WgpuDevice,
    WorkgroupCount, UNIFORM_ALIGN,
};
use crate::{Binary, CompiledOp, InvariantError, RVec, StorageView, Tensor};

#[derive(Debug, Clone)]
pub enum UnaryOp {
    Gelu,
}

#[derive(Clone)]
pub enum LazyOp {
    Empty,
    Binary(Binary),
    Unary(Tensor, UnaryOp),
    Const,
}

impl std::fmt::Debug for LazyOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LazyOp::Empty => write!(f, "Empty"),
            LazyOp::Binary(b) => write!(f, "{:?}", b),
            LazyOp::Unary(_, _) => write!(f, "Unary"),
            LazyOp::Const => write!(f, "Const"),
        }
    }
}

impl LazyOp {
    pub fn srcs(&self) -> RVec<&Tensor> {
        match self {
            LazyOp::Binary(b) => b.srcs(),
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OperationError {
    #[error("Failed to compile operation: {0}")]
    CompileError(String),
    #[error("Failed to get storage layout: {0}")]
    StorageLayoutError(#[from] PoolError),
    #[error(transparent)]
    InvariantError(#[from] InvariantError),
    #[error(transparent)]
    UniformError(#[from] encase::internal::Error),
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

    fn compile(
        &self,
        dst: &Tensor,
        uniform: &mut CpuUniform,
        device: &WgpuDevice,
    ) -> Result<CompiledOp, OperationError>;

    fn storage_layout(&self, device: &WgpuDevice) -> Result<BindGroupLayoutHandle, OperationError>;

    /// # Output Inference
    ///
    /// Inference is an overloaded term, in this context it means to determine
    /// the metadata of the output tensor given the input tensors.
    ///
    /// It also checks the invariants that need to hold for Binary.
    fn infer_output(&self, srcs: &[&Tensor]) -> Result<StorageView, OperationError>;
}
