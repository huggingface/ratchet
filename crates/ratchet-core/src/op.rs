use std::fmt::Debug;

use encase::internal::WriteInto;
use encase::ShaderType;

use crate::gpu::{CpuUniform, PoolError, WgpuDevice, UNIFORM_ALIGN};
use crate::{rvec, Binary, CompiledOp, InvariantError, Matmul, RVec, Softmax, StorageView, Tensor};

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum LazyOp {
    Matmul(Matmul),
    Binary(Binary),
    Softmax(Softmax),
    Const,
}

impl LazyOp {
    pub fn srcs(&self) -> RVec<&Tensor> {
        match self {
            LazyOp::Binary(b) => b.srcs(),
            LazyOp::Matmul(m) => m.srcs(),
            LazyOp::Softmax(s) => s.srcs(),
            LazyOp::Const => rvec![], //end of the line kid
            _ => unimplemented!(),
        }
    }

    pub fn supports_inplace(&self) -> bool {
        match self {
            LazyOp::Binary(b) => b.supports_inplace(),
            LazyOp::Matmul(m) => m.supports_inplace(),
            LazyOp::Softmax(s) => s.supports_inplace(),
            LazyOp::Const => false,
            _ => false,
        }
    }

    pub fn is_const(&self) -> bool {
        matches!(self, LazyOp::Const)
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

    fn supports_inplace(&self) -> bool {
        false
    }

    fn compile(
        &self,
        dst: &Tensor,
        uniform: &mut CpuUniform,
        device: &WgpuDevice,
        can_inplace: bool,
    ) -> Result<CompiledOp, OperationError>;

    fn check_invariants(srcs: &[&Tensor]) -> Result<(), OperationError>;

    /// # Output Inference
    ///
    /// Inference is an overloaded term, in this context it means to determine
    /// the metadata of the output tensor given the input tensors.
    fn infer_output(&self, srcs: &[&Tensor]) -> Result<StorageView, OperationError>;
}
