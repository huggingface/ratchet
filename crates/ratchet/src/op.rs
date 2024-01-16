use std::fmt::Debug;

use encase::internal::WriteInto;
use encase::ShaderType;
use wgpu::DynamicOffset;

use crate::gpu::{
    BindGroupLayoutHandle, ComputePipelineHandle, CpuUniform, WgpuDevice, WorkgroupCount,
    UNIFORM_ALIGN,
};
use crate::{Binary, RVec, Tensor};

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
    pub fn compile(
        &self,
        device: &WgpuDevice,
        uniform: &mut CpuUniform,
    ) -> Option<(ComputePipelineHandle, WorkgroupCount, DynamicOffset)> {
        match self {
            LazyOp::Binary(b) => Some(b.compile(device, uniform).unwrap()),
            LazyOp::Const => None,
            _ => unimplemented!(),
        }
    }

    pub fn srcs(&self) -> RVec<&Tensor> {
        match self {
            LazyOp::Binary(b) => b.srcs(),
            _ => unimplemented!(),
        }
    }

    pub fn can_compile(&self) -> bool {
        match self {
            LazyOp::Empty => false,
            LazyOp::Binary(_) => true,
            LazyOp::Unary(_, _) => true,
            LazyOp::Const => false,
        }
    }

    pub fn storage_layout(&self, device: &WgpuDevice) -> BindGroupLayoutHandle {
        match self {
            LazyOp::Binary(b) => b.storage_layout(device),
            _ => unimplemented!(),
        }
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

    fn compile(
        &self,
        device: &WgpuDevice,
        uniform: &mut CpuUniform,
    ) -> Result<(ComputePipelineHandle, WorkgroupCount, DynamicOffset), OperationError>;

    fn storage_layout(&self, device: &WgpuDevice) -> BindGroupLayoutHandle;
}
