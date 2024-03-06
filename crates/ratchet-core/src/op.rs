use std::fmt::Debug;

use encase::internal::WriteInto;
use encase::ShaderType;

use crate::gpu::{
    BindGroupLayoutDescriptor, ComputePipelineDescriptor, CpuUniform, PipelineLayoutDescriptor,
    PoolError, WgpuDevice, WorkgroupCount, UNIFORM_ALIGN,
};
use crate::{ops::*, rvec, CompiledOp, InvariantError, KernelElement, RVec, StorageView, Tensor};

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum LazyOp {
    Const,
    Matmul(Matmul),
    Binary(Binary),
    Unary(Unary),
    Reindex(Reindex),
    // ---- Everything below this line shouldn't exist ----
    Softmax(Softmax),
    Norm(Norm),
    View(View),             //Should be general class, metadata modification
    Conv(Conv),             //Really it's a matmul
    Select(IndexSelect),    //Can probably be Reindex
    IndexWrite(IndexWrite), //Above 2 should be merged
}

impl LazyOp {
    pub fn name(&self) -> &'static str {
        match self {
            LazyOp::Binary(b) => b.name(),
            LazyOp::Matmul(m) => m.name(),
            LazyOp::Softmax(s) => s.name(),
            LazyOp::Unary(u) => u.name(),
            LazyOp::Reindex(r) => r.name(),
            LazyOp::Norm(n) => n.name(),
            LazyOp::Conv(c) => c.name(),
            LazyOp::Select(s) => s.name(),
            LazyOp::IndexWrite(iw) => iw.name(),
            LazyOp::View(_) => "View",
            LazyOp::Const => "Const",
        }
    }

    pub fn srcs(&self) -> RVec<&Tensor> {
        match self {
            LazyOp::Binary(b) => b.srcs(),
            LazyOp::Matmul(m) => m.srcs(),
            LazyOp::Softmax(s) => s.srcs(),
            LazyOp::Unary(u) => u.srcs(),
            LazyOp::Reindex(r) => r.srcs(),
            LazyOp::Norm(n) => n.srcs(),
            LazyOp::Conv(c) => c.srcs(),
            LazyOp::Select(s) => s.srcs(),
            LazyOp::IndexWrite(iw) => iw.srcs(),
            LazyOp::View(v) => rvec![v.input()],
            LazyOp::Const => rvec![], //end of the line kid
        }
    }

    pub fn supports_inplace(&self) -> bool {
        match self {
            LazyOp::Binary(b) => b.supports_inplace(),
            LazyOp::Matmul(m) => m.supports_inplace(),
            LazyOp::Softmax(s) => s.supports_inplace(),
            LazyOp::Unary(u) => u.supports_inplace(),
            LazyOp::Reindex(r) => r.supports_inplace(),
            LazyOp::Norm(n) => n.supports_inplace(),
            LazyOp::Conv(c) => c.supports_inplace(),
            LazyOp::Select(s) => s.supports_inplace(),
            LazyOp::IndexWrite(iw) => iw.supports_inplace(),
            LazyOp::View(_v) => true,
            LazyOp::Const => false,
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
    #[error(transparent)]
    UnknownError(#[from] anyhow::Error),
}

///A trait for types that are written into uniform buffers, these
///hold the metadata for a shader.
pub trait OpMetadata: Debug + Sized + ShaderType + WriteInto {
    const __IS_VALID_META: () = {
        assert!(std::mem::size_of::<Self>() <= UNIFORM_ALIGN);
    };

    fn n_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

/// # MetaOperation
///
/// Meta Operation is a family of operations that can be compiled into relatively similar shaders.
/// Some types may implement both Operation and MetaOperation, if there is no variance
/// in output shape or invariants between the members of the family.
pub trait MetaOperation: Debug + 'static {
    ///Meta is a struct containing all data written into our uniform buffer.
    ///Typically contains shapes or strides.
    type Meta: OpMetadata;

    /// Return the file stem of the kernel source file.
    fn kernel_name(&self) -> &'static str;

    fn srcs(&self) -> RVec<&Tensor>;

    fn supports_inplace(&self) -> bool {
        false
    }

    /// # Kernel Element
    ///
    /// Determine the largest possible unit data type that can be used (e.g f32, vec2<f32>, vec4<f32>)
    fn kernel_element(&self, dst: &Tensor) -> KernelElement;

    /// # Calculate Dispatch
    ///
    /// Determine required amount of workgroups to execute the operation.
    fn calculate_dispatch(&self, dst: &Tensor) -> Result<WorkgroupCount, OperationError>;

    /// # Storage Bind Group Layout
    ///
    /// Determine the layout of the storage bind group.
    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError>;

    fn metadata(
        &self,
        dst: &Tensor,
        kernel_element: &KernelElement,
    ) -> Result<Self::Meta, OperationError>;

    fn compile(
        &self,
        dst: &Tensor,
        uniform: &mut CpuUniform,
        device: &WgpuDevice,
        can_inplace: bool,
    ) -> Result<CompiledOp, OperationError> {
        let kernel_element = self.kernel_element(dst);
        let meta = self.metadata(dst, &kernel_element)?;
        let offset = uniform.write(&meta)?;

        let workgroup_count = self.calculate_dispatch(dst)?;

        let storage_layout = device
            .get_or_create_bind_group_layout(&self.storage_bind_group_layout(can_inplace)?)?;
        let uniform_layout =
            device.get_or_create_bind_group_layout(&BindGroupLayoutDescriptor::uniform())?;
        let pipeline_layout = device.get_or_create_pipeline_layout(&PipelineLayoutDescriptor {
            entries: rvec![storage_layout, uniform_layout],
        })?;

        let pipeline_descriptor = ComputePipelineDescriptor {
            pipeline_layout,
            kernel_name: self.kernel_name(),
            kernel_element,
        };
        let pipeline_handle = device.get_or_create_compute_pipeline(&pipeline_descriptor)?;

        //TODO: Not sure i like this call here
        let storage_bind_groups = CompiledOp::create_storage_bind_groups(
            &self.srcs(),
            dst,
            rvec![storage_layout],
            device,
            can_inplace,
            self.kernel_name(),
        )?;

        Ok(CompiledOp::new(
            pipeline_handle,
            workgroup_count,
            storage_bind_groups,
            offset as _,
            self.kernel_name().to_string(),
        ))
    }
}

/// # Operation
///
/// An operation is a user facing type that represents a computation.
/// It checks the invariants that must be true before this operation can be executed.
///
/// An Operation is a member of a family of operations, called a MetaOperation, it may be the only
/// member.
///
/// Some types may implement both Operation and MetaOperation, if there is no variance
/// in output shape or invariants between the members of the family.
pub trait Operation: Debug + 'static {
    //These 2 methods below should be moved to another trait
    //They're unrelated
    fn check_invariants(srcs: &[&Tensor]) -> Result<(), OperationError>;

    /// # Output Inference
    ///
    /// Inference is an overloaded term, in this context it means to determine
    /// the metadata of the output tensor given the input tensors.
    fn infer_output(&self, srcs: &[&Tensor]) -> Result<StorageView, OperationError>;
}
