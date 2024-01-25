use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{
        BindGroupLayoutDescriptor, ComputePipelineDescriptor, CpuUniform,
        PipelineLayoutDescriptor, WgpuDevice, WorkgroupCount,
    },
    rvec, wgc, CompiledOp, Enforcer, KernelElement, OpMetadata, Operation, OperationError, RVec,
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
    pub fn kernel_key(&self) -> &'static str {
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

    fn name(&self) -> &'static str {
        "Binary"
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.lhs, &self.rhs]
    }

    //TODO: we can refactor this into composite methods and share a single `compile` impl on the
    //trait
    fn compile(
        &self,
        dst: &Tensor,
        uniform: &mut CpuUniform,
        device: &WgpuDevice,
    ) -> Result<CompiledOp, OperationError> {
        let lhs = &self.lhs;
        let M = lhs.shape()[0] as u32;
        let N = lhs.shape()[1] as u32;
        let offset = uniform.write(&BinaryMeta { M, N })?;
        let wgcx = WorkgroupCount::div_ceil(M as _, 64);

        let storage_layout =
            device.get_or_create_bind_group_layout(&BindGroupLayoutDescriptor::binary())?;
        let uniform_layout =
            device.get_or_create_bind_group_layout(&BindGroupLayoutDescriptor::uniform())?;
        let pipeline_layout = device.get_or_create_pipeline_layout(&PipelineLayoutDescriptor {
            entries: rvec![storage_layout, uniform_layout],
        })?;

        let pipeline_handle =
            device.get_or_create_compute_pipeline(&ComputePipelineDescriptor {
                pipeline_layout,
                kernel_name: "add",
                kernel_element: KernelElement::Scalar,
            })?;

        let storage_bind_groups = CompiledOp::create_storage_bind_groups(
            &self.srcs(),
            dst,
            rvec![storage_layout],
            device,
        );

        Ok(CompiledOp::new(
            pipeline_handle,
            wgc![wgcx as _, 1, 1],
            storage_bind_groups,
            offset as _,
        ))
    }

    fn infer_output(&self, srcs: &[&Tensor]) -> Result<StorageView, OperationError> {
        Ok(srcs[0].view().clone())
    }

    fn check_invariants(srcs: &[&Tensor]) -> Result<(), OperationError> {
        Enforcer::check_input_arity(srcs, 2)?;
        Enforcer::check_dtype_match(srcs)?;
        Ok(())
    }
}
