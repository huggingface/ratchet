use crate::gpu::{GpuBindGroup, GpuWorkload};
use crate::RVec;
use std::sync::Arc;
use wgpu::DynamicOffset;

#[derive(Debug)]
pub struct CompiledOp {
    workload: GpuWorkload,
    pipeline: Arc<wgpu::ComputePipeline>,
    storage_groups: RVec<GpuBindGroup>,
    offset: DynamicOffset,                  //offset into the uniform buffer
    uniform_group: Option<wgpu::BindGroup>, //create this after compilation is finished
}
