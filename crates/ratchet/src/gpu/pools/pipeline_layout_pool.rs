use crate::{gpu::WgpuDevice, RVec};

use super::{
    static_resource_pool::{
        StaticResourcePool, StaticResourcePoolAccessor as _, StaticResourcePoolReadLockAccessor,
    },
    BindGroupLayoutHandle,
};

slotmap::new_key_type! { pub struct GpuPipelineLayoutHandle; }

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PipelineLayoutDescriptor {
    pub entries: RVec<BindGroupLayoutHandle>,
}

#[derive(Default)]
pub struct GpuPipelineLayoutPool {
    inner:
        StaticResourcePool<GpuPipelineLayoutHandle, PipelineLayoutDescriptor, wgpu::PipelineLayout>,
}

impl GpuPipelineLayoutPool {
    pub fn get_or_create(
        &self,
        desc: &PipelineLayoutDescriptor,
        device: &WgpuDevice,
    ) -> GpuPipelineLayoutHandle {
        self.inner.get_or_create(desc, |desc| {
            let bind_groups = device.get_bind_group_layout_resources();

            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &desc
                    .entries
                    .iter()
                    .map(|handle| bind_groups.get(*handle).unwrap())
                    .collect::<Vec<_>>(),
                push_constant_ranges: &[],
            })
        })
    }

    /// Locks the resource pool for resolving handles.
    ///
    /// While it is locked, no new resources can be added.
    pub fn resources(
        &self,
    ) -> StaticResourcePoolReadLockAccessor<'_, GpuPipelineLayoutHandle, wgpu::PipelineLayout> {
        self.inner.resources()
    }

    pub fn num_resources(&self) -> usize {
        self.inner.num_resources()
    }
}
