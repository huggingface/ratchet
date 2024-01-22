use crate::{gpu::WgpuDevice, RVec};

use super::{
    static_resource_pool::{
        StaticResourcePool, StaticResourcePoolAccessor as _, StaticResourcePoolReadLockAccessor,
    },
    BindGroupLayoutHandle,
};

slotmap::new_key_type! { pub struct PipelineLayoutHandle; }

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PipelineLayoutDescriptor {
    pub entries: RVec<BindGroupLayoutHandle>,
}

#[derive(Default)]
pub(crate) struct PipelineLayoutPool {
    inner: StaticResourcePool<PipelineLayoutHandle, PipelineLayoutDescriptor, wgpu::PipelineLayout>,
}

impl PipelineLayoutPool {
    pub fn new() -> Self {
        Self {
            inner: StaticResourcePool::default(),
        }
    }

    pub fn get_or_create(
        &self,
        desc: &PipelineLayoutDescriptor,
        device: &WgpuDevice,
    ) -> PipelineLayoutHandle {
        self.inner.get_or_create(desc, |desc| {
            let bind_groups = device.bind_group_layout_resources();

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
    ) -> StaticResourcePoolReadLockAccessor<'_, PipelineLayoutHandle, wgpu::PipelineLayout> {
        self.inner.resources()
    }

    pub fn num_resources(&self) -> usize {
        self.inner.num_resources()
    }
}
