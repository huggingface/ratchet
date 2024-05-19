use crate::{gpu::WgpuDevice, rvec, RVec};

use super::{static_resource_pool::StaticResourcePool, StaticResourcePoolReadLockAccessor};

pub trait BindGroupLayoutEntryExt {
    fn compute_storage_buffer(binding: u32, read_only: bool) -> Self;
    fn dynamic_uniform_buffer() -> Self;
}

impl BindGroupLayoutEntryExt for wgpu::BindGroupLayoutEntry {
    fn compute_storage_buffer(binding: u32, read_only: bool) -> Self {
        Self {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                min_binding_size: None,
                has_dynamic_offset: false,
            },
            count: None,
        }
    }

    fn dynamic_uniform_buffer() -> Self {
        Self {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                min_binding_size: None,
                has_dynamic_offset: true,
            },
            count: None,
        }
    }
}

slotmap::new_key_type! { pub struct BindGroupLayoutHandle; }

#[derive(Debug, Clone, Hash, PartialEq, Eq, Default)]
pub struct BindGroupLayoutDescriptor {
    pub entries: RVec<wgpu::BindGroupLayoutEntry>,
}

impl BindGroupLayoutDescriptor {
    //Used for unary, binary, ternary (NOT INPLACE)
    fn entries(ro_length: usize) -> RVec<wgpu::BindGroupLayoutEntry> {
        let mut read_only: RVec<wgpu::BindGroupLayoutEntry> = (0..ro_length)
            .map(|idx| wgpu::BindGroupLayoutEntry::compute_storage_buffer(idx as u32, true))
            .collect();
        read_only.push(wgpu::BindGroupLayoutEntry::compute_storage_buffer(
            ro_length as u32,
            false,
        ));
        read_only
    }

    pub fn unary() -> Self {
        Self {
            entries: Self::entries(1),
        }
    }

    pub fn unary_inplace() -> Self {
        Self {
            entries: rvec![wgpu::BindGroupLayoutEntry::compute_storage_buffer(0, false)],
        }
    }

    pub fn binary() -> Self {
        Self {
            entries: Self::entries(2),
        }
    }

    pub fn binary_inplace() -> Self {
        Self {
            entries: rvec![
                wgpu::BindGroupLayoutEntry::compute_storage_buffer(0, false),
                wgpu::BindGroupLayoutEntry::compute_storage_buffer(1, true)
            ],
        }
    }

    pub fn ternary() -> Self {
        Self {
            entries: Self::entries(3),
        }
    }

    pub fn nthary(ro: usize) -> Self {
        Self {
            entries: Self::entries(ro),
        }
    }

    pub fn uniform() -> Self {
        Self {
            entries: rvec![wgpu::BindGroupLayoutEntry::dynamic_uniform_buffer()],
        }
    }
}

pub struct BindGroupLayoutPool {
    inner:
        StaticResourcePool<BindGroupLayoutHandle, BindGroupLayoutDescriptor, wgpu::BindGroupLayout>,
}

impl Default for BindGroupLayoutPool {
    fn default() -> Self {
        Self::new()
    }
}

impl BindGroupLayoutPool {
    pub fn new() -> Self {
        Self {
            inner: StaticResourcePool::default(),
        }
    }
}

impl BindGroupLayoutPool {
    pub fn get_or_create(
        &self,
        descriptor: &BindGroupLayoutDescriptor,
        device: &WgpuDevice,
    ) -> BindGroupLayoutHandle {
        self.inner.get_or_create(descriptor, |desc| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &desc.entries,
            })
        })
    }

    /// Locks the resource pool for resolving handles.
    ///
    /// While it is locked, no new resources can be added.
    pub fn resources(
        &self,
    ) -> StaticResourcePoolReadLockAccessor<'_, BindGroupLayoutHandle, wgpu::BindGroupLayout> {
        self.inner.resources()
    }
}
