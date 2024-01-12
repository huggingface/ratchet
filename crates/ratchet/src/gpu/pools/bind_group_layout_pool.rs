use crate::{rvec, Device, RVec};

use super::static_resource_pool::{PoolError, StaticResourcePool};

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
    pub fn unary() -> Self {
        Self {
            entries: rvec![
                wgpu::BindGroupLayoutEntry::compute_storage_buffer(0, true),
                wgpu::BindGroupLayoutEntry::compute_storage_buffer(1, false)
            ],
        }
    }

    pub fn unary_inplace() -> Self {
        Self {
            entries: rvec![wgpu::BindGroupLayoutEntry::compute_storage_buffer(0, false)],
        }
    }

    pub fn binary() -> Self {
        Self {
            entries: rvec![
                wgpu::BindGroupLayoutEntry::compute_storage_buffer(0, true),
                wgpu::BindGroupLayoutEntry::compute_storage_buffer(1, true),
                wgpu::BindGroupLayoutEntry::compute_storage_buffer(2, false)
            ],
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
            entries: rvec![
                wgpu::BindGroupLayoutEntry::compute_storage_buffer(0, true),
                wgpu::BindGroupLayoutEntry::compute_storage_buffer(1, true),
                wgpu::BindGroupLayoutEntry::compute_storage_buffer(2, true),
                wgpu::BindGroupLayoutEntry::compute_storage_buffer(3, false)
            ],
        }
    }

    pub fn quaternary() -> RVec<Self> {
        rvec![
            Self {
                entries: (0..4)
                    .map(|i| wgpu::BindGroupLayoutEntry::compute_storage_buffer(i, true))
                    .collect()
            },
            Self {
                entries: rvec![wgpu::BindGroupLayoutEntry::compute_storage_buffer(0, false)],
            }
        ]
    }

    pub fn uniform() -> Self {
        Self {
            entries: rvec![wgpu::BindGroupLayoutEntry::dynamic_uniform_buffer()],
        }
    }
}

pub struct BindGroupLayoutPool {
    pool:
        StaticResourcePool<BindGroupLayoutHandle, BindGroupLayoutDescriptor, wgpu::BindGroupLayout>,
}

impl BindGroupLayoutPool {
    pub fn new() -> Self {
        Self {
            pool: StaticResourcePool::default(),
        }
    }
}

impl BindGroupLayoutPool {
    pub fn get_or_create(
        &mut self,
        descriptor: &BindGroupLayoutDescriptor,
        device: &Device,
    ) -> BindGroupLayoutHandle {
        self.pool.get_or_create(descriptor, |desc| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &desc.entries,
            })
        })
    }

    pub fn get_resource(
        &self,
        handle: BindGroupLayoutHandle,
    ) -> Result<&wgpu::BindGroupLayout, PoolError> {
        self.pool.get_resource(handle)
    }

    pub fn num_resources(&self) -> usize {
        self.pool.num_resources()
    }
}
