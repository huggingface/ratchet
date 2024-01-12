// Adapted from https://github.com/rerun-io/rerun MIT licensed
use super::{DynamicResource, DynamicResourcePool, DynamicResourcesDesc, PoolError};
use crate::gpu::Device;

#[derive(Clone, Hash, PartialEq, Eq, Debug, derive_new::new)]
pub struct BufferDescriptor {
    pub size: wgpu::BufferAddress,
    pub usage: wgpu::BufferUsages,
    pub mapped_at_creation: bool,
}

slotmap::new_key_type! { pub struct GpuBufferHandle; }

/// A reference-counter baked buffer.
/// Once all instances are dropped, the buffer will be marked for reclamation in the following pass.
pub type GPUBuffer =
    std::sync::Arc<DynamicResource<GpuBufferHandle, BufferDescriptor, wgpu::Buffer>>;

impl DynamicResourcesDesc for BufferDescriptor {
    fn resource_size_in_bytes(&self) -> u64 {
        self.size
    }

    fn allow_reuse(&self) -> bool {
        if std::env::var("RATCHET_DEBUG").is_ok() {
            false
        } else {
            !self.mapped_at_creation
        }
    }
}

pub struct BufferPool {
    pool: DynamicResourcePool<GpuBufferHandle, BufferDescriptor, wgpu::Buffer>,
}

impl BufferPool {
    pub fn new() -> Self {
        Self {
            pool: DynamicResourcePool::default(),
        }
    }

    pub fn allocate(&mut self, desc: &BufferDescriptor, device: &Device) -> GPUBuffer {
        self.pool.allocate(desc, |desc| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: desc.size,
                usage: desc.usage,
                mapped_at_creation: desc.mapped_at_creation,
            })
        })
    }

    pub fn begin_pass(&mut self, pass_index: u64) {
        self.pool.begin_pass(pass_index, |res| res.destroy());
    }

    /// Method to retrieve a resource from a weak handle (used by [`super::GpuBindGroupPool`])
    pub fn get(&self, handle: GpuBufferHandle) -> Result<GPUBuffer, PoolError> {
        self.pool.get_from_handle(handle)
    }

    pub fn num_resources(&self) -> usize {
        self.pool.num_resources()
    }

    pub fn total_gpu_size_in_bytes(&self) -> u64 {
        self.pool.total_resource_size_in_bytes()
    }
}
