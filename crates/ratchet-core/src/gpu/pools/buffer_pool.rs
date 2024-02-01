// Adapted from https://github.com/rerun-io/rerun MIT licensed
use super::{DynamicResource, DynamicResourcePool, DynamicResourcesDesc, PoolError};
use crate::{
    gpu::{WgpuDevice, MIN_STORAGE_BUFFER_SIZE},
    RawGPUBuffer,
};

#[derive(Clone, Hash, PartialEq, Eq, Debug, derive_new::new)]
pub struct BufferDescriptor {
    pub size: wgpu::BufferAddress,
    pub usage: wgpu::BufferUsages,
    pub mapped_at_creation: bool,
}

impl BufferDescriptor {
    pub fn fields(&self) -> (wgpu::BufferAddress, wgpu::BufferUsages, bool) {
        (self.size, self.usage, self.mapped_at_creation)
    }
}

//All slotmap keys are COPY
slotmap::new_key_type! { pub struct GpuBufferHandle; }

/// A reference-counter baked buffer.
/// Once all instances are dropped, the buffer will be marked for reclamation in the following pass.
pub type PooledGPUBuffer =
    std::sync::Arc<DynamicResource<GpuBufferHandle, BufferDescriptor, RawGPUBuffer>>;

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
    inner: DynamicResourcePool<GpuBufferHandle, BufferDescriptor, RawGPUBuffer>,
}

impl BufferPool {
    pub fn new() -> Self {
        Self {
            inner: DynamicResourcePool::default(),
        }
    }

    pub fn get_or_create(&self, desc: &BufferDescriptor, device: &WgpuDevice) -> PooledGPUBuffer {
        let descriptor = if (desc.size as usize) < MIN_STORAGE_BUFFER_SIZE {
            BufferDescriptor {
                size: MIN_STORAGE_BUFFER_SIZE as _,
                usage: desc.usage,
                mapped_at_creation: desc.mapped_at_creation,
            }
        } else {
            desc.clone()
        };
        self.inner.get_or_create(&descriptor, |descriptor| {
            let (size, usage, mapped_at_creation) = descriptor.fields();
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage,
                mapped_at_creation,
            })
        })
    }

    pub fn begin_pass(&mut self, pass_index: u64) {
        self.inner.begin_pass(pass_index, |res| res.destroy());
    }

    /// Method to retrieve a resource from a weak handle (used by [`super::GpuBindGroupPool`])
    pub fn get(&self, handle: GpuBufferHandle) -> Result<PooledGPUBuffer, PoolError> {
        self.inner.get_from_handle(handle)
    }

    pub fn num_resources(&self) -> usize {
        self.inner.num_resources()
    }

    pub fn total_gpu_size_in_bytes(&self) -> u64 {
        self.inner.total_resource_size_in_bytes()
    }
}
