use rustc_hash::FxHashMap;
use wgpu::BufferUsages;

use crate::{
    gpu::{BufferDescriptor, BufferPool, GPUBuffer, GpuBufferHandle},
    DeviceError, Tensor, TensorId,
};
use std::cell::{Ref, RefCell, RefMut};

use super::{BufferUsagesExt, CpuUniform, WgpuDevice, UNIFORM_ALIGN};

#[derive(Clone, Debug, thiserror::Error)]
pub enum AllocatorError {
    #[error("Buffer not found")]
    BufferNotFound,
}

pub struct BufferAllocator {
    //TODO: should this be RefCell
    pool: RefCell<BufferPool>,
}

impl BufferAllocator {
    pub fn new() -> Self {
        Self {
            pool: BufferPool::new().into(),
        }
    }

    pub fn begin_pass(&self, pass_index: u64) {
        self.pool.borrow_mut().begin_pass(pass_index);
    }

    pub fn get(&self, handle: GpuBufferHandle) -> GPUBuffer {
        self.pool.borrow().get(handle).unwrap()
    }

    pub fn pool(&self) -> Ref<BufferPool> {
        self.pool.borrow()
    }

    pub fn pool_mut(&self) -> RefMut<BufferPool> {
        self.pool.borrow_mut()
    }

    pub fn create_buffer(&self, desc: &BufferDescriptor, device: &WgpuDevice) -> GPUBuffer {
        self.pool.borrow_mut().get_or_create(desc, device)
    }

    pub fn create_buffer_init(
        &self,
        desc: &BufferDescriptor,
        contents: &[u8],
        device: &WgpuDevice,
    ) -> GPUBuffer {
        let buf = self.pool.borrow_mut().get_or_create(desc, device);
        device.queue().write_buffer(&buf.inner, 0, contents);
        buf
    }

    pub fn create_uniform_init(&self, uniform: CpuUniform, device: &WgpuDevice) -> GPUBuffer {
        let mut uniform = uniform.into_inner();
        uniform.resize(
            uniform.len() + UNIFORM_ALIGN - uniform.len() % UNIFORM_ALIGN,
            0u8,
        );
        let desc = BufferDescriptor::new(
            uniform.len() as _,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            false,
        );

        let resource = self.pool.borrow_mut().get_or_create(&desc, device);
        device
            .queue()
            .write_buffer(&resource.inner, 0, uniform.as_slice());
        resource
    }

    //Specific allocation method for the graph
    fn graph_allocate(
        &self,
        descriptor: BufferDescriptor,
        free: &mut Vec<GPUBuffer>,
        device: &WgpuDevice,
    ) -> GPUBuffer {
        let required_size = descriptor.size as _;
        let mut closest_index = None;
        let mut closest_size_diff: Option<usize> = None;
        for (idx, buffer) in free.iter().enumerate() {
            let current_size = buffer.descriptor.size as _;

            if current_size >= required_size {
                let size_diff = usize::abs_diff(current_size, required_size);

                if closest_size_diff.map_or(true, |diff| size_diff < diff) {
                    closest_index = Some(idx);
                    closest_size_diff = Some(size_diff);
                }
            }
        }

        if std::env::var("RATCHET_DEBUG").is_ok() {
            return self.create_buffer(&descriptor, device);
        }

        match closest_index {
            Some(idx) => free.remove(idx),
            None => self.create_buffer(&descriptor, device),
        }
    }

    /// # Graph memory allocation
    ///
    /// Simple greedy algorithm for allocating all required buffers to store
    /// activations during an inference pass.
    pub fn allocate_cfg(
        &self,
        execution_order: &[Tensor],
        device: &WgpuDevice,
    ) -> Result<FxHashMap<TensorId, GPUBuffer>, DeviceError> {
        let mut free = Vec::new(); //TODO: switch to BTreeMap
        let mut assignments = FxHashMap::default();

        for t in execution_order {
            if t.resolved() {
                let storage_resource = t
                    .storage()
                    .try_read()
                    .ok_or(AllocatorError::BufferNotFound)?;
                assignments.insert(t.id(), storage_resource.try_gpu()?.inner.clone());
                continue;
            }

            for source in t.op().srcs() {
                //Add support for inplace here when added
                let num_bytes = source.num_bytes();
                assignments.entry(source.id()).or_insert_with(|| {
                    self.graph_allocate(
                        BufferDescriptor::new(num_bytes as _, BufferUsages::standard(), false),
                        &mut free,
                        device,
                    )
                });
            }

            //release my buffer
            if let Some(buf) = assignments.get(&t.id()) {
                free.push(buf.clone());
            }
        }
        Ok(assignments)
    }
}
