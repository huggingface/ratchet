use rustc_hash::FxHashMap;
use wgpu::BufferUsages;

use crate::{
    gpu::{BufferDescriptor, BufferPool, GpuBufferHandle, PooledGPUBuffer},
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

    pub fn get(&self, handle: GpuBufferHandle) -> PooledGPUBuffer {
        self.pool.borrow().get(handle).unwrap()
    }

    pub fn pool(&self) -> Ref<BufferPool> {
        self.pool.borrow()
    }

    pub fn pool_mut(&self) -> RefMut<BufferPool> {
        self.pool.borrow_mut()
    }

    pub fn create_buffer(&self, desc: &BufferDescriptor, device: &WgpuDevice) -> PooledGPUBuffer {
        self.pool.borrow_mut().get_or_create(desc, device)
    }

    pub fn create_buffer_init(
        &self,
        desc: &BufferDescriptor,
        contents: &[u8],
        device: &WgpuDevice,
    ) -> PooledGPUBuffer {
        let buf = self.pool.borrow_mut().get_or_create(desc, device);
        device.queue().write_buffer(&buf.inner, 0, contents);
        buf
    }

    pub fn create_uniform_init(&self, uniform: CpuUniform, device: &WgpuDevice) -> PooledGPUBuffer {
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

    /// # Graph memory allocation
    ///
    /// Greedy algorithm, that takes the first buffer larger than the request
    /// In future, since we know the entire graph and sizes, we can
    /// do better.
    fn graph_allocate(
        &self,
        descriptor: BufferDescriptor,
        free: &mut Vec<PooledGPUBuffer>,
        device: &WgpuDevice,
    ) -> PooledGPUBuffer {
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

    /// # Inplace operations
    ///
    /// If an operation supports inplace, we need to "lease" the buffer
    /// from the actual source (i.e the first non-inplace operation)
    ///
    /// On what conditions do we terminate the upward traversal?
    /// 1. We reach an operation that does not support inplace
    /// 2. We reach an operation that has more than one consumer
    /// 3. We reach an operation that has more than one source
    fn determine_tensor_source<'a>(source: &'a Tensor, execution_order: &[Tensor]) -> &'a Tensor {
        let mut true_source = source;
        loop {
            let cant_inplace = !true_source.op().supports_inplace();
            let multiple_sources = true_source.op().srcs().len() > 1;
            let ts_index = execution_order
                .iter()
                .position(|t| t.id() == true_source.id())
                .unwrap();
            let multiple_consumers = execution_order[ts_index + 1..]
                .iter()
                .filter(|t| t.op().srcs().contains(&true_source))
                .count()
                > 1;
            if cant_inplace || multiple_sources || multiple_consumers {
                break;
            }

            true_source = true_source.op().srcs()[0];
        }
        true_source
    }

    /// # Graph memory allocation
    ///
    /// Simple greedy algorithm
    /// 1. Iterate over all tensors in reverse order
    /// 2. For each tensor, loop through it's sources
    /// 3. Each source is inserted into the assignments.
    /// 4. Release my output buffer (because we traverse in reverse order, when I arrive at myself,
    ///    my output buffer is no longer needed)
    pub fn allocate_cfg(
        &self,
        execution_order: &[Tensor],
        device: &WgpuDevice,
    ) -> Result<FxHashMap<TensorId, PooledGPUBuffer>, DeviceError> {
        let mut free = Vec::new(); //TODO: switch to BTreeMap
        let mut assignments = FxHashMap::default();

        for t in execution_order.iter().rev() {
            if t.resolved() {
                assignments.insert(
                    t.id(),
                    t.storage().as_ref().unwrap().try_gpu()?.inner.clone(),
                );
                continue;
            }

            // I need all of my sources to be allocated in order to compute my output value.
            // We "lease" the buffer, and it is released when we reach it in the execution order.
            // If the current tensor is an inplace operation,
            // we traverse upwards until we find a non-inplace operation.
            for source in t.op().srcs() {
                let true_source = Self::determine_tensor_source(source, execution_order);
                assignments.entry(true_source.id()).or_insert_with(|| {
                    self.graph_allocate(
                        BufferDescriptor::new(
                            true_source.num_bytes() as _,
                            BufferUsages::standard(),
                            false,
                        ),
                        &mut free,
                        device,
                    )
                });
            }

            //My buffer is no longer needed, since we traverse in reverse order
            //Earlier tensors can use my buffer
            if let Some(buf) = assignments.get(&t.id()) {
                free.push(buf.clone());
            }
        }

        //The output never gets allocated in the above loop, because it is not a source.
        //We know we need an allocation for the output.
        //We traverse upwards until we find the first non-inplace operation, and use it's buffer.
        let output = execution_order.last().unwrap();
        let output_source = Self::determine_tensor_source(output, execution_order);

        //If output source is allocated, we can use it's buffer
        //Otherwise, we need to allocate a new buffer
        let output_buffer = assignments
            .get(&output_source.id())
            .cloned()
            .unwrap_or_else(|| {
                self.graph_allocate(
                    BufferDescriptor::new(
                        output_source.num_bytes() as _,
                        BufferUsages::standard(),
                        false,
                    ),
                    &mut free,
                    device,
                )
            });
        assignments.insert(output.id(), output_buffer);

        Ok(assignments)
    }
}
