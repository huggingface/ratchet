use rustc_hash::FxHashMap;
use wgpu::BufferUsages;

use crate::{
    gpu::{BufferDescriptor, BufferPool, GpuBufferHandle, PooledGPUBuffer},
    DeviceError, Tensor, TensorId,
};
use std::{
    cell::{Ref, RefCell, RefMut},
    sync::Arc,
};

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
        free: &mut Vec<GraphBuffer>,
        device: &WgpuDevice,
    ) -> GraphBuffer {
        let required_size = descriptor.size as _;
        let mut closest_index = None;
        let mut closest_size_diff: Option<usize> = None;
        for (idx, buffer) in free.iter().enumerate() {
            let current_size = buffer.0.descriptor.size as _;
            if current_size >= required_size {
                let size_diff = usize::abs_diff(current_size, required_size);

                if closest_size_diff.map_or(true, |diff| size_diff < diff) {
                    closest_index = Some(idx);
                    closest_size_diff = Some(size_diff);
                }
            }
        }

        if std::env::var("RATCHET_DEBUG").is_ok() {
            return GraphBuffer::from(self.create_buffer(&descriptor, device));
        }

        let result = match closest_index {
            Some(idx) => free.remove(idx),
            None => GraphBuffer::from(self.create_buffer(&descriptor, device)),
        };
        result
    }

    /// # Inplace operations
    ///
    /// If an operation supports inplace, we need to "lease" the buffer
    /// from the actual source (i.e the first non-inplace operation)
    ///
    /// On what conditions do we terminate the upward traversal?
    /// 1. We reach an operation that does not support inplace
    /// 2. We reach an operation that has more than one consumer
    /// 3. We reach an operation that has more than one source (this condition is wrong)
    fn determine_tensor_source<'a>(source: &'a Tensor, execution_order: &[Tensor]) -> &'a Tensor {
        let mut true_source = source;
        loop {
            let cant_inplace = !true_source.op().supports_inplace();
            //TODO: This condition isn't correct, binary can be inplace
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
            log::debug!(
                "Conditions: {:?} {:?} {:?}",
                cant_inplace,
                multiple_sources,
                multiple_consumers
            );
            if cant_inplace || multiple_sources || multiple_consumers {
                break;
            }

            true_source = true_source.op().srcs()[0];
        }
        log::debug!("Traversed to true source: {:?}", true_source.id());
        true_source
    }

    /// # Graph memory allocation
    ///
    /// Simple greedy algorithm
    /// 1. Iterate over all tensors in reverse order (leaf -> root)
    /// 2. For each tensor, loop through it's input values.
    ///     a. Assign a buffer for each input value, if it is not already assigned
    ///     b. If the input value is an inplace operation, traverse upwards until we find
    ///        the "true" buffer source (i.e the first non-inplace operation).
    /// 3. We release our **output** buffer, because the value is no longer needed,
    ///    and earlier tensors can use it.
    pub fn allocate_cfg(
        &self,
        execution_order: &[Tensor],
        device: &WgpuDevice,
    ) -> Result<FxHashMap<TensorId, GraphBuffer>, DeviceError> {
        let mut free = Vec::new(); //TODO: switch to BTreeMap
        let mut assignments = FxHashMap::default();
        //Assignments already needs all of the constants in it.
        for t in execution_order.iter().rev() {
            if t.resolved() {
                //Consts are immediately resolved
                let storage_guard = t.storage();
                let pooled = storage_guard
                    .as_ref()
                    .ok_or(AllocatorError::BufferNotFound)?
                    .try_gpu()?
                    .inner
                    .clone();
                assignments.insert(t.id(), GraphBuffer::from(pooled));
            }
        }

        for t in execution_order.iter().rev() {
            if t.resolved() {
                //Never release Consts
                continue;
            }
            log::debug!("Leasing sources for t: {:?}", t.id());

            // I need all of my sources to be allocated in order to compute my output value.
            // We "lease" the buffer, and it is released when we reach it in the execution order.
            // If the current tensor is an inplace operation,
            // we traverse upwards until we find a non-inplace operation.
            for source in t.op().srcs() {
                log::debug!("Processing source: {:?}", source.id());
                let true_source = Self::determine_tensor_source(source, execution_order);
                log::debug!("Inserting assingment: {:?}", true_source.id());
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
                let just_allocated = &assignments[&true_source.id()];
                log::debug!(
                    "Assigned: {:?} -> {:?}",
                    true_source.id(),
                    just_allocated.inner().global_id(),
                );

                if true_source.id() != source.id() {
                    log::debug!(
                        "Double Assignment: {:?} -> {:?}",
                        source.id(),
                        just_allocated.inner().global_id(),
                    );
                    assignments.insert(source.id(), GraphBuffer::from(just_allocated.clone()));
                }
            }

            //My buffer is no longer needed, since we traverse in reverse order
            //Earlier tensors can use my buffer
            if let Some(buf) = assignments.get(&t.id()) {
                log::debug!(
                    "Tensor: {:?} refcount: {}",
                    t.id(),
                    Arc::strong_count(buf.inner())
                );
                //if value == 1, he's the last one and we can release
                if Arc::strong_count(buf.inner()) == 1 {
                    log::debug!("Releasing buffer: {:?}", buf.inner().global_id());
                    free.push(buf.clone());
                }
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

// We currently use a 2nd arc on top of the pool
// to track graph allocations
#[derive(Clone)]
pub struct GraphBuffer(Arc<PooledGPUBuffer>);

impl GraphBuffer {
    pub fn inner(&self) -> &Arc<PooledGPUBuffer> {
        &self.0
    }
}

impl From<PooledGPUBuffer> for GraphBuffer {
    fn from(buf: PooledGPUBuffer) -> Self {
        Self(buf.into())
    }
}
