use crate::gpu::{BufferDescriptor, BufferPool, GPUBuffer, GpuBufferHandle};
use std::cell::{Ref, RefCell, RefMut};

use super::WgpuDevice;

pub struct BufferAllocator {
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

    pub fn create_buffer(&self, desc: &BufferDescriptor, device: &WgpuDevice) -> GPUBuffer {
        self.pool.borrow_mut().allocate(desc, device)
    }

    pub fn pool(&self) -> Ref<BufferPool> {
        self.pool.borrow()
    }

    pub fn pool_mut(&self) -> RefMut<BufferPool> {
        self.pool.borrow_mut()
    }

    pub fn create_buffer_init(
        &self,
        desc: &BufferDescriptor,
        contents: &[u8],
        device: &WgpuDevice,
    ) -> GPUBuffer {
        let buf = self.pool.borrow_mut().allocate(desc, device);
        device.queue().write_buffer(&buf.inner, 0, contents);
        buf
    }

    // A greedy algorithm to allocate the minimum number of storage buffers
    // for intermediate values during an inference pass of the CFG.
    pub fn allocate_storage() {
        todo!()
    }
}
