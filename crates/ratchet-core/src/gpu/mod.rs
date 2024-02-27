mod buffer_allocator;
mod device;
mod pools;
mod uniform;
mod workload;

pub use buffer_allocator::*;
pub use device::*;
pub use pools::*;
pub use uniform::*;
pub use workload::*;

pub const MIN_STORAGE_BUFFER_SIZE: usize = 16;
pub const STORAGE_BUFFER_ALIGN: usize = 256;

/// Usages we use everywhere
pub trait BufferUsagesExt {
    fn standard() -> Self;
}

impl BufferUsagesExt for wgpu::BufferUsages {
    fn standard() -> Self {
        Self::COPY_DST | Self::COPY_SRC | Self::STORAGE
    }
}
