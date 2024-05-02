use std::num::NonZeroU64;

use crate::{
    gpu::{BindGroupEntry, BindGroupLayoutDescriptor},
    rvec, OperationError,
};

use super::{BindGroupDescriptor, GpuBindGroup, PooledGPUBuffer, WgpuDevice};
use encase::DynamicUniformBuffer;

///We use a single uniform buffer for all operations to hold their parameters.
///Every operation writes its metadata into this buffer, and an offset is returned.
///This offset is used when binding the buffer.
pub struct CpuUniform(DynamicUniformBuffer<Vec<u8>>);

///Uniforms must be 256-byte aligned, encase handles this for us.
pub const UNIFORM_ALIGN: usize = 256;
pub const DEFAULT_UNIFORM_SIZE: usize = 16384;

impl Default for CpuUniform {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuUniform {
    pub fn new() -> Self {
        Self(DynamicUniformBuffer::new(Vec::with_capacity(
            DEFAULT_UNIFORM_SIZE,
        )))
    }

    pub fn into_inner(self) -> Vec<u8> {
        self.0.into_inner()
    }

    /// Consumes the CPU repr of the uniform buffer and writes to the GPU.
    pub fn into_gpu(self, device: &WgpuDevice) -> Result<GpuUniform, OperationError> {
        let buf = device.create_uniform_init(self);
        let layout =
            device.get_or_create_bind_group_layout(&BindGroupLayoutDescriptor::uniform())?;
        let bind_group = device.get_or_create_bind_group(&BindGroupDescriptor {
            entries: rvec![BindGroupEntry {
                handle: buf.handle,
                offset: 0,
                size: NonZeroU64::new(UNIFORM_ALIGN as u64),
            }],
            layout,
        })?;

        Ok(GpuUniform { buf, bind_group })
    }
}

pub struct GpuUniform {
    buf: PooledGPUBuffer,
    bind_group: GpuBindGroup,
}

impl GpuUniform {
    pub fn bind_group(&self) -> &GpuBindGroup {
        &self.bind_group
    }
}

impl std::ops::Deref for CpuUniform {
    type Target = DynamicUniformBuffer<Vec<u8>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for CpuUniform {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
