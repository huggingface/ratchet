use crate::{
    gpu::GPUBuffer,
    gpu::{BufferDescriptor, BufferUsagesExt, WgpuDevice},
    storage::{RawCPUBuffer, Storable},
    Device, DeviceError, Shape, TensorDType,
};
use bytemuck::NoUninit;
use half::{bf16, f16};
use std::{alloc::Layout, fmt::Debug};
use wgpu::BufferUsages;

use crate::DType;

#[derive(Clone, derive_new::new)]
pub struct RawGPUBuffer {
    pub(crate) buf: GPUBuffer,
}

impl RawGPUBuffer {
    pub fn from_slice<T: TensorDType>(data: &[T], shape: &Shape, device: &WgpuDevice) -> Self {
        assert_eq!(data.len(), shape.numel());
        let bytes: &[u8] = bytemuck::cast_slice(data);

        let mut min_bytes = [0; 16];
        let bytes = if bytes.len() < 16 {
            min_bytes[..bytes.len()].copy_from_slice(bytes);
            &min_bytes
        } else {
            bytes
        };
        let buffer = device
            .create_buffer_init(
                &BufferDescriptor::new(
                    bytes.len() as _,
                    BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                    false,
                ),
                bytes,
            )
            .unwrap();
        device.queue().submit(None);
        device.poll(wgpu::Maintain::Wait);
        Self { buf: buffer }
    }

    pub fn inner(&self) -> &GPUBuffer {
        &self.buf
    }

    pub fn usage(&self) -> BufferUsages {
        self.buf.usage()
    }
}

impl std::fmt::Debug for RawGPUBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GPUStorage")
            .field("buffer", &self.buf.global_id())
            .field("size", &self.buf.size())
            .field("usage", &self.buf.usage())
            .finish()
    }
}

impl PartialEq for RawGPUBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.buf.global_id() == other.buf.global_id()
    }
}

impl Storable for RawGPUBuffer {
    fn to_device(self, _: &WgpuDevice) -> Result<RawGPUBuffer, DeviceError> {
        Ok(self)
    }

    fn to_cpu(self) -> RawCPUBuffer {
        todo!()
    }

    fn n_bytes(&self) -> usize {
        self.buf.size() as usize
    }

    fn dump(&self, _: DType, _: bool) -> String {
        format!("{:?}", self)
    }
}
