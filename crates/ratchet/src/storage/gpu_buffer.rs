use crate::{
    gpu::{BufferDescriptor, WgpuDevice},
    gpu::{BufferUsagesExt, GPUBuffer},
    storage::{RawCPUBuffer, Storable},
    DeviceError, Shape, TensorDType,
};

use wgpu::BufferUsages;

use crate::DType;

#[derive(Clone, derive_new::new)]
pub struct RawGPUBuffer {
    pub(crate) buf: GPUBuffer,
}

impl RawGPUBuffer {
    const MIN_SIZE: usize = 16;

    pub fn from_slice<T: TensorDType>(data: &[T], shape: &Shape, device: &WgpuDevice) -> Self {
        assert_eq!(data.len(), shape.numel());
        Self::from_bytes(bytemuck::cast_slice(data), device)
    }

    pub(crate) fn from_bytes(bytes: &[u8], device: &WgpuDevice) -> Self {
        let num_bytes = bytes.len();
        let mut min_bytes = [0; Self::MIN_SIZE];
        let bytes = if num_bytes < Self::MIN_SIZE {
            min_bytes[..num_bytes].copy_from_slice(bytes);
            &min_bytes
        } else {
            bytes
        };
        let buffer = device
            .create_buffer_init(
                &BufferDescriptor::new(bytes.len() as _, BufferUsages::standard(), false),
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
        f.debug_struct("RawGPUBuffer")
            .field("buf", &self.buf.global_id())
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
