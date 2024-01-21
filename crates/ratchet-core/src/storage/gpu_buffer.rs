use crate::{
    gpu::{BufferDescriptor, WgpuDevice},
    gpu::{BufferUsagesExt, PooledGPUBuffer},
    storage::{CPUBuffer, DeviceStorage},
    Device, DeviceError, Shape,
};

use bytemuck::NoUninit;
use wgpu::BufferUsages;

use crate::DType;

#[derive(Clone, Debug, derive_new::new)]
pub struct GPUBuffer {
    pub(crate) inner: PooledGPUBuffer,
    pub(crate) alignment: usize,
}

impl GPUBuffer {
    const MIN_SIZE: usize = 16;

    pub fn from_slice<T: NoUninit>(data: &[T], shape: &Shape, device: &WgpuDevice) -> Self {
        assert_eq!(data.len(), shape.numel());
        Self::from_bytes(
            bytemuck::cast_slice(data),
            std::mem::align_of::<T>(),
            device,
        )
    }

    pub(crate) fn from_bytes(bytes: &[u8], alignment: usize, device: &WgpuDevice) -> Self {
        let num_bytes = bytes.len();
        let mut min_bytes = [0; Self::MIN_SIZE];
        let bytes = if num_bytes < Self::MIN_SIZE {
            min_bytes[..num_bytes].copy_from_slice(bytes);
            &min_bytes
        } else {
            bytes
        };
        let inner = device
            .create_buffer_init(
                &BufferDescriptor::new(bytes.len() as _, BufferUsages::standard(), false),
                bytes,
            )
            .unwrap();
        device.queue().submit(None);
        device.poll(wgpu::Maintain::Wait);
        Self { inner, alignment }
    }

    /// Returns true if the buffer has all the given usages.
    pub(crate) fn validate_usages(&self, usages: BufferUsages) -> Result<(), DeviceError> {
        match self.inner.usage().contains(usages) {
            true => Ok(()),
            false => Err(DeviceError::InvalidBufferUsage(self.inner.usage(), usages)),
        }
    }

    pub fn inner(&self) -> &PooledGPUBuffer {
        &self.inner
    }

    pub fn usage(&self) -> BufferUsages {
        self.inner.usage()
    }
}

impl DeviceStorage for GPUBuffer {
    fn to_device(&self, _: &Device) -> Result<GPUBuffer, DeviceError> {
        Ok(self.clone())
    }

    fn to_cpu(&self, device: &Device) -> Result<CPUBuffer, DeviceError> {
        self.validate_usages(BufferUsages::COPY_SRC)?;
        let device = device.try_gpu()?;
        let buffer_slice = self.inner.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        let alignment = self.alignment;

        wgpu::util::DownloadBuffer::read_buffer(
            device,
            device.queue(),
            &buffer_slice,
            move |buffer| {
                tx.send(match buffer {
                    Ok(db) => Ok(CPUBuffer::from_bytes(&db, alignment)),
                    Err(error) => Err(error),
                })
                .expect("Failed to send result of read_buffer");
            },
        );
        device.poll(wgpu::Maintain::Wait);
        let storage = rx.recv().unwrap()?;

        Ok(storage)
    }

    fn n_bytes(&self) -> usize {
        self.inner.size() as usize
    }

    fn dump(&self, _: DType, _: bool) -> String {
        format!("{:?}", self)
    }
}
