use crate::{
    gpu::{BufferDescriptor, WgpuDevice},
    gpu::{BufferUsagesExt, GPUBuffer},
    storage::{DeviceStorage, RawCPUBuffer},
    Device, DeviceError, Shape, TensorDType, TensorError,
};

use wgpu::BufferUsages;

use crate::DType;

#[derive(Clone, derive_new::new)]
pub struct RawGPUBuffer {
    pub(crate) inner: GPUBuffer,
    pub(crate) alignment: usize,
}

impl RawGPUBuffer {
    const MIN_SIZE: usize = 16;

    pub fn from_slice<T: TensorDType>(data: &[T], shape: &Shape, device: &WgpuDevice) -> Self {
        assert_eq!(data.len(), shape.numel());
        Self::from_bytes(bytemuck::cast_slice(data), T::dt().size_of(), device)
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
        let buffer = device
            .create_buffer_init(
                &BufferDescriptor::new(bytes.len() as _, BufferUsages::standard(), false),
                bytes,
            )
            .unwrap();
        device.queue().submit(None);
        device.poll(wgpu::Maintain::Wait);
        Self {
            inner: buffer,
            alignment,
        }
    }

    /// Returns true if the buffer has all the given usages.
    pub(crate) fn validate_usages(&self, usages: BufferUsages) -> Result<(), DeviceError> {
        match self.inner.usage().contains(usages) {
            true => Ok(()),
            false => Err(DeviceError::InvalidBufferUsage(self.inner.usage(), usages)),
        }
    }

    pub fn inner(&self) -> &GPUBuffer {
        &self.inner
    }

    pub fn usage(&self) -> BufferUsages {
        self.inner.usage()
    }
}

impl std::fmt::Debug for RawGPUBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawGPUBuffer")
            .field("buf", &self.inner.global_id())
            .finish()
    }
}

impl PartialEq for RawGPUBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.inner.global_id() == other.inner.global_id()
    }
}

impl DeviceStorage for RawGPUBuffer {
    fn to_device(self, _: &Device) -> Result<RawGPUBuffer, DeviceError> {
        Ok(self)
    }

    fn to_cpu(&self, device: &Device) -> Result<RawCPUBuffer, DeviceError> {
        self.validate_usages(BufferUsages::COPY_SRC)?;
        let device = device.try_gpu()?;
        let buffer_slice = self.inner.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        let alignment = self.alignment;

        wgpu::util::DownloadBuffer::read_buffer(
            device,
            device.queue(),
            &buffer_slice,
            move |buffer| {
                tx.send(match buffer {
                    Ok(db) => Ok(RawCPUBuffer::from_bytes(&db, alignment)),
                    Err(error) => Err(error),
                })
                .expect("Failed to send result of read_buffer");
            },
        );
        device.poll(wgpu::Maintain::Wait);
        //TODO: fix unwrap
        let storage = pollster::block_on(async { rx.receive().await })
            .ok_or(TensorError::TransferError)
            .unwrap()
            .map_err(|_| TensorError::TransferError)
            .unwrap();

        Ok(storage)
    }

    fn n_bytes(&self) -> usize {
        self.inner.size() as usize
    }

    fn dump(&self, _: DType, _: bool) -> String {
        format!("{:?}", self)
    }
}
