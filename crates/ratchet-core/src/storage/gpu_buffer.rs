use crate::{
    gpu::{BufferDescriptor, WgpuDevice},
    gpu::{BufferUsagesExt, PooledGPUBuffer},
    storage::{CPUBuffer, DeviceStorage},
    Device, DeviceError, Shape, TensorDType,
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
            .get_or_create_buffer_init(
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

    #[allow(unused)]
    pub fn deep_clone(&self, device: &WgpuDevice) -> Self {
        let clone = device
            .get_or_create_buffer(&BufferDescriptor::new(
                self.inner.size(),
                self.inner.usage(),
                false,
            ))
            .unwrap();
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.inner, 0, &clone, 0, self.inner.size());
        device.queue().submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
        Self {
            inner: clone,
            alignment: self.alignment,
        }
    }

    pub fn from_disk<T: TensorDType, R: std::io::BufRead + std::io::Seek>(
        reader: &mut R,
        shape: &Shape,
        device: &Device,
    ) -> Result<Self, DeviceError> {
        //There is no faster way to do this
        CPUBuffer::from_disk::<T, R>(reader, shape)?.to_device(device)
    }

    pub fn trim_id(id: wgpu::Id<wgpu::Buffer>) -> Option<String> {
        let id = format!("{:?}", id);
        let trimmed = id.trim_start_matches("Id(").trim_end_matches(')');
        if trimmed.len() > 12 && trimmed.chars().all(|c| c.is_numeric()) {
            Some(trimmed[12..].to_string())
        } else {
            None
        }
    }

    #[cfg(feature = "plotting")]
    pub fn plot_fmt(&self) -> String {
        let id_string = Self::trim_id(self.inner().global_id()).unwrap_or_default();
        format!("GPU:#{}\n{} bytes", id_string, self.inner.size())
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait)]
impl DeviceStorage for GPUBuffer {
    fn to_device(&self, _: &Device) -> Result<GPUBuffer, DeviceError> {
        Ok(self.clone())
    }

    #[cfg(target_arch = "wasm32")]
    async fn to_cpu(&self, device: &Device) -> Result<CPUBuffer, DeviceError> {
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
                    Ok(db) => Ok(CPUBuffer::from_bytes(&db, alignment)),
                    Err(error) => Err(error),
                })
                .expect("Failed to send result of read_buffer");
            },
        );
        device.poll(wgpu::Maintain::Wait);
        Ok(rx.receive().await.unwrap()?)
    }

    #[cfg(not(target_arch = "wasm32"))]
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
        let mut result = String::new();
        let id_string = Self::trim_id(self.inner().global_id()).unwrap_or_default();
        result.push_str(&format!("GPU Buffer #{}\n", id_string));
        result.push_str(&format!("Size: {} bytes\n", self.inner.size()));
        result
    }
}
