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
    const MIN_SIZE: u64 = 16;

    pub fn from_slice<T: NoUninit>(data: &[T], shape: &Shape, device: &WgpuDevice) -> Self {
        assert_eq!(data.len(), shape.numel());
        Self::from_bytes(
            bytemuck::cast_slice(data),
            std::mem::align_of::<T>(),
            device,
        )
    }

    //We have to use from_bytes here, as buffers may be reused and we need to
    //ensure that the buffer is zeroed
    pub fn zeros<T: TensorDType>(shape: &Shape, device: &WgpuDevice) -> Self {
        Self::from_bytes(
            vec![0; shape.numel() * T::dt().size_of()].as_slice(),
            T::dt().size_of(),
            device,
        )
    }

    /// # Safety
    ///
    /// We don't check the provided shape here.
    /// The caller should ensure that this data is laid out correctly.
    /// We also require that all of the elements have the same alignment.
    pub unsafe fn from_quantized<T: NoUninit>(data: &[T], device: &WgpuDevice) -> Self {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        Self::from_bytes(bytes, std::mem::align_of::<T>(), device)
    }

    pub(crate) fn from_bytes(bytes: &[u8], alignment: usize, device: &WgpuDevice) -> Self {
        let inner = device
            .get_or_create_buffer_init(
                &BufferDescriptor::new(bytes.len() as _, BufferUsages::standard(), false),
                bytes.into(),
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
            .get_or_create_buffer(
                &BufferDescriptor::new(self.inner.size(), self.inner.usage(), false),
                true,
            )
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
        let storage = wgpu_buffer_to_cpu_buffer(&self.inner, self.alignment, device);
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

#[cfg(target_arch = "wasm32")]
pub async fn wgpu_buffer_to_cpu_buffer(
    src_buf: &wgpu::Buffer,
    alignment: usize,
    device: WgpuDevice,
) -> CPUBuffer {
    assert!(src_buf.usage().contains(wgpu::BufferUsages::COPY_SRC));
    let buffer_slice = src_buf.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();

    wgpu::util::DownloadBuffer::read_buffer(
        &device,
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
    rx.receive().await.unwrap().unwrap()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn wgpu_buffer_to_cpu_buffer(
    src_buf: &wgpu::Buffer,
    alignment: usize,
    device: &WgpuDevice,
) -> CPUBuffer {
    assert!(src_buf.usage().contains(wgpu::BufferUsages::COPY_SRC));
    let buffer_slice = src_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();

    wgpu::util::DownloadBuffer::read_buffer(device, device.queue(), &buffer_slice, move |buffer| {
        tx.send(match buffer {
            Ok(db) => Ok(CPUBuffer::from_bytes(&db, alignment)),
            Err(error) => Err(error),
        })
        .expect("Failed to send result of read_buffer");
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap()
}
