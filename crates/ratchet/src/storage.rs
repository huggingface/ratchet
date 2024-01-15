use crate::{
    gpu::GPUBuffer,
    gpu::{BufferDescriptor, WgpuDevice},
    Device, DeviceError, Shape, TensorDType,
};
use std::{alloc::Layout, fmt::Debug};
use wgpu::{util::DownloadBuffer, BufferUsages};

use crate::DType;

/// Storage is a wrapper around a raw buffer.
/// RawStorage is Optional as the tensor may not be resolved.
#[derive(derive_new::new, Debug)]
pub struct Storage {
    raw: Option<RawStorage>,
}

unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

impl Storage {
    pub fn from_slice<T: TensorDType>(data: &[T], shape: &Shape, device: &Device) -> Self {
        assert_eq!(data.len(), shape.numel());
        let bytes: &[u8] = bytemuck::cast_slice(data);
        match device {
            Device::CPU => {
                let mut storage =
                    unsafe { RawCPUBuffer::uninitialized(bytes.len(), T::dt().size_of()) };
                storage.as_bytes_mut().copy_from_slice(bytes);
                let raw = Some(RawStorage::CPU(storage));
                Self { raw }
            }
            Device::GPU(wgpu_device) => {
                let raw = Some(RawStorage::GPU(RawGPUBuffer::from_slice(
                    data,
                    shape,
                    wgpu_device,
                )));
                Self { raw }
            }
        }
    }

    pub fn set_raw(&mut self, raw: RawStorage) {
        self.raw = Some(raw);
    }

    pub fn raw(&self) -> Option<&RawStorage> {
        self.raw.as_ref()
    }

    pub fn try_gpu(&self) -> Option<&GPUBuffer> {
        match self.raw.as_ref()? {
            RawStorage::GPU(raw) => Some(&raw.buf),
            _ => None,
        }
    }

    pub fn dump(&self, dtype: DType, full: bool) -> String {
        self.raw
            .as_ref()
            .map(|raw| match raw {
                RawStorage::CPU(raw) => raw.dump(dtype, full),
                RawStorage::GPU(raw) => raw.dump(dtype, full),
            })
            .unwrap_or_else(|| "None".to_string())
    }
}

#[derive(Debug)]
pub enum RawStorage {
    CPU(RawCPUBuffer),
    GPU(RawGPUBuffer),
}

impl From<GPUBuffer> for RawStorage {
    fn from(buf: GPUBuffer) -> Self {
        Self::GPU(RawGPUBuffer { buf })
    }
}

pub trait Storable: Debug + Clone + 'static {
    // To be expanded to other devices
    fn to_device(self, device: &WgpuDevice) -> Result<RawGPUBuffer, DeviceError>;
    fn to_cpu(self) -> RawCPUBuffer;
    fn n_bytes(&self) -> usize;
    fn dump(&self, dt: DType, full: bool) -> String;
}

#[derive(derive_new::new, Debug, PartialEq, Eq)]
pub struct RawCPUBuffer(*mut u8, Layout);

impl RawCPUBuffer {
    unsafe fn uninitialized(size: usize, alignment: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(size, alignment).unwrap();
        let data = if size == 0 {
            std::ptr::null()
        } else {
            let ptr = std::alloc::alloc(layout);
            assert!(!ptr.is_null());
            ptr
        } as *mut u8;
        Self(data, layout)
    }

    pub fn inner(&self) -> (*mut u8, Layout) {
        (self.0, self.1)
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.0, self.1.size()) }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.0, self.1.size()) }
    }

    pub fn from_bytes(bytes: &[u8], alignment: usize) -> Self {
        let mut storage = unsafe { Self::uninitialized(bytes.len(), alignment) };
        storage.as_bytes_mut().copy_from_slice(bytes);
        storage
    }
}

impl Clone for RawCPUBuffer {
    fn clone(&self) -> Self {
        let (ptr, layout) = self.inner();
        let alloc = unsafe { std::alloc::alloc(layout) };
        unsafe { ptr.copy_to_nonoverlapping(alloc, layout.size()) };

        Self(alloc, layout)
    }
}

impl Drop for RawCPUBuffer {
    fn drop(&mut self) {
        if !self.0.is_null() && self.1.size() > 0 {
            unsafe { std::alloc::dealloc(self.0, self.1) }
        }
    }
}

impl Storable for RawCPUBuffer {
    fn to_device(self, device: &WgpuDevice) -> Result<RawGPUBuffer, DeviceError> {
        let mut min_bytes = [0; 16];
        let bytes = if self.as_bytes().len() < 16 {
            min_bytes[..self.as_bytes().len()].copy_from_slice(self.as_bytes());
            &min_bytes
        } else {
            self.as_bytes()
        };

        let buffer = device.create_buffer_init(
            &BufferDescriptor::new(
                bytes.len() as _,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                false,
            ),
            device.queue(),
            bytes,
        )?;
        device.queue().submit(None);
        device.poll(wgpu::Maintain::Wait);
        Ok(RawGPUBuffer { buf: buffer })
    }

    fn to_cpu(self) -> RawCPUBuffer {
        self
    }

    fn n_bytes(&self) -> usize {
        self.1.size()
    }

    fn dump(&self, dtype: DType, full: bool) -> String {
        let bytes = unsafe { std::slice::from_raw_parts(self.0, self.1.size()) };

        fn dump_inner<T: TensorDType>(data: &[T], full: bool) -> String {
            let length = if data.len() < 64 { data.len() } else { 64 };
            if full {
                format!("{:?}", data)
            } else {
                format!("{:?}...{:?}", &data[..length], &data[data.len() - length..])
            }
        }
        match dtype {
            DType::F32 => dump_inner(bytemuck::cast_slice::<u8, f32>(bytes), full),
            DType::I32 => dump_inner(bytemuck::cast_slice::<u8, i32>(bytes), full),
            DType::U32 => dump_inner(bytemuck::cast_slice::<u8, u32>(bytes), full),
            _ => todo!(),
        }
    }
}

#[derive(Clone, derive_new::new)]
pub struct RawGPUBuffer {
    buf: GPUBuffer,
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
                device.queue(),
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
