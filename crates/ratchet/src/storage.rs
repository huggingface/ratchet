use crate::{
    gpu::GPUBuffer,
    gpu::{BufferDescriptor, WgpuDevice},
    Device, DeviceError, Shape, Strides, TensorDType,
};
use derive_new::new;
use std::{alloc::Layout, fmt::Debug, ops::RangeBounds};
use wgpu::{Buffer, BufferUsages};

use crate::DType;

/// Storage is a wrapper around a raw buffer.
///
/// Storage provides the interpretation of the raw buffer.
/// DType, Shape & Strides can all be thought of as a lens through which the raw buffer is viewed.
#[derive(new, Debug)]
pub struct Storage {
    device: Device,
    dtype: DType,
    shape: Shape,
    strides: Strides,
    raw: Option<RawStorage>,
}

impl Storage {}

#[derive(Debug)]
pub enum RawStorage {
    CPU(RawCPUBuffer),
    GPU(RawGPUBuffer),
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
    pub fn inner(&self) -> (*mut u8, Layout) {
        (self.0, self.1)
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.0, self.1.size()) }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.0, self.1.size()) }
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
        Ok(RawGPUBuffer(buffer))
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

#[derive(Clone)]
pub struct RawGPUBuffer(GPUBuffer);

impl From<GPUBuffer> for RawGPUBuffer {
    fn from(b: GPUBuffer) -> Self {
        Self(b)
    }
}

impl std::fmt::Debug for RawGPUBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GPUStorage")
            .field("buffer", &self.0.global_id())
            .field("size", &self.0.size())
            .field("usage", &self.0.usage())
            .finish()
    }
}

impl PartialEq for RawGPUBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.0.global_id() == other.0.global_id()
    }
}

impl RawGPUBuffer {
    pub fn new(buffer: GPUBuffer) -> Self {
        Self(buffer)
    }

    pub fn inner(&self) -> &GPUBuffer {
        &self.0
    }

    pub fn set_inner(&mut self, b: GPUBuffer) {
        self.0 = b;
    }

    pub fn as_entire_binding(&self) -> wgpu::BindingResource {
        self.0.as_entire_binding()
    }

    pub fn usage(&self) -> wgpu::BufferUsages {
        self.0.usage()
    }

    pub fn slice<S: RangeBounds<wgpu::BufferAddress>>(&self, bounds: S) -> wgpu::BufferSlice {
        self.0.slice(bounds)
    }

    pub fn unmap(&self) {
        self.0.unmap();
    }

    pub fn buffer_id(&self) -> wgpu::Id<Buffer> {
        self.0.global_id()
    }

    pub fn size(&self) -> wgpu::BufferAddress {
        self.0.size()
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
        self.0.size() as usize
    }

    fn dump(&self, _: DType, _: bool) -> String {
        format!("{:?}", self)
    }
}
