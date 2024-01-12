use crate::{
    gpu::{BufferDescriptor, WgpuDevice},
    gpu::{DeviceError, GPUBuffer},
    DataType,
};
use std::{alloc::Layout, fmt::Debug, ops::RangeBounds};
use wgpu::{Buffer, BufferUsages};

use crate::DType;

pub enum Storage {
    CPU(CPUStorage),
    GPU(GPUStorage),
}

pub trait Storable: Debug + Clone + 'static {
    fn to_device(self, device: &WgpuDevice) -> Result<GPUStorage, DeviceError>;
    fn to_cpu(self) -> CPUStorage;
    fn n_bytes(&self) -> usize;
    fn dump(&self, dt: DType, full: bool) -> String;
}

#[derive(derive_new::new, Debug, PartialEq, Eq)]
pub struct CPUStorage(*mut u8, Layout);

impl CPUStorage {
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

impl Clone for CPUStorage {
    fn clone(&self) -> Self {
        let (ptr, layout) = self.inner();
        let alloc = unsafe { std::alloc::alloc(layout) };
        unsafe { ptr.copy_to_nonoverlapping(alloc, layout.size()) };

        Self(alloc, layout)
    }
}

impl Drop for CPUStorage {
    fn drop(&mut self) {
        if !self.0.is_null() && self.1.size() > 0 {
            unsafe { std::alloc::dealloc(self.0, self.1) }
        }
    }
}

impl Storable for CPUStorage {
    fn to_device(self, device: &WgpuDevice) -> Result<GPUStorage, DeviceError> {
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
        Ok(GPUStorage(buffer))
    }

    fn to_cpu(self) -> CPUStorage {
        self
    }

    fn n_bytes(&self) -> usize {
        self.1.size()
    }

    fn dump(&self, dtype: DType, full: bool) -> String {
        let bytes = unsafe { std::slice::from_raw_parts(self.0, self.1.size()) };

        fn dump_inner<T: DataType>(data: &[T], full: bool) -> String {
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
pub struct GPUStorage(GPUBuffer);

impl From<GPUBuffer> for GPUStorage {
    fn from(b: GPUBuffer) -> Self {
        Self(b)
    }
}

impl std::fmt::Debug for GPUStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GPUStorage")
            .field("buffer", &self.0.global_id())
            .field("size", &self.0.size())
            .field("usage", &self.0.usage())
            .finish()
    }
}

impl PartialEq for GPUStorage {
    fn eq(&self, other: &Self) -> bool {
        self.0.global_id() == other.0.global_id()
    }
}

impl GPUStorage {
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

impl Storable for GPUStorage {
    fn to_device(self, _: &WgpuDevice) -> Result<GPUStorage, DeviceError> {
        Ok(self)
    }

    fn to_cpu(self) -> CPUStorage {
        todo!()
    }

    fn n_bytes(&self) -> usize {
        self.0.size() as usize
    }

    fn dump(&self, _: DType, _: bool) -> String {
        format!("{:?}", self)
    }
}
