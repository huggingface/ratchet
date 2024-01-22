use bytemuck::NoUninit;

use crate::{storage::DeviceStorage, Device, DeviceError, GPUBuffer, Shape, TensorDType};

use std::{alloc::Layout, fmt::Debug};

use crate::DType;

#[derive(derive_new::new, Debug, PartialEq, Eq)]
pub struct RawCPUBuffer(*mut u8, Layout);

impl RawCPUBuffer {
    pub fn into_raw_parts(&self) -> (*mut u8, Layout) {
        (self.0, self.1)
    }

    pub fn n_bytes(&self) -> usize {
        self.1.size()
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.0, self.1.size()) }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.0, self.1.size()) }
    }

    pub fn uninitialized(size: usize, alignment: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(size, alignment).unwrap();
        let data = if size == 0 {
            std::ptr::null()
        } else {
            let ptr = unsafe { std::alloc::alloc(layout) };
            assert!(!ptr.is_null());
            ptr
        } as *mut u8;
        println!("Unintialized: {:p}", data);
        Self(data, layout)
    }
}

impl Clone for RawCPUBuffer {
    fn clone(&self) -> Self {
        let (ptr, layout) = self.into_raw_parts();
        let data = if layout.size() == 0 {
            std::ptr::null()
        } else {
            let ptr = unsafe { std::alloc::alloc(layout) };
            assert!(!ptr.is_null());
            ptr
        } as *mut u8;
        println!("Cloning: {:p} -> {:p}", ptr, data);
        unsafe { ptr.copy_to_nonoverlapping(data, layout.size()) };
        Self(data, layout)
    }
}

impl Drop for RawCPUBuffer {
    fn drop(&mut self) {
        if !self.0.is_null() && self.1.size() > 0 {
            println!("DROPPING: {:p}", self.0);
            unsafe { std::alloc::dealloc(self.0, self.1) }
        }
    }
}

/// Managed CPU buffer
#[derive(Debug, Clone, derive_new::new)]
pub struct CPUBuffer {
    inner: RawCPUBuffer,
}

unsafe impl Send for CPUBuffer {}
unsafe impl Sync for CPUBuffer {}

impl CPUBuffer {
    pub fn from_slice<T: NoUninit>(data: &[T], shape: &Shape) -> Self {
        assert_eq!(data.len(), shape.numel());
        let bytes: &[u8] = bytemuck::cast_slice(data);
        Self::from_bytes(bytes, std::mem::align_of::<T>())
    }

    pub fn inner(&self) -> &RawCPUBuffer {
        &self.inner
    }

    pub fn from_bytes(bytes: &[u8], alignment: usize) -> Self {
        let mut raw = RawCPUBuffer::uninitialized(bytes.len(), alignment);
        raw.as_bytes_mut().copy_from_slice(bytes);
        Self::from(raw)
    }

    pub fn deep_clone(&self) -> Result<Self, DeviceError> {
        Ok(Self::from(self.inner().clone()))
    }
}

impl From<RawCPUBuffer> for CPUBuffer {
    fn from(raw: RawCPUBuffer) -> Self {
        CPUBuffer { inner: raw }
    }
}

impl DeviceStorage for CPUBuffer {
    fn to_device(&self, device: &Device) -> Result<GPUBuffer, DeviceError> {
        let gpu_device = device.try_gpu()?;
        let raw = self.inner();
        let (ptr, layout) = raw.into_raw_parts();
        let bytes = unsafe { std::slice::from_raw_parts(ptr, layout.size()) };
        Ok(GPUBuffer::from_bytes(bytes, layout.align(), gpu_device))
    }

    fn to_cpu(&self, _device: &Device) -> Result<CPUBuffer, DeviceError> {
        Ok(self.clone())
    }

    fn n_bytes(&self) -> usize {
        self.inner().n_bytes()
    }

    fn dump(&self, dtype: DType, full: bool) -> String {
        let bytes = self.inner().as_bytes();

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
