use crate::{
    storage::{DeviceStorage, RawGPUBuffer},
    Device, DeviceError, Shape, TensorDType,
};

use std::{alloc::Layout, fmt::Debug};

use crate::DType;

#[derive(derive_new::new, Debug, PartialEq, Eq)]
pub struct RawCPUBuffer(*mut u8, Layout);

unsafe impl Send for RawCPUBuffer {}

impl RawCPUBuffer {
    pub fn from_slice<T: TensorDType>(data: &[T], shape: &Shape) -> Self {
        assert_eq!(data.len(), shape.numel());
        let bytes: &[u8] = bytemuck::cast_slice(data);
        Self::from_bytes(bytes, T::dt().size_of())
    }

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

impl DeviceStorage for RawCPUBuffer {
    fn to_device(self, device: &Device) -> Result<RawGPUBuffer, DeviceError> {
        let (bytes, align, gpu_device) = (self.as_bytes(), self.1.align(), device.try_gpu()?);
        Ok(RawGPUBuffer::from_bytes(bytes, align, gpu_device))
    }

    fn to_cpu(&self, _device: &Device) -> Result<RawCPUBuffer, DeviceError> {
        Ok(self.clone())
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
