use bytemuck::{NoUninit, Pod};
use half::{bf16, f16};

use crate::{storage::DeviceStorage, Device, DeviceError, GPUBuffer, Shape, TensorDType};

use std::{alloc::Layout, fmt::Debug, mem::MaybeUninit, sync::Arc};

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

    pub fn into_bytes(self) -> Vec<u8> {
        let x = unsafe { Vec::from_raw_parts(self.0, self.1.size(), self.1.size()) };
        std::mem::forget(self);
        x
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
        Self(data, layout)
    }
}

impl Clone for RawCPUBuffer {
    fn clone(&self) -> Self {
        let mut new = Self::uninitialized(self.n_bytes(), self.1.align());
        new.as_bytes_mut().copy_from_slice(self.as_bytes());
        new
    }
}

impl Drop for RawCPUBuffer {
    fn drop(&mut self) {
        if !self.0.is_null() && self.1.size() > 0 {
            unsafe { std::alloc::dealloc(self.0, self.1) }
        }
    }
}

/// Managed CPU buffer
#[derive(Debug, Clone)]
pub struct CPUBuffer {
    inner: Arc<RawCPUBuffer>,
}

unsafe impl Send for CPUBuffer {}
unsafe impl Sync for CPUBuffer {}

impl CPUBuffer {
    pub fn new(inner: RawCPUBuffer) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    /// # Safety
    ///
    /// We don't check the provided shape here.
    /// The caller should ensure that this data is laid out correctly.
    /// We also require that all of the elements have the same alignment.
    pub unsafe fn from_quantized<T: NoUninit>(data: &[T]) -> Self {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        Self::from_bytes(bytes, std::mem::align_of::<T>())
    }

    pub fn from_slice<T: NoUninit>(data: &[T], shape: &Shape) -> Self {
        assert_eq!(data.len(), shape.numel());
        let bytes: &[u8] = bytemuck::cast_slice(data);
        Self::from_bytes(bytes, std::mem::align_of::<T>())
    }

    pub fn to_slice<T: NoUninit + Pod>(&self, shape: &Shape) -> &[T] {
        assert_eq!(self.n_bytes(), shape.numel() * std::mem::size_of::<T>());
        bytemuck::cast_slice(self.inner().as_bytes())
    }

    pub fn zeros<T: TensorDType>(shape: &Shape) -> Self {
        let n_bytes = shape.numel() * T::dt().size_of();
        let mut raw = RawCPUBuffer::uninitialized(n_bytes, std::mem::align_of::<T>());
        raw.as_bytes_mut().fill(0);
        Self::new(raw)
    }

    pub fn from_disk<T: TensorDType, R: std::io::BufRead + std::io::Seek>(
        reader: &mut R,
        shape: &Shape,
    ) -> Result<Self, DeviceError> {
        let dt = T::dt();
        let n_bytes = shape.numel() * dt.size_of();
        let mut buf: Vec<MaybeUninit<u8>> = Vec::with_capacity(n_bytes);
        unsafe {
            buf.set_len(n_bytes);
        }
        let buf_slice =
            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n_bytes) };
        reader.read_exact(buf_slice).unwrap();
        let buf = unsafe { std::mem::transmute::<_, Vec<u8>>(buf) };
        Ok(Self::from_bytes(&buf, dt.size_of()))
    }

    pub fn inner(&self) -> &RawCPUBuffer {
        &self.inner
    }

    pub fn from_bytes(bytes: &[u8], alignment: usize) -> Self {
        let mut raw = RawCPUBuffer::uninitialized(bytes.len(), alignment);
        raw.as_bytes_mut().copy_from_slice(bytes);
        Self::new(raw)
    }

    pub unsafe fn into_bytes(self) -> Vec<u8> {
        Arc::try_unwrap(self.inner).unwrap().into_bytes()
    }

    pub fn deep_clone(&self) -> Result<Self, DeviceError> {
        Ok(Self::new((*self.inner()).clone()))
    }

    #[cfg(feature = "plotting")]
    pub fn plot_fmt(&self) -> String {
        format!("CPU: {} bytes", self.n_bytes())
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait)]
impl DeviceStorage for CPUBuffer {
    fn to_device(&self, device: &Device) -> Result<GPUBuffer, DeviceError> {
        let gpu_device = device.try_gpu()?;
        let bytes = self.inner().as_bytes();
        let layout = self.inner().1;
        Ok(GPUBuffer::from_bytes(bytes, layout.align(), gpu_device))
    }

    #[cfg(target_arch = "wasm32")]
    async fn to_cpu(&self, _device: &Device) -> Result<CPUBuffer, DeviceError> {
        Ok(self.clone())
    }

    #[cfg(not(target_arch = "wasm32"))]
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
                format!(
                    "{:?} ... {:?}",
                    &data[..length],
                    &data[data.len() - length..]
                )
            }
        }
        // TODO: Implement dump for quantized types
        match dtype {
            DType::F32 => dump_inner(bytemuck::cast_slice::<u8, f32>(bytes), full),
            DType::I32 => dump_inner(bytemuck::cast_slice::<u8, i32>(bytes), full),
            DType::U32 => dump_inner(bytemuck::cast_slice::<u8, u32>(bytes), full),
            DType::F16 => dump_inner(bytemuck::cast_slice::<u8, f16>(bytes), full),
            DType::BF16 => dump_inner(bytemuck::cast_slice::<u8, bf16>(bytes), full),
            dt => format!("(dump not implemented for {dt})"),
        }
    }
}
