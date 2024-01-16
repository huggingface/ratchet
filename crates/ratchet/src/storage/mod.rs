mod cpu_buffer;
mod gpu_buffer;

pub use cpu_buffer::*;
pub use gpu_buffer::*;

use crate::{gpu::GPUBuffer, Device, DeviceError, Shape, TensorDType};
use bytemuck::NoUninit;
use half::{bf16, f16};
use std::fmt::Debug;

use crate::DType;

macro_rules! impl_read_to_host {
    ($($dtype:ident => $type:ty),*) => {
        pub(crate) fn read_to_host<A: NoUninit>(shape: Shape, dt: DType, bytes: &[A]) -> Storage {
            match dt {
                $(DType::$dtype => Storage::from_slice::<$type>(bytemuck::cast_slice(bytes), &shape, &Device::CPU),)*
                _ => todo!("Attempted to read GPU tensor to host with unsupported dtype: {:?}", dt),
            }
        }
    };
}

#[derive(Debug)]
pub struct Storage {
    raw: Option<RawStorage>, //Optional as the tensor may not be resolved
}

unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

impl Storage {
    pub fn empty() -> Self {
        Self { raw: None }
    }

    pub fn from_slice<T: TensorDType>(data: &[T], shape: &Shape, device: &Device) -> Self {
        assert_eq!(data.len(), shape.numel());
        match device {
            Device::CPU => Self {
                raw: Some(RawStorage::CPU(RawCPUBuffer::from_slice(data, shape))),
            },
            Device::GPU(d) => Self {
                raw: Some(RawStorage::GPU(RawGPUBuffer::from_slice(data, shape, d))),
            },
        }
    }

    pub fn set_raw(&mut self, raw: RawStorage) {
        self.raw = Some(raw);
    }

    pub fn raw(&self) -> Option<&RawStorage> {
        self.raw.as_ref()
    }

    pub fn try_gpu(&self) -> Result<&RawGPUBuffer, DeviceError> {
        match self.raw.as_ref() {
            Some(RawStorage::GPU(raw)) => Ok(raw),
            _ => Err(DeviceError::DeviceMismatch(
                "GPU".to_string(),
                "CPU".to_string(),
            )),
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

    impl_read_to_host!(
        F32 => f32,
        I32 => i32,
        U32 => u32,
        F16 => f16,
        BF16 => bf16
    );
}

impl From<RawStorage> for Storage {
    fn from(raw: RawStorage) -> Self {
        Self { raw: Some(raw) }
    }
}

impl From<RawCPUBuffer> for Storage {
    fn from(raw: RawCPUBuffer) -> Self {
        Self {
            raw: Some(RawStorage::CPU(raw)),
        }
    }
}

impl From<RawGPUBuffer> for Storage {
    fn from(raw: RawGPUBuffer) -> Self {
        Self {
            raw: Some(RawStorage::GPU(raw)),
        }
    }
}

#[derive(Debug)]
pub enum RawStorage {
    CPU(RawCPUBuffer),
    GPU(RawGPUBuffer),
}

impl RawStorage {
    pub fn from_gpu(buf: GPUBuffer, dtype: DType) -> Self {
        RawStorage::GPU(RawGPUBuffer {
            buf,
            alignment: dtype.size_of(),
        })
    }
}

pub trait Storable: Debug + Clone + 'static {
    // To be expanded to other devices
    fn to_device(self, device: &Device) -> Result<RawGPUBuffer, DeviceError>;
    /// Creates a copy of the device buffer on the CPU
    fn to_cpu(&self, device: &Device) -> Result<RawCPUBuffer, DeviceError>;
    fn n_bytes(&self) -> usize;
    fn dump(&self, dt: DType, full: bool) -> String;
}
