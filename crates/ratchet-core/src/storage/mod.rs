mod cpu_buffer;
mod gpu_buffer;

use std::io::{BufRead, Seek};

use bytemuck::NoUninit;
pub use cpu_buffer::*;
pub use gpu_buffer::*;

use crate::{Device, DeviceError, Shape, TensorDType};

use crate::DType;

#[derive(Debug)]
pub enum Storage {
    CPU(CPUBuffer),
    GPU(GPUBuffer),
}

impl Storage {
    pub unsafe fn from_quantized<T: NoUninit>(data: &[T], device: &Device) -> Self {
        match device {
            Device::CPU => Storage::CPU(unsafe { CPUBuffer::from_quantized(data) }),
            Device::GPU(g) => Storage::GPU(unsafe { GPUBuffer::from_quantized(data, g) }),
        }
    }

    pub fn from_disk<T: TensorDType, R: BufRead + Seek>(
        reader: &mut R,
        shape: &Shape,
        device: &Device,
    ) -> Result<Self, DeviceError> {
        match device {
            Device::CPU => Ok(Storage::CPU(CPUBuffer::from_disk::<T, R>(reader, shape)?)),
            Device::GPU(_) => Ok(Storage::GPU(GPUBuffer::from_disk::<T, R>(
                reader, shape, device,
            )?)),
        }
    }

    pub fn zeros<T: TensorDType>(shape: &Shape, device: &Device) -> Self {
        match device {
            Device::CPU => Storage::CPU(CPUBuffer::zeros::<T>(shape)),
            Device::GPU(g) => Storage::GPU(GPUBuffer::zeros::<T>(shape, g)),
        }
    }

    pub fn from_slice<T: NoUninit>(data: &[T], shape: &Shape, device: &Device) -> Self {
        match device {
            Device::CPU => Storage::CPU(CPUBuffer::from_slice(data, shape)),
            Device::GPU(g) => Storage::GPU(GPUBuffer::from_slice(data, shape, g)),
        }
    }

    pub fn from_bytes(data: &[u8], alignment: usize, device: &Device) -> Self {
        match device {
            Device::CPU => Storage::CPU(CPUBuffer::from_bytes(data, alignment)),
            Device::GPU(g) => Storage::GPU(GPUBuffer::from_bytes(data, alignment, g)),
        }
    }

    pub unsafe fn into_bytes(self) -> Vec<u8> {
        match self {
            Storage::CPU(c) => unsafe { c.into_bytes() },
            _ => todo!(),
        }
    }

    pub fn dump(&self, dt: DType, full: bool) -> String {
        match self {
            Storage::CPU(c) => c.dump(dt, full),
            Storage::GPU(g) => g.dump(dt, full),
        }
    }

    pub fn try_cpu(&self) -> Result<&CPUBuffer, DeviceError> {
        match self {
            Storage::CPU(c) => Ok(c),
            Storage::GPU(_g) => Err(DeviceError::DeviceMismatch(
                "CPU".to_string(),
                "GPU".to_string(),
            )),
        }
    }

    pub fn try_gpu(&self) -> Result<&GPUBuffer, DeviceError> {
        match self {
            Storage::GPU(g) => Ok(g),
            Storage::CPU(_c) => Err(DeviceError::DeviceMismatch(
                "GPU".to_string(),
                "CPU".to_string(),
            )),
        }
    }

    pub fn deep_clone(&self, device: &Device) -> Result<Self, DeviceError> {
        match self {
            Storage::CPU(c) => {
                assert!(device.is_cpu());
                Ok(Storage::CPU(c.deep_clone()?))
            }
            Storage::GPU(g) => {
                let wgpu_device = device.try_gpu()?;
                Ok(Storage::GPU(g.deep_clone(wgpu_device)))
            }
        }
    }

    #[cfg(feature = "plotting")]
    pub fn plot_fmt(&self) -> String {
        match self {
            Storage::CPU(c) => c.plot_fmt(),
            Storage::GPU(g) => g.plot_fmt(),
        }
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait)]
pub trait DeviceStorage: std::fmt::Debug + Clone + 'static {
    // To be expanded to other devices
    fn to_device(&self, device: &Device) -> Result<GPUBuffer, DeviceError>;
    /// Creates a copy of the device buffer on the CPU
    #[cfg(target_arch = "wasm32")]
    async fn to_cpu(&self, device: &Device) -> Result<CPUBuffer, DeviceError>;
    #[cfg(not(target_arch = "wasm32"))]
    fn to_cpu(&self, device: &Device) -> Result<CPUBuffer, DeviceError>;
    fn n_bytes(&self) -> usize;
    fn dump(&self, dt: DType, full: bool) -> String;
}
