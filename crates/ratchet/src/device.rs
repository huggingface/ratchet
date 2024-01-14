use crate::gpu::{PoolError, WgpuDevice};

#[derive(Debug, thiserror::Error)]
pub enum DeviceError {
    #[error("Failed to acquire device with error: {0:?}")]
    DeviceAcquisitionFailed(#[from] wgpu::RequestDeviceError),
    #[error("Failed to get adapter.")]
    AdapterRequestFailed,
    #[error("Failed to create storage with error: {0:?}")]
    StorageCreationFailed(#[from] PoolError), //shouldn't be PoolError
    #[error("Device mismatch, requested device: {0:?}, actual device: {1:?}")]
    DeviceMismatch(String, String),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub enum Device {
    #[default]
    CPU,
    GPU(WgpuDevice),
}

impl Device {
    pub fn label(&self) -> String {
        match self {
            Device::CPU => "CPU".to_string(),
            Device::GPU(gpu) => format!("GPU:{}", gpu.ordinal()),
        }
    }

    pub fn get_gpu(&self) -> Result<WgpuDevice, DeviceError> {
        match self {
            Device::CPU => Err(DeviceError::DeviceMismatch(
                "CPU".to_string(),
                "GPU".to_string(),
            )),
            Device::GPU(gpu) => Ok(gpu.clone()),
        }
    }
}
