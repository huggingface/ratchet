use crate::gpu::{AllocatorError, PoolError, WgpuDevice};

#[derive(Clone, Debug, thiserror::Error)]
pub enum DeviceError {
    #[error("Failed to acquire device with error: {0:?}")]
    DeviceAcquisitionFailed(#[from] wgpu::RequestDeviceError),
    #[error("Failed to request adapter required for WebGPU. Please ensure that your browser supports WebGPU.
             (Chrome 121+, Firefox Nightly, Edge & all Chromium based browsers)")]
    AdapterRequestFailed,
    #[error("Failed to create storage with error: {0:?}")]
    StorageCreationFailed(#[from] PoolError), //TODO: shouldn't be PoolError
    #[error("Device mismatch, requested device: {0:?}, actual device: {1:?}")]
    DeviceMismatch(String, String),
    #[error("Failed to allocate buffer with error: {0:?}")]
    BufferAllocationFailed(#[from] AllocatorError),
    #[error("Invalid GPU Buffer Usage, current: {0:?}, required: {1:?}")]
    InvalidBufferUsage(wgpu::BufferUsages, wgpu::BufferUsages),
    #[error("Failed to transfer buffer with error: {0:?}")]
    BufferTransferFailed(#[from] wgpu::BufferAsyncError),
}

pub enum DeviceRequest {
    CPU,
    GPU,
}

#[derive(Clone, Default, PartialEq)]
pub enum Device {
    #[default]
    CPU,
    GPU(WgpuDevice),
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::CPU => write!(f, "CPU"),
            Device::GPU(gpu) => write!(f, "GPU:{}", gpu.ordinal()),
        }
    }
}

impl Device {
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::CPU)
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, Device::GPU(_))
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn request_device(request: DeviceRequest) -> Result<Self, DeviceError> {
        match request {
            DeviceRequest::CPU => Ok(Device::CPU),
            DeviceRequest::GPU => Ok(Device::GPU(WgpuDevice::new().await?)),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn request_device(request: DeviceRequest) -> Result<Self, DeviceError> {
        match request {
            DeviceRequest::CPU => Ok(Device::CPU),
            DeviceRequest::GPU => Ok(Device::GPU(pollster::block_on(async {
                WgpuDevice::new().await
            })?)),
        }
    }

    pub fn label(&self) -> String {
        format!("{:?}", self)
    }

    pub fn try_gpu(&self) -> Result<&WgpuDevice, DeviceError> {
        match self {
            Device::GPU(gpu) => Ok(gpu),
            Device::CPU => Err(DeviceError::DeviceMismatch(
                "GPU".to_string(),
                "CPU".to_string(),
            )),
        }
    }
}
