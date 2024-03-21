use crate::{gpu::*, Tensor, TensorId};
use rustc_hash::FxHashMap;
use std::sync::Arc;
use wgpu::{Adapter, Limits};

use crate::DeviceError;

pub const MAX_BUFFER_SIZE: u64 = (2 << 29) - 1;

/// # Device
///
/// A device is a handle to a physical GPU.
/// It is used to create resources and submit commands to the GPU.
///
/// Currently, WebGPU doesn't support multiple devices.
/// Ordinal should always be 0.
#[derive(Clone)]
pub struct WgpuDevice {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    ordinal: u32,
    buffer_allocator: Arc<BufferAllocator>,
    bind_group_pool: Arc<BindGroupPool>,
    bind_group_layout_pool: Arc<BindGroupLayoutPool>,
    pipeline_layout_pool: Arc<PipelineLayoutPool>,
    compute_pipeline_pool: Arc<ComputePipelinePool>,
}

impl std::ops::Deref for WgpuDevice {
    type Target = wgpu::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl std::fmt::Debug for WgpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "wgpu:{}", self.ordinal)
    }
}

impl PartialEq for WgpuDevice {
    fn eq(&self, other: &Self) -> bool {
        self.ordinal == other.ordinal && self.device.global_id() == other.device.global_id()
    }
}

impl WgpuDevice {
    pub async fn new() -> Result<Self, DeviceError> {
        #[cfg(target_arch = "wasm32")]
        let adapter = Self::select_adapter().await;
        #[cfg(not(target_arch = "wasm32"))]
        let adapter = Self::select_adapter()?;
        log::info!("Using adapter {:?}", adapter.get_info());

        #[allow(unused_mut)]
        let mut features = wgpu::Features::default();
        #[cfg(feature = "gpu-profiling")]
        {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        let mut device_descriptor = wgpu::DeviceDescriptor {
            label: Some("ratchet"),
            features,
            limits: Limits {
                max_buffer_size: MAX_BUFFER_SIZE,
                max_storage_buffer_binding_size: MAX_BUFFER_SIZE as u32,
                ..Default::default()
            },
        };
        let device_request = adapter.request_device(&device_descriptor, None).await;
        let (device, queue) = if let Err(e) = device_request {
            log::error!("Failed to acq. device, trying with reduced limits: {:?}", e);
            device_descriptor.limits = adapter.limits();
            adapter.request_device(&device_descriptor, None).await
        } else {
            device_request
        }?;

        let device = Arc::new(device);
        Ok(Self {
            queue: Arc::new(queue),
            ordinal: 0,
            buffer_allocator: Arc::new(BufferAllocator::new()),
            bind_group_pool: Arc::new(BindGroupPool::new()),
            bind_group_layout_pool: Arc::new(BindGroupLayoutPool::new()),
            pipeline_layout_pool: Arc::new(PipelineLayoutPool::new()),
            compute_pipeline_pool: Arc::new(ComputePipelinePool::new()),
            device,
        })
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    #[cfg(target_arch = "wasm32")]
    async fn select_adapter() -> Adapter {
        let instance = wgpu::Instance::default();
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .unwrap()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn select_adapter() -> Result<Adapter, DeviceError> {
        use wgpu::DeviceType;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            dx12_shader_compiler: wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default(),
            ..Default::default()
        });
        let backends = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY);
        let adapter = instance
            .enumerate_adapters(backends)
            .max_by_key(|adapter| match adapter.get_info().device_type {
                DeviceType::DiscreteGpu => 5,
                DeviceType::Other => 4,
                DeviceType::IntegratedGpu => 3,
                DeviceType::VirtualGpu => 2,
                DeviceType::Cpu => 1,
            })
            .ok_or(DeviceError::AdapterRequestFailed)?;
        Ok(adapter)
    }
}

impl WgpuDevice {
    pub fn get_or_create_buffer_init(
        &self,
        desc: &BufferDescriptor,
        contents: &[u8],
    ) -> Result<PooledGPUBuffer, DeviceError> {
        Ok(self
            .buffer_allocator
            .create_buffer_init(desc, contents, self))
    }

    pub fn create_uniform_init(&self, cpu_uniform: CpuUniform) -> PooledGPUBuffer {
        self.buffer_allocator.create_uniform_init(cpu_uniform, self)
    }

    pub fn get_or_create_buffer(
        &self,
        desc: &BufferDescriptor,
    ) -> Result<PooledGPUBuffer, DeviceError> {
        Ok(self.buffer_allocator.create_buffer(desc, self))
    }

    pub fn get_buffer(&self, handle: GpuBufferHandle) -> Result<PooledGPUBuffer, DeviceError> {
        Ok(self.buffer_allocator.get(handle))
    }

    pub fn get_or_create_bind_group(
        &self,
        desc: &BindGroupDescriptor,
    ) -> Result<GpuBindGroup, PoolError> {
        Ok(self.bind_group_pool.get_or_create(desc, self))
    }

    pub fn get_or_create_bind_group_layout(
        &self,
        desc: &BindGroupLayoutDescriptor,
    ) -> Result<BindGroupLayoutHandle, PoolError> {
        Ok(self.bind_group_layout_pool.get_or_create(desc, self))
    }

    pub fn get_or_create_pipeline_layout(
        &self,
        desc: &PipelineLayoutDescriptor,
    ) -> Result<PipelineLayoutHandle, PoolError> {
        Ok(self.pipeline_layout_pool.get_or_create(desc, self))
    }

    pub fn get_or_create_compute_pipeline(
        &self,
        desc: &ComputePipelineDescriptor,
    ) -> Result<ComputePipelineHandle, PoolError> {
        Ok(self.compute_pipeline_pool.get_or_create(desc, self))
    }

    pub fn bind_group_layout_resources(
        &self,
    ) -> StaticResourcePoolReadLockAccessor<'_, BindGroupLayoutHandle, wgpu::BindGroupLayout> {
        self.bind_group_layout_pool.resources()
    }

    pub fn pipeline_layout_resources(
        &self,
    ) -> StaticResourcePoolReadLockAccessor<'_, PipelineLayoutHandle, wgpu::PipelineLayout> {
        self.pipeline_layout_pool.resources()
    }

    pub fn pipeline_resources(
        &self,
    ) -> StaticResourcePoolReadLockAccessor<'_, ComputePipelineHandle, wgpu::ComputePipeline> {
        self.compute_pipeline_pool.resources()
    }

    /// Allocates all buffers required for storage of activations.
    /// Additionally, allocates buffer for leaf node, the tensor upon which resolve was called.
    pub fn allocate_cfg(
        &self,
        execution_order: &[&Tensor],
        device: &WgpuDevice,
    ) -> Result<FxHashMap<TensorId, PooledGPUBuffer>, DeviceError> {
        self.buffer_allocator.allocate_cfg(execution_order, device)
    }

    pub fn begin_pass(&self) {
        self.buffer_allocator.begin_pass(0);
    }
}
