use std::borrow::Cow;

use crate::{gpu::WgpuDevice, KernelElement, KERNELS};

use super::{
    PipelineLayoutHandle, StaticResourcePool, StaticResourcePoolAccessor,
    StaticResourcePoolReadLockAccessor,
};

slotmap::new_key_type! { pub struct ComputePipelineHandle; }

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ComputePipelineDescriptor {
    pub pipeline_layout: PipelineLayoutHandle,
    pub kernel_name: &'static str, //string uniquely identifying the kernel
    pub kernel_element: KernelElement,
    //aux_ctx: Option<RVec<(&'static str, u32)>>, Used for sizing SMEM
}

impl ComputePipelineDescriptor {
    pub fn build_kernel_key(&self) -> String {
        format!("{}_{}", self.kernel_name, self.kernel_element.as_str())
    }
}

pub struct ComputePipelinePool {
    inner:
        StaticResourcePool<ComputePipelineHandle, ComputePipelineDescriptor, wgpu::ComputePipeline>,
}

impl ComputePipelinePool {
    pub fn new() -> Self {
        Self {
            inner: StaticResourcePool::default(),
        }
    }

    pub fn get_or_create(
        &self,
        desc: &ComputePipelineDescriptor,
        device: &WgpuDevice,
    ) -> ComputePipelineHandle {
        self.inner.get_or_create(desc, |desc| {
            let kernel_key = desc.build_kernel_key();
            let shader = KERNELS.get(kernel_key.as_str()).unwrap();
            let label = Some(kernel_key.as_str());

            let shader_module_desc = wgpu::ShaderModuleDescriptor {
                label,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader)),
            };

            //We don't cache shader modules because pipelines are cached
            let module = if std::env::var("RATCHET_CHECKED").is_ok() {
                log::warn!("Using checked shader compilation");
                device.create_shader_module(shader_module_desc)
            } else {
                unsafe { device.create_shader_module_unchecked(shader_module_desc) }
            };

            let pipeline_layouts = device.pipeline_layout_resources();
            let pipeline_layout = pipeline_layouts.get(desc.pipeline_layout).unwrap();

            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label,
                layout: Some(pipeline_layout),
                module: &module,
                entry_point: "main",
            })
        })
    }

    /// Locks the resource pool for resolving handles.
    ///
    /// While it is locked, no new resources can be added.
    pub fn resources(
        &self,
    ) -> StaticResourcePoolReadLockAccessor<'_, ComputePipelineHandle, wgpu::ComputePipeline> {
        self.inner.resources()
    }
}
