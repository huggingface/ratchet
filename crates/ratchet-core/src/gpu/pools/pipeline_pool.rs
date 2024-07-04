use crate::{gpu::WgpuDevice, KernelKey, KernelModuleHandle};

use super::{
    PipelineLayoutHandle, StaticResourcePool, StaticResourcePoolAccessor,
    StaticResourcePoolReadLockAccessor,
};

slotmap::new_key_type! { pub struct ComputePipelineHandle; }

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ComputePipelineDescriptor {
    pub pipeline_layout: PipelineLayoutHandle,
    pub kernel_key: KernelKey,
    pub kernel_module: KernelModuleHandle,
}

pub struct ComputePipelinePool {
    inner:
        StaticResourcePool<ComputePipelineHandle, ComputePipelineDescriptor, wgpu::ComputePipeline>,
}

impl Default for ComputePipelinePool {
    fn default() -> Self {
        Self::new()
    }
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
            let label = Some(desc.kernel_key.as_str());
            //println!("LABEL: {:?}", label);
            let kernel_resources = device.kernel_module_resources();

            let module = kernel_resources.get(desc.kernel_module).unwrap();

            let pipeline_layouts = device.pipeline_layout_resources();
            let pipeline_layout = pipeline_layouts.get(desc.pipeline_layout).unwrap();

            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label,
                layout: Some(pipeline_layout),
                module,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions {
                    zero_initialize_workgroup_memory: false,
                    ..Default::default()
                },
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
