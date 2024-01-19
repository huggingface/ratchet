use std::borrow::Cow;

use crate::gpu::WgpuDevice;

use super::{
    PipelineLayoutHandle, StaticResourcePool, StaticResourcePoolAccessor,
    StaticResourcePoolReadLockAccessor,
};

slotmap::new_key_type! { pub struct ComputePipelineHandle; }

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum KernelElement {
    Vec4,
    Vec2,
    Scalar,
}

impl KernelElement {
    pub fn as_size(&self) -> usize {
        self.into()
    }
}

impl From<&KernelElement> for usize {
    fn from(item: &KernelElement) -> Self {
        match item {
            KernelElement::Vec4 => 4,
            KernelElement::Vec2 => 2,
            KernelElement::Scalar => 1,
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct ComputePipelineDescriptor {
    pub pipeline_layout: PipelineLayoutHandle,
    pub kernel_key: &'static str, //string uniquely identifying the kernel
    pub elem: KernelElement,
    //aux_ctx: Option<RVec<(&'static str, u32)>>, Used for sizing SMEM
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
            //TODO: shader shouldn't be hardcoded
            let shader = include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/kernels",
                "/add_scalar.wgsl"
            ));
            let label = Some(desc.kernel_key);
            let module = if std::env::var("RATCHET_CHECKED").is_ok() {
                log::warn!("Using checked shader compilation");
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label,
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader)),
                })
            } else {
                unsafe {
                    device.create_shader_module_unchecked(wgpu::ShaderModuleDescriptor {
                        label,
                        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader)),
                    })
                }
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
