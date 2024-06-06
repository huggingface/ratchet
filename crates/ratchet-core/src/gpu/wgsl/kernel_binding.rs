use crate::{Ident, RenderFragment};
use inline_wgsl::wgsl;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum BindingType {
    Storage,
    Uniform,
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum BindingMode {
    ReadOnly,
    ReadWrite,
}

impl BindingMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            BindingMode::ReadOnly => "read",
            BindingMode::ReadWrite => "read_write",
        }
    }
}

#[derive(Debug, derive_new::new)]
pub struct KernelBinding {
    name: Ident,
    group: usize,
    binding: usize,
    ty: BindingType,
    mode: BindingMode,
    accessor: String,
}

impl From<KernelBinding> for wgpu::BindGroupLayoutEntry {
    fn from(val: KernelBinding) -> Self {
        let (binding_type, has_dynamic_offset) = match val.ty {
            BindingType::Storage => (
                wgpu::BufferBindingType::Storage {
                    read_only: matches!(val.mode, BindingMode::ReadOnly),
                },
                false,
            ),
            BindingType::Uniform => (wgpu::BufferBindingType::Uniform, true),
        };

        wgpu::BindGroupLayoutEntry {
            binding: val.binding as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: binding_type,
                min_binding_size: None,
                has_dynamic_offset,
            },
            count: None,
        }
    }
}

impl RenderFragment for KernelBinding {
    fn render(&self) -> crate::WgslFragment {
        let KernelBinding {
            name,
            group,
            binding,
            accessor,
            ..
        } = self;
        let mode = self.mode.as_str();

        let result = match self.ty {
            BindingType::Storage => wgsl! {
                @group('group) @binding('binding) var<storage, 'mode> 'name: 'accessor;
            },
            BindingType::Uniform => wgsl! {
                @group('group) @binding('binding) var<uniform> 'name: 'accessor;
            },
        };
        result.into()
    }
}
