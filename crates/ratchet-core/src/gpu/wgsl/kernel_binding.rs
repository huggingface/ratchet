use crate::{Ident, RenderFragment};
use inline_wgsl::wgsl;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum BindingType {
    Storage,
    Uniform,
}

#[derive(Debug)]
pub(crate) enum BindingMode {
    ReadOnly,
    ReadWrite,
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

impl Into<wgpu::BindGroupLayoutEntry> for KernelBinding {
    fn into(self) -> wgpu::BindGroupLayoutEntry {
        let (binding_type, has_dynamic_offset) = match self.ty {
            BindingType::Storage => (
                wgpu::BufferBindingType::Storage {
                    read_only: matches!(self.mode, BindingMode::ReadOnly),
                },
                false,
            ),
            BindingType::Uniform => (wgpu::BufferBindingType::Uniform, true),
        };

        wgpu::BindGroupLayoutEntry {
            binding: self.binding as u32,
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
        let mode = match self.mode {
            BindingMode::ReadOnly => "read",
            BindingMode::ReadWrite => "read_write",
        };

        let KernelBinding {
            name,
            group,
            binding,
            accessor,
            ..
        } = self;

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

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::naga::front::wgsl::parse_str;

    #[test]
    fn test_kernel_binding_render() {
        let binding = KernelBinding::new(
            "input".into(),
            0,
            0,
            BindingType::Storage,
            BindingMode::ReadOnly,
            "array<vec4<f32>>".to_string(),
        );

        let rendered = binding.render();
        parse_str(&rendered.to_string()).unwrap();
    }
}
