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

impl RenderFragment for KernelBinding {
    fn render(&self) -> crate::WgslFragment {
        let ty = match self.ty {
            BindingType::Storage => "storage",
            BindingType::Uniform => "uniform",
        };
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

        wgsl! {
            @group('group) @binding('binding) var<'ty, 'mode> 'name: 'accessor;
        }
        .into()
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
