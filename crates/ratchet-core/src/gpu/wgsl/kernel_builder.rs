use crate::{Accessor, BindGroupLayoutDescriptor, Scalar, Tensor, Vec3, WorkgroupSize};

/// A builder for generating a kernel in WGSL.

#[derive(Debug)]
pub struct WgslFragment(String);

impl std::fmt::Display for WgslFragment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for WgslFragment {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl WgslFragment {
    pub fn new(capacity: usize) -> Self {
        Self(String::with_capacity(capacity))
    }

    pub fn write(&mut self, s: &str) {
        self.0.push_str(s);
    }

    pub fn write_fragment(&mut self, fragment: WgslFragment) {
        self.0.push_str(&fragment.0);
    }
}

pub trait RenderFragment {
    fn render(&self) -> WgslFragment;
}

pub struct WgslKernel(String);

impl std::fmt::Display for WgslKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub struct WgslKernelBuilder {
    pub indent: usize,
    pub kernel: String,
}

impl WgslKernelBuilder {
    pub fn new() -> Self {
        Self {
            indent: 0,
            kernel: String::new(),
        }
    }

    pub fn indent(&mut self) {
        self.indent += 1;
    }

    pub fn dedent(&mut self) {
        self.indent -= 1;
    }

    pub fn write_fragment(&mut self, fragment: WgslFragment) {
        self.kernel.push_str("\t".repeat(self.indent).as_str());
        self.kernel.push_str(&fragment.0);
    }

    pub fn bind_tensors(&mut self, tensors: &[&Tensor], bgld: &BindGroupLayoutDescriptor) {
        let mut fragment = WgslFragment::new(1024);
        let segments = tensors
            .iter()
            .flat_map(|t| t.bindings())
            .collect::<Vec<_>>();
        for (binding, s) in bgld.entries.iter().zip(segments.iter()) {
            let buffer_binding_type = match binding.ty {
                wgpu::BindingType::Buffer { ty, .. } => ty,
                _ => panic!("Unsupported binding type"),
            };
            matches!(buffer_binding_type, wgpu::BufferBindingType::Storage { .. });
            fragment.write(format!("@group(0) @binding({})\n", binding.binding).as_str());
            fragment.write_fragment(binding.render());
            fragment.write(" ");
            fragment.write(format!("X{}: ", binding.binding).as_str());
            fragment.write_fragment(s.render());
        }
        self.write_fragment(fragment);
    }

    pub fn render(self) -> WgslKernel {
        WgslKernel(self.kernel)
    }

    pub fn write_main(&mut self, workgroup_size: WorkgroupSize, builtins: &[BuiltIn]) {
        let mut fragment = WgslFragment::new(512);
        fragment.write(&format!("{}\n", workgroup_size));
        fragment.write("fn main(\n");
        for (b, builtin) in builtins.iter().enumerate() {
            let mut builtin = builtin.render();
            if b < builtins.len() - 1 {
                builtin.write(",\n");
            }
            fragment.write_fragment(builtin);
        }
        fragment.write(") {\n");
        self.write_fragment(fragment);
    }
}

/// WGSL built-in variables.
pub enum BuiltIn {
    LocalInvocationId,
    GlobalInvocationId,
    LocalInvocationIndex,
    WorkgroupId,
    NumWorkgroups,
    SubgroupId,
    SubgroupSize,
}

impl BuiltIn {
    /// Renders the built-in variable.
    pub fn render(&self) -> WgslFragment {
        let mut fragment = WgslFragment::new(128);
        fragment.write("@builtin(");
        let builtin = match self {
            BuiltIn::LocalInvocationId
            | BuiltIn::GlobalInvocationId
            | BuiltIn::WorkgroupId
            | BuiltIn::NumWorkgroups => {
                let var = self.render_var();
                format!("{var}) {var}: {}", Vec3::<u32>::render())
            }
            BuiltIn::LocalInvocationIndex | BuiltIn::SubgroupId | BuiltIn::SubgroupSize => {
                let var = self.render_var();
                format!("{var}) {var}: {}", Scalar::<u32>::render())
            }
        };
        fragment.write(builtin.as_str());
        fragment
    }

    /// Returns the variable name for the built-in.
    pub fn render_var(&self) -> &'static str {
        match self {
            BuiltIn::LocalInvocationId => "local_invocation_id",
            BuiltIn::GlobalInvocationId => "global_invocation_id",
            BuiltIn::LocalInvocationIndex => "local_invocation_index",
            BuiltIn::WorkgroupId => "workgroup_id",
            BuiltIn::NumWorkgroups => "num_workgroups",
            BuiltIn::SubgroupId => "subgroup_id",
            BuiltIn::SubgroupSize => "subgroup_size",
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    pub fn test_builtin_render() {
        use crate::BuiltIn;
        let local_id = BuiltIn::LocalInvocationId;
        let fragment = local_id.render();
        println!("{}", fragment);
        assert_eq!(
            fragment.0,
            "@builtin(local_invocation_id) local_invocation_id: vec3<u32>"
        );
    }
}
