use crate::{
    BindGroupLayoutDescriptor, DeviceFeatures, OpMetadata, RVec, Scalar, Tensor, Vec3, WgslArray,
    WgslPrimitive, WorkgroupSize,
};

use super::dtype::WgslDType;

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
    pub workgroup_size: WorkgroupSize,
    pub builtins: Vec<BuiltIn>,
    pub globals: WgslFragment,
    pub main: WgslFragment,
    pub features: DeviceFeatures,
}

impl WgslKernelBuilder {
    pub fn new(
        workgroup_size: WorkgroupSize,
        builtins: Vec<BuiltIn>,
        features: DeviceFeatures,
    ) -> Self {
        let mut globals = WgslFragment::new(2048);
        if features.SHADER_F16 {
            globals.write("enable f16;\n");
        }
        let mut builder = Self {
            workgroup_size: workgroup_size.clone(),
            builtins: builtins.clone(),
            globals,
            main: WgslFragment::new(2048),
            features,
        };
        builder.init_main(workgroup_size, &builtins);
        builder
    }

    pub fn render(mut self) -> WgslKernel {
        self.main.write("}\n");
        WgslKernel(format!("{}\n{}", self.globals, self.main))
    }

    fn init_main(&mut self, workgroup_size: WorkgroupSize, builtins: &[BuiltIn]) {
        self.main.write(&format!("{}\n", workgroup_size));
        self.main.write("fn main(\n");
        for (b, builtin) in builtins.iter().enumerate() {
            let mut builtin = builtin.render();
            if b < builtins.len() - 1 {
                builtin.write(",\n");
            }
            self.main.write_fragment(builtin);
        }
        self.main.write(") {\n");
    }

    pub fn write_main(&mut self, fragment: WgslFragment) {
        self.main.write_fragment(fragment);
    }

    fn write_globals(&mut self, fragment: WgslFragment) {
        self.globals.write_fragment(fragment);
    }

    pub fn write_metadata<M: OpMetadata>(&mut self) {
        self.write_globals(M::render_wgsl());
    }

    pub fn shared_memory<P: WgslPrimitive<T, N>, T: WgslDType, const N: usize>(
        &mut self,
        name: &str,
        size: usize,
    ) {
        let mut fragment = WgslFragment::new(64);
        fragment.write(&format!(
            "var<workgroup> {}: array<{}, {}>;\n",
            name,
            P::render_type(),
            size
        ));
        self.write_globals(fragment);
    }

    pub fn workgroup_var<P: WgslPrimitive<T, N>, T: WgslDType, const N: usize>(
        &mut self,
        name: &str,
    ) {
        let mut fragment = WgslFragment::new(64);
        fragment.write(&format!("var<workgroup> {}: {};\n", name, P::render_type()));
        self.write_globals(fragment);
    }

    //TODO: value shouldn't be &str
    pub fn constant<P: WgslPrimitive<T, N>, T: WgslDType, const N: usize>(
        &mut self,
        name: &str,
        value: P,
    ) {
        let mut fragment = WgslFragment::new(64);
        fragment.write(&format!(
            "const {}: {} = {};\n",
            name,
            P::render_type(),
            value
        ));
        self.write_globals(fragment);
    }

    pub fn write_bindings(
        &mut self,
        bindings: &BindGroupLayoutDescriptor,
        bind_vars: RVec<WgslFragment>,
    ) {
        let mut fragment = WgslFragment::new(512);
        for (binding, bind_var) in bindings.entries.iter().zip(bind_vars) {
            let buffer_binding_type = match binding.ty {
                wgpu::BindingType::Buffer { ty, .. } => ty,
                _ => panic!("Unsupported binding type"),
            };
            matches!(buffer_binding_type, wgpu::BufferBindingType::Storage { .. });
            fragment.write(format!("@group(0) @binding({})\n", binding.binding).as_str());
            fragment.write_fragment(binding.render());
            fragment.write_fragment(bind_var);
        }
        self.write_globals(fragment);
    }

    pub fn reduction<P: WgslPrimitive<T, N>, T: WgslDType + num_traits::Float, const N: usize>(
        &mut self,
        instruction: ReduceInstruction,
    ) {
        self.shared_memory::<P, T, N>("smem", self.workgroup_size.x as usize);
        self.constant("BLOCK_SIZE", Scalar::<u32>::new(self.workgroup_size.x));

        match instruction.kind {
            ReduceKind::MAX => {
                self.workgroup_var::<Scalar<T>, _, 1>("maximum");
                self.constant("minFloat", Scalar::<T>::new(T::from(-65500).unwrap()))
            }
            ReduceKind::SUM => self.workgroup_var::<Scalar<T>, _, 1>("sum"),
        }

        let initVar = match instruction.kind {
            ReduceKind::MAX => "minFloat",
            ReduceKind::SUM => "0.",
        };

        let (reduce_func, reduce_body) = match instruction.kind {
            ReduceKind::MAX => (
                "block_max",
                "smem[index] = max(smem[index], smem[index + stride]);",
            ),
            ReduceKind::SUM => ("block_sum", "smem[index] += smem[index + stride];"),
        };

        self.write_globals(
            format!(
                r#"
fn {reduce_func}(index: u32, stride: u32) {{
    if index < stride {{
        {reduce_body}
    }}
    workgroupBarrier();
}}
"#
            )
            .into(),
        );

        let accessor = P::render_type();

        let reduce_var = match N {
            1 => "metadata.N",
            2 => "metadata.ND2",
            4 => "metadata.ND4",
            _ => panic!("Invalid dimension"),
        };

        let indexing = format!(
            r#"
    let batch_stride = workgroup_id.y * metadata.M * {reduce_var}; 
    let row_start = batch_stride + workgroup_id.x * {reduce_var}; 
    let index = local_invocation_id.x;
    "#
        );
        self.write_main(indexing.into());

        let mut smem_reduce: WgslFragment = format!(
            r#"
smem[index] = {accessor}({initVar});
for (var i: u32 = index; i < {reduce_var}; i += BLOCK_SIZE) {{
    smem[index] = max(smem[index], X[row_start + i]); 
}}
workgroupBarrier();
"#
        )
        .into();

        let steps = (self.workgroup_size.x - 1).ilog2() as u32;
        for i in (0..=steps).rev().map(|x| 2u32.pow(x)) {
            smem_reduce.write(&format!(
                r#"
{reduce_func}(index, {i}u);"#,
            ));
        }

        let finalize = match (N, instruction.kind) {
            (1, ReduceKind::MAX) => "maximum = smem[0];".into(),
            (1, ReduceKind::SUM) => "sum = smem[0]".into(),
            (2, ReduceKind::MAX) => "maximum = max(smem[0].x, smem[0].y);".into(),
            (2, ReduceKind::SUM) => format!("sum = dot(smem[0], {accessor}(1.0, 1.0));"),
            (4, ReduceKind::MAX) => {
                "maximum = max(smem[0].x, max(smem[0].y, max(smem[0].z, smem[0].w)));".into()
            }
            (4, ReduceKind::SUM) => format!("sum = dot(smem[0], {accessor}(1.0, 1.0, 1.0, 1.0));"),
            _ => panic!("Invalid reduction finalize"),
        };

        smem_reduce.write(&format!(
            r#"
if index == 0 {{
    {finalize}
}}
workgroupBarrier();
"#
        ));

        self.write_main(smem_reduce);
    }
}

pub enum ReduceKind {
    MAX,
    SUM,
}

pub struct ReduceInstruction<'a> {
    pub input: &'a Tensor,
    pub kind: ReduceKind,
    pub axis: usize,
}

/// WGSL built-in variables.
#[derive(Debug, Clone)]
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
                format!("{var}) {var}: {}", Vec3::<u32>::render_type())
            }
            BuiltIn::LocalInvocationIndex | BuiltIn::SubgroupId | BuiltIn::SubgroupSize => {
                let var = self.render_var();
                format!("{var}) {var}: {}", Scalar::<u32>::render_type())
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
