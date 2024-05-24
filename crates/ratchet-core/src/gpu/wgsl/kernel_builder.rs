use inline_wgsl::wgsl;
use std::fmt::Write;

use crate::{
    BindingMode, BindingType, DeviceFeatures, KernelBinding, OpMetadata, RVec, Scalar, Vec3,
    WgslPrimitive, WorkgroupSize,
};

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

impl From<&str> for WgslFragment {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl WgslFragment {
    pub fn new(capacity: usize) -> Self {
        Self(String::with_capacity(capacity))
    }

    pub fn write(&mut self, s: impl AsRef<str>) {
        self.0.write_str(s.as_ref()).unwrap();
    }

    pub fn write_fragment(&mut self, fragment: WgslFragment) {
        self.write(&fragment.0);
    }
}

pub trait RenderFragment {
    fn render(&self) -> WgslFragment;
}

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct Ident(String);

impl std::fmt::Display for Ident {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for Ident {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

pub struct WgslKernelBuilder {
    pub bindings: RVec<KernelBinding>,
    pub workgroup_size: WorkgroupSize,
    pub builtins: Vec<BuiltIn>,
    pub globals: WgslFragment,
    pub main: WgslFragment,
    pub features: DeviceFeatures,
}

#[derive(thiserror::Error, Debug)]
pub enum KernelBuildError {
    #[error("Failed to build kernel: {0}")]
    BuildError(#[from] wgpu::naga::front::wgsl::ParseError),
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
            bindings: RVec::new(),
            workgroup_size: workgroup_size.clone(),
            builtins: builtins.clone(),
            globals,
            main: WgslFragment::new(2048),
            features,
        };
        builder.init_main(workgroup_size, &builtins);
        builder
    }

    pub fn build(mut self) -> Result<wgpu::naga::Module, KernelBuildError> {
        self.main.write("}\n");
        let mut source = self.globals;
        for binding in self.bindings.iter() {
            source.write(binding.render().0.as_str());
        }
        source.write(self.main.0.as_str());
        println!("{}", source);
        Ok(wgpu::naga::front::wgsl::parse_str(source.0.as_str())?)
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

    pub fn write_main(&mut self, fragment: impl Into<WgslFragment>) {
        self.main.write_fragment(fragment.into());
    }

    pub fn write_global(&mut self, fragment: impl Into<WgslFragment>) {
        self.globals.write_fragment(fragment.into());
    }

    pub fn write_metadata<M: OpMetadata>(&mut self) {
        self.write_global(M::render());
    }

    fn register_binding(
        &mut self,
        ty: BindingType,
        mode: BindingMode,
        name: impl Into<Ident>,
        accessor: impl ToString,
    ) {
        let group = !matches!(ty, BindingType::Storage) as usize;
        let binding = KernelBinding::new(
            name.into(),
            group,
            self.bindings.len(),
            ty,
            mode,
            accessor.to_string(),
        );
        self.bindings.push(binding);
    }

    pub(crate) fn register_storage(
        &mut self,
        name: impl Into<Ident>,
        mode: BindingMode,
        accessor: impl ToString,
    ) {
        self.register_binding(BindingType::Storage, mode, name, accessor);
    }

    pub(crate) fn register_uniform(&mut self, name: impl Into<Ident>, accessor: impl ToString) {
        self.register_binding(BindingType::Uniform, BindingMode::ReadOnly, name, accessor);
    }

    //    pub fn reduction<P: WgslPrimitive<T, N>, T: WgslDType + num_traits::Float, const N: usize>(
    //        &mut self,
    //        instruction: ReduceInstruction,
    //    ) {
    //        self.shared_memory::<P, T, N>("smem", self.workgroup_size.x as usize);
    //        self.constant("BLOCK_SIZE", Scalar::<u32>::new(self.workgroup_size.x));
    //
    //        match instruction.kind {
    //            ReduceKind::MAX => {
    //                self.workgroup_var::<Scalar<T>, _, 1>("maximum");
    //                self.constant("minFloat", Scalar::<T>::new(T::from(-65500).unwrap()))
    //            }
    //            ReduceKind::SUM => self.workgroup_var::<Scalar<T>, _, 1>("sum"),
    //        }
    //
    //        let initVar = match instruction.kind {
    //            ReduceKind::MAX => "minFloat",
    //            ReduceKind::SUM => "0.",
    //        };
    //
    //        let (reduce_func, reduce_fn_body) = match instruction.kind {
    //            ReduceKind::MAX => (
    //                "block_max",
    //                "smem[index] = max(smem[index], smem[index + stride]);",
    //            ),
    //            ReduceKind::SUM => ("block_sum", "smem[index] += smem[index + stride];"),
    //        };
    //
    //        self.write_globals(
    //            format!(
    //                r#"
    //fn {reduce_func}(index: u32, stride: u32) {{
    //    if index < stride {{
    //        {reduce_fn_body}
    //    }}
    //    workgroupBarrier();
    //}}
    //"#
    //            )
    //            .into(),
    //        );
    //
    //        let accessor = P::render_type();
    //
    //        let reduce_var = match N {
    //            1 => "metadata.N",
    //            2 => "metadata.ND2",
    //            4 => "metadata.ND4",
    //            _ => panic!("Invalid dimension"),
    //        };
    //
    //        let body = instruction.body;
    //        let mut smem_reduce: WgslFragment = format!(
    //            r#"
    //smem[index] = {accessor}({initVar});
    //for (var i: u32 = index; i < {reduce_var}; i += BLOCK_SIZE) {{
    //    {body}
    //}}
    //workgroupBarrier();
    //"#
    //        )
    //        .into();
    //
    //        let steps = (self.workgroup_size.x - 1).ilog2() as u32;
    //        for i in (0..=steps).rev().map(|x| 2u32.pow(x)) {
    //            smem_reduce.write(&format!(
    //                r#"
    //{reduce_func}(index, {i}u);"#,
    //            ));
    //        }
    //
    //        let finalize = match (N, instruction.kind) {
    //            (1, ReduceKind::MAX) => "maximum = smem[0];".into(),
    //            (1, ReduceKind::SUM) => "sum = smem[0]".into(),
    //            (2, ReduceKind::MAX) => "maximum = max(smem[0].x, smem[0].y);".into(),
    //            (2, ReduceKind::SUM) => format!("sum = dot(smem[0], {accessor}(1.0, 1.0));"),
    //            (4, ReduceKind::MAX) => {
    //                "maximum = max(smem[0].x, max(smem[0].y, max(smem[0].z, smem[0].w)));".into()
    //            }
    //            (4, ReduceKind::SUM) => format!("sum = dot(smem[0], {accessor}(1.0, 1.0, 1.0, 1.0));"),
    //            _ => panic!("Invalid reduction finalize"),
    //        };
    //
    //        smem_reduce.write(&format!(
    //            r#"
    //if index == 0 {{
    //    {finalize}
    //}}
    //workgroupBarrier();
    //"#
    //        ));
    //
    //        self.write_main(smem_reduce);
    //    }
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
        let var = self.render_var();
        let builtin_type = match self {
            BuiltIn::LocalInvocationId
            | BuiltIn::GlobalInvocationId
            | BuiltIn::WorkgroupId
            | BuiltIn::NumWorkgroups => Vec3::<u32>::render_type(),
            BuiltIn::LocalInvocationIndex | BuiltIn::SubgroupId | BuiltIn::SubgroupSize => {
                Scalar::<u32>::render_type()
            }
        };
        fragment.write(wgsl! { @builtin('var) 'var: 'builtin_type });
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
