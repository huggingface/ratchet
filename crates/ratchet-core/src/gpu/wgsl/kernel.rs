use rustc_hash::FxHashMap;

use crate::{
    CpuUniform, DynamicKernelMetadata, KernelElement, KernelKey, KernelMetadata, KernelSource,
    OperationError, Tensor, WgslFragment, WgslKernelBuilder, WgslPrimitive, WorkgroupSize,
    Workload,
};

#[derive(Debug)]
pub enum DynMetaField {
    Vec4U32(String),
    U32(String),
}

#[derive(Debug)]
pub struct DynMetadata {
    fields: FxHashMap<String, DynMetaField>,
}

impl DynamicKernelMetadata for DynMetadata {
    fn render(&self) -> crate::WgslFragment {
        let mut fragment = WgslFragment::new(512);
        fragment.push(r#"struct Meta {"#);
        for (name, field) in self.fields.iter() {
            fragment.push(format!("{}: {}", name, field.render()));
        }
        fragment.push("}\n");
        fragment
    }

    fn write(&self, uniform: &mut CpuUniform) -> Result<u64, OperationError> {}
}

pub trait Kernel {
    /// # Metadata
    ///
    /// Each kernel has zero or more required metadata fields (e.g shape, strides, etc).
    /// This is stored in a uniform buffer, for faster access.
    ///
    /// The metadata is limited to 256 bytes per kernel.
    ///
    /// There are 2 methods for the metadata:
    /// 1. render() - renders the actual WGSL struct definition for the struct
    /// 2. write_metadata() - writes the actual metadata values to the uniform buffer
    ///
    /// There are 2 flavours of metadata:
    /// 1. Static Metadata - the structure is known at compile time, so we get `render` for free
    ///    with the derive macro: `WgslMetadata`. The author still needs to implement
    ///    `write_metadata`.
    /// 2. Dynamic Metadata - the structure is not known at compile time, so the author must
    ///   implement both `render` and `write_metadata`.
    type Metadata: KernelMetadata + 'static;

    fn metadata(
        &self,
        dst: &Tensor,
        kernel_element: &KernelElement,
    ) -> Result<Self::Metadata, OperationError> {
        todo!()
    }

    /// # Calculate Dispatch
    ///
    /// Determine required amount of workgroups to execute the operation.
    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError>;

    /// # Kernel Key
    ///
    /// Construct a unique cache key for a kernel.
    /// If the key is registered in the compute module pool, the module is reused.
    ///
    /// Default implementation is provided, but care must be taken to ensure that the key is
    /// unique via the `additional` parameter.
    fn kernel_key(
        &self,
        workgroup_size: &WorkgroupSize,
        inplace: bool,
        dst: &Tensor,
        kernel_element: &KernelElement,
    ) -> KernelKey {
        KernelKey::new(
            &self.kernel_name(),
            &self.srcs(),
            dst,
            workgroup_size,
            inplace,
            kernel_element,
            None,
        )
    }

    /// # Kernel Element
    ///
    /// Determine the largest possible unit data type that can be used (e.g f32, vec2<f32>, vec4<f32>)
    fn kernel_element(&self, dst: &Tensor) -> KernelElement;

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError>;
}

/// This trait is focused on the generation of the kernel source code.
pub trait KernelRenderable {
    ///Every WGSL must have 1 or more bindings registered
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError>;

    ///Generate the kernel source
    fn render<P: WgslPrimitive>(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError>;
}
