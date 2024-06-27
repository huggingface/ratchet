use crate::{
    CpuUniform, KernelElement, KernelKey, KernelSource, OperationError, Tensor, WgslKernelBuilder,
    WgslPrimitive, WorkgroupSize, Workload,
};

pub trait Kernel {
    /// # Calculate Dispatch
    ///
    /// Determine required amount of workgroups to execute the operation.
    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError>;

    /// Kernel Name
    fn kernel_name(&self) -> String;

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

    /// # Metadata
    ///
    /// Each kernel has zero or more required metadata fields (e.g shape, strides, etc).
    /// This is stored in a uniform buffer, for faster access.
    ///
    /// The metadata is limited to 256 bytes per kernel.
    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        dst: &Tensor,
        kernel_element: &KernelElement,
    ) -> Result<u64, OperationError>;

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError>;
}

pub trait KernelRender {
    ///Every WGSL must have 1 or more bindings registered
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError>;

    ///Generate the kernel source
    fn build<P: WgslPrimitive>(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError>;
}
