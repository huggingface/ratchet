use crate::{
    KernelSource, OperationError, Tensor, WgslKernelBuilder, WgslPrimitive, WorkgroupSize,
};

pub trait Kernel {
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
