use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, DType, KernelElement, MetaOperation, OpGuards, OpMetadata, Operation,
    OperationError, RVec, StorageView, Strides, Tensor,
};

#[derive(new, Debug, Clone)]
pub struct Concat {
    inputs: RVec<Tensor>,
}

impl Operation for Concat {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        todo!()
    }
}

impl OpGuards for Concat {
    fn check_shapes(&self) {
        todo!()
    }

    fn check_dtypes(&self) {
        todo!()
    }
}

impl MetaOperation for Concat {
    fn srcs(&self) -> RVec<&Tensor> {
        self.inputs.iter().collect()
    }

    fn kernel_key(&self, dst: &Tensor) -> String {
        todo!()
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<WorkgroupCount, OperationError> {
        todo!()
    }

    fn storage_bind_group_layout(
        &self,
        _: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        todo!()
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        _: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        todo!()
    }
}

#[cfg(test)]
mod tests {}
