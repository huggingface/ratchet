mod binary;
mod conv;
mod matmul;
mod norm;
mod reindex;
mod softmax;
mod unary;

pub use binary::*;
pub use conv::*;
pub use matmul::*;
pub use norm::*;
pub use reindex::*;
pub use softmax::*;
pub use unary::*;

use crate::{Enforcer, Operation, Shape, StorageView, Strides, Tensor};

/// # KernelElement
///
/// Used to select the largest possible data type for a kernel.
/// If (dimension of interest % KE) == 0, it is safe to use.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum KernelElement {
    Vec4,
    Vec2,
    Scalar,
}

impl KernelElement {
    pub fn as_size(&self) -> usize {
        self.into()
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            KernelElement::Vec4 => "vec4",
            KernelElement::Vec2 => "vec2",
            KernelElement::Scalar => "scalar",
        }
    }
}

impl From<&KernelElement> for usize {
    fn from(item: &KernelElement) -> Self {
        match item {
            KernelElement::Vec4 => 4,
            KernelElement::Vec2 => 2,
            KernelElement::Scalar => 1,
        }
    }
}

#[derive(Debug, derive_new::new, Clone)]
pub struct View {
    input: Tensor,
    shape: Shape,
}

impl View {
    pub fn input(&self) -> &Tensor {
        &self.input
    }
}

impl Operation for View {
    fn check_invariants(srcs: &[&crate::Tensor]) -> Result<(), crate::OperationError> {
        Enforcer::check_input_arity(srcs, 1)?;
        Ok(())
    }

    fn infer_output(
        &self,
        srcs: &[&crate::Tensor],
    ) -> Result<crate::StorageView, crate::OperationError> {
        Enforcer::assert_equal_numel(&[srcs[0].shape(), &self.shape])?;
        //TODO: check if view is valid
        let strides = Strides::from(&self.shape);
        Ok(StorageView::new(self.shape.clone(), srcs[0].dt(), strides))
    }
}
