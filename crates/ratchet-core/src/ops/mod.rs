mod binary;
mod matmul;
mod softmax;

pub use binary::*;
pub use matmul::*;
pub use softmax::*;

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
