mod binary;
mod matmul;

pub use binary::*;
pub use matmul::*;

/// # KernelElement
///
/// Used to select the largest possible data type for a kernel.
/// If (dimension of interest % KE) == 0, it is safe to use.
pub enum KernelElement {
    Vec4,
    Vec2,
    Scalar,
}
