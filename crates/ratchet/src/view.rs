use crate::{DType, Shape, Strides};

/// How to view the underlying storage.
/// DType, Shape & Strides are all a matter of interpretation.
#[derive(Clone)]
pub struct View {
    dtype: DType,
    shape: Shape,
    strides: Strides,
    offset: usize,
}
