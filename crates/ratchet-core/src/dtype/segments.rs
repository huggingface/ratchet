use crate::{RVec, TensorBinding};

/// # Bindings
///
/// Quantized tensors are made up of segments.
/// The underlying buffer can be viewed as a blob of bytes, which cannot be read without interpretation.
///
/// The segments are the different chunks of the underlying bytes, which correspond to different
/// components of the quantized tensor.
///
/// E.g
///
/// pub struct BlockQ8_0 {
///     pub(crate) d: f16,
///     pub(crate) qs: [i8; QK8_0],
/// }
///
/// The above block is a GGUF block containing d, a scaling factor, and qs, which are the unscaled
/// tensor values.
///
/// Because of the padding and alignment requirements of WebGPU, we extract each of the components
/// of these blocks, and put them into separate segments.
///
/// | q q q q q q q q q q q q q q q q q q q q q q pad pad | d d d pad |
///
/// This is what the buffer may look like in memory. The segments give us the address of |.
pub trait Bindings {
    fn bindings(numel: usize) -> RVec<TensorBinding>;
}
