/// Ratchet memory layouts.
///
/// We closely follow the memory layout of the original GGUF implementation,
/// but often need 2 variants of each block type for devices that don't support f16.
use crate::{rvec, Align, BufferSegment, RVec};
use derive_new::new;
use half::f16;

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
pub trait Segments {
    fn segments(&self, numel: usize) -> RVec<BufferSegment>;
}

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;

#[cfg(test)]
use test_strategy::Arbitrary;

pub struct BlockQ8_0<T> {
    pub(crate) d: T,
    pub(crate) qs: [i8; QK8_0],
}

#[cfg_attr(test, derive(Arbitrary))]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default, new)]
pub struct Q8_0<T: std::fmt::Debug>(std::marker::PhantomData<T>);

//TODO: Segments could be derived using a macro
//Analyse the field structure of the block.
impl<T> Segments for Q8_0<T>
where
    T: std::fmt::Debug,
{
    fn segments(&self, numel: usize) -> RVec<BufferSegment> {
        let mut offset = 0;
        let qs_nbytes: u64 = numel.align_for_offset() as u64;
        let qs_segment = BufferSegment::new(offset, qs_nbytes);
        let d_nbytes: u64 = ((numel / QK8_0) * std::mem::size_of::<T>()).align_for_offset() as u64;
        offset += qs_nbytes;
        let d_segment = BufferSegment::new(offset, d_nbytes);
        rvec![qs_segment, d_segment]
    }
}

type BlockQ8_0F = BlockQ8_0<f32>;
type BlockQ8_0H = BlockQ8_0<f16>;

//We make these unit types for the sake of type safety.
#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub struct Q8_0F(Q8_0<f32>);

impl Segments for Q8_0F {
    fn segments(&self, numel: usize) -> RVec<BufferSegment> {
        self.0.segments(numel)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub struct Q8_0H(Q8_0<f16>);

impl Segments for Q8_0H {
    fn segments(&self, numel: usize) -> RVec<BufferSegment> {
        self.0.segments(numel)
    }
}
