use crate::{rvec, Align, DType, RVec, TensorSegment};
use derive_new::new;

use super::segments::Segments;

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;

/// GGUF data types that are supported in Ratchet.
///
/// For actual blocks extracted from GGUF, see `ratchet-loader`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum GGUFDType {
    Q4K(Q4K),
    Q6K(Q6K),
    Q8_0(Q8_0),
}

impl GGUFDType {
    pub fn size_of(self) -> usize {
        match self {
            GGUFDType::Q8_0(_) => 36, //32 + 4
            _ => unimplemented!(),
        }
    }

    pub(crate) fn to_u32(self) -> u32 {
        match self {
            GGUFDType::Q8_0(_) => 8,
            GGUFDType::Q4K(_) => 12,
            GGUFDType::Q6K(_) => 14,
        }
    }

    pub fn segments(&self, numel: usize) -> RVec<TensorSegment> {
        match self {
            GGUFDType::Q4K(_) => Q4K::segments(numel),
            GGUFDType::Q6K(_) => Q6K::segments(numel),
            GGUFDType::Q8_0(_) => Q8_0::segments(numel),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default, new)]
pub struct Q4K;

impl Segments for Q4K {
    fn segments(numel: usize) -> RVec<TensorSegment> {
        let mut offset = 0;
        let ds_nbytes: u64 = (numel * 4).align() as u64;
        let ds_segment = TensorSegment::new(offset, ds_nbytes, DType::F16);

        let dmins_nbytes: u64 = (numel * 4).align() as u64;
        offset += ds_nbytes;
        let dmins_segment = TensorSegment::new(offset, dmins_nbytes, DType::F16);

        let scales_nbytes: u64 = (numel * K_SCALE_SIZE).align() as u64;
        offset += dmins_nbytes;
        let scales_segment = TensorSegment::new(offset, scales_nbytes, DType::U32);

        let qs_nbytes: u64 = (numel * QK_K / 2).align() as u64;
        offset += scales_nbytes;
        let qs_segment = TensorSegment::new(offset, qs_nbytes, DType::U32);

        rvec![ds_segment, dmins_segment, scales_segment, qs_segment]
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default, new)]
pub struct Q6K;

impl Segments for Q6K {
    fn segments(numel: usize) -> RVec<TensorSegment> {
        let mut offset = 0;
        let ql_nbytes: u64 = (numel * QK_K / 2).align() as u64;
        let ql_segment = TensorSegment::new(offset, ql_nbytes, DType::U32);

        let qh_nbytes: u64 = (numel * QK_K / 4).align() as u64;
        offset += ql_nbytes;
        let qh_segment = TensorSegment::new(offset, qh_nbytes, DType::U32);

        let scales_nbytes: u64 = (numel * QK_K / 16).align() as u64;
        offset += qh_nbytes;
        let scales_segment = TensorSegment::new(offset, scales_nbytes, DType::I32);

        let q_nbytes: u64 = (numel * 4).align() as u64;
        offset += scales_nbytes;
        //TODO: this should take in `compute_precision` beceause this can vary F32 or F16 now
        let q_segment = TensorSegment::new(offset, q_nbytes, DType::F32);

        rvec![ql_segment, qh_segment, scales_segment, q_segment]
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default, new)]
pub struct Q8_0;

impl Segments for Q8_0 {
    fn segments(numel: usize) -> RVec<TensorSegment> {
        let mut offset = 0;
        let qs_nbytes: u64 = numel.align() as u64;
        let qs_segment = TensorSegment::new(offset, qs_nbytes, DType::U32);

        let d_nbytes: u64 = ((numel / QK8_0) * 4).align() as u64;
        offset += qs_nbytes;
        let d_segment = TensorSegment::new(offset, d_nbytes, DType::F32);

        rvec![qs_segment, d_segment,]
    }
}
