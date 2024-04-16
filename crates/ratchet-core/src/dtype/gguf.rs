use crate::{rvec, Align, BufferSegment, RVec};
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

    pub fn segments(&self, numel: usize) -> RVec<BufferSegment> {
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
    fn segments(numel: usize) -> RVec<BufferSegment> {
        let mut offset = 0;
        let ds_len: u64 = (numel * 4).align() as u64;
        let ds_segment = BufferSegment::new(offset, ds_len);

        let dmins_len: u64 = (numel * 4).align() as u64;
        offset += ds_len;
        let dmins_segment = BufferSegment::new(offset, dmins_len);

        let scales_len: u64 = (numel * K_SCALE_SIZE).align() as u64;
        offset += dmins_len;
        let scales_segment = BufferSegment::new(offset, scales_len);

        let qs_len: u64 = (numel * QK_K / 2).align() as u64;
        offset += scales_len;
        let qs_segment = BufferSegment::new(offset, qs_len);

        rvec![ds_segment, dmins_segment, scales_segment, qs_segment]
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default, new)]
pub struct Q6K;

impl Segments for Q6K {
    fn segments(numel: usize) -> RVec<BufferSegment> {
        let mut offset = 0;
        let ql_len: u64 = (numel * QK_K / 2).align() as u64;
        let ql_segment = BufferSegment::new(offset, ql_len);

        let qh_len: u64 = (numel * QK_K / 4).align() as u64;
        offset += ql_len;
        let qh_segment = BufferSegment::new(offset, qh_len);

        let scales_len: u64 = (numel * QK_K / 16).align() as u64;
        offset += qh_len;
        let scales_segment = BufferSegment::new(offset, scales_len);

        let q_len: u64 = (numel * 4).align() as u64;
        offset += scales_len;
        let q_segment = BufferSegment::new(offset, q_len);

        rvec![ql_segment, qh_segment, scales_segment, q_segment,]
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default, new)]
pub struct Q8_0;

impl Segments for Q8_0 {
    fn segments(numel: usize) -> RVec<BufferSegment> {
        let mut offset = 0;
        let qs_len: u64 = numel.align() as u64;
        let qs_segment = BufferSegment::new(offset, qs_len);

        let d_len: u64 = ((numel / QK8_0) * 4).align() as u64;
        offset += qs_len;
        let d_segment = BufferSegment::new(offset, d_len);

        rvec![qs_segment, d_segment,]
    }
}
