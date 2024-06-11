use crate::{rvec, Align, BufferSegment, RVec};
use derive_new::new;

use super::segments::Bindings;

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
#[cfg(test)]
use test_strategy::Arbitrary;

#[cfg_attr(test, derive(Arbitrary))]
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

    pub fn bindings(&self, numel: usize) -> RVec<BufferSegment> {
        match self {
            GGUFDType::Q4K(_) => Q4K::bindings(numel),
            GGUFDType::Q6K(_) => Q6K::bindings(numel),
            GGUFDType::Q8_0(_) => Q8_0::bindings(numel),
        }
    }
}

impl std::fmt::Display for GGUFDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GGUFDType::Q4K(_) => write!(f, "Q4K"),
            GGUFDType::Q6K(_) => write!(f, "Q6K"),
            GGUFDType::Q8_0(_) => write!(f, "Q8_0"),
        }
    }
}

#[cfg_attr(test, derive(Arbitrary))]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default, new)]
pub struct Q4K;

impl Bindings for Q4K {
    fn bindings(numel: usize) -> RVec<BufferSegment> {
        let mut offset = 0;
        let ds_nbytes: u64 = (numel * 4).align() as u64;
        let ds_segment = BufferSegment::new(offset, ds_nbytes);

        let dmins_nbytes: u64 = (numel * 4).align() as u64;
        offset += ds_nbytes;
        let dmins_segment = BufferSegment::new(offset, dmins_nbytes);

        let scales_nbytes: u64 = (numel * K_SCALE_SIZE).align() as u64;
        offset += dmins_nbytes;
        let scales_segment = BufferSegment::new(offset, scales_nbytes);

        let qs_nbytes: u64 = (numel * QK_K / 2).align() as u64;
        offset += scales_nbytes;
        let qs_segment = BufferSegment::new(offset, qs_nbytes);

        rvec![ds_segment, dmins_segment, scales_segment, qs_segment]
    }
}

#[cfg_attr(test, derive(Arbitrary))]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default, new)]
pub struct Q6K;

impl Bindings for Q6K {
    fn bindings(numel: usize) -> RVec<BufferSegment> {
        let mut offset = 0;
        let ql_nbytes: u64 = (numel * QK_K / 2).align() as u64;
        let ql_segment = BufferSegment::new(offset, ql_nbytes);

        let qh_nbytes: u64 = (numel * QK_K / 4).align() as u64;
        offset += ql_nbytes;
        let qh_segment = BufferSegment::new(offset, qh_nbytes);

        let scales_nbytes: u64 = (numel * QK_K / 16).align() as u64;
        offset += qh_nbytes;
        let scales_segment = BufferSegment::new(offset, scales_nbytes);

        let q_nbytes: u64 = (numel * 4).align() as u64;
        offset += scales_nbytes;
        let q_segment = BufferSegment::new(offset, q_nbytes);

        rvec![ql_segment, qh_segment, scales_segment, q_segment]
    }
}

#[cfg_attr(test, derive(Arbitrary))]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default, new)]
pub struct Q8_0;

impl Bindings for Q8_0 {
    fn bindings(numel: usize) -> RVec<BufferSegment> {
        let mut offset = 0;
        let qs_nbytes: u64 = numel.align() as u64;
        let qs_segment = BufferSegment::new(offset, qs_nbytes);

        let d_nbytes: u64 = ((numel / QK8_0) * 4).align() as u64;
        offset += qs_nbytes;
        let d_segment = BufferSegment::new(offset, d_nbytes);

        rvec![qs_segment, d_segment,]
    }
}
