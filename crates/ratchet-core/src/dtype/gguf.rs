use derive_new::new;
use smallvec::SmallVec;

use crate::{BufferSegment, RVec, Segments};

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default, new)]
pub struct Q4K;

impl Segments for Q4K {
    fn segments(numel: usize) -> RVec<BufferSegment> {
        let ds_len: u64 = (numel * 4).align() as u64;
        let offset = 0;
        let ds_segment = BufferSegment::new(offset, Some(ds_len));

        let dmins_len: u64 = (numel * 4).align() as u64;
        let offset = offset + ds_len;
        let dmins_segment = BufferSegment::new(offset, Some(dmins_len));

        let scales_len: u64 = (numel * K_SCALE_SIZE).align() as u64;
        let offset = offset + dmins_len;
        let scales_segment = BufferSegment::new(offset, Some(scales_len));

        let qs_len: u64 = (numel * QK_K / 2).align() as u64;
        let offset = offset + scales_len;
        let qs_segment = BufferSegment::new(offset, Some(qs_len));

        SmallVec::<[BufferSegment; 4]>::from_vec(vec![
            ds_segment,
            dmins_segment,
            scales_segment,
            qs_segment,
        ])
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default, new)]
pub struct Q6K;

impl Segments for Q6K {
    fn segments(numel: usize) -> RVec<BufferSegment> {
        let ql_len: u64 = (numel * QK_K / 2).align() as u64;
        let offset = 0;
        let ql_segment = BufferSegment::new(offset, Some(ql_len));

        let qh_len: u64 = (numel * QK_K / 4).align() as u64;
        let offset = offset + ql_len;
        let qh_segment = BufferSegment::new(offset, Some(qh_len));

        let scales_len: u64 = (numel * QK_K / 16).align() as u64;
        let offset = offset + qh_len;
        let scales_segment = BufferSegment::new(offset, Some(scales_len));

        let q_len: u64 = (numel * 4).align() as u64;
        let offset = offset + scales_len;
        let q_segment = BufferSegment::new(offset, Some(q_len));

        SmallVec::<[BufferSegment; 4]>::from_vec(vec![
            ql_segment,
            qh_segment,
            scales_segment,
            q_segment,
        ])
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum GGUFDType {
    Q4K(Q4K),
    Q6K(Q6K),
}

pub trait Align {
    fn calculate_alignment(&self) -> usize;
    fn align(&self) -> usize;
}

impl Align for usize {
    fn calculate_alignment(&self) -> usize {
        let remainder = self % 256;
        if remainder == 0 {
            0
        } else {
            256 - remainder
        }
    }

    fn align(&self) -> usize {
        self + &self.calculate_alignment()
    }
}

impl GGUFDType {
    pub fn segments(&self, numel: usize) -> RVec<BufferSegment> {
        match self {
            GGUFDType::Q4K(_) => Q4K::segments(numel),
            GGUFDType::Q6K(_) => Q6K::segments(numel),
        }
    }
}
