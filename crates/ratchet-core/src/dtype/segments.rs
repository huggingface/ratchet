use crate::{BufferSegment, RVec};

pub trait Segments {
    fn segments(numel: usize) -> RVec<BufferSegment>;
}
