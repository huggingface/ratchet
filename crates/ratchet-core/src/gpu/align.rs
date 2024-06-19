///WebGPU is very specific about buffer alignment.
///Since in Ratchet, any buffer may be copied back from GPU -> CPU, all buffers have a size
///that is a multiple of COPY_BUFFER_ALIGNMENT (4 bytes).
///
///However, WebGPU also has more stringent alignment for storage buffer offsets.
///This is controlled by `min_storage_buffer_offset_alignment` in wgpu::Limits.
///This defaults to 256
///
///For quantized data types in Ratchet, each "segment" of quantized block (mins, scales, qs, zero
///point etc.) is extracted and put into separate segments. Thus, these segments must be aligned to
///256.

///The `Align` trait provides methods to calculate the alignment of a usize, and to align a usize
pub trait Align {
    const STORAGE_BUFFER_OFFSET_ALIGNMENT: usize = 256;
    const COPY_BUFFER_ALIGNMENT: usize = 4;

    fn calculate_alignment(&self, alignment: usize) -> usize;
    fn align_for_copy(&self) -> usize;
    fn align_for_offset(&self) -> usize;
}

impl Align for usize {
    fn calculate_alignment(&self, alignment: usize) -> usize {
        let remainder = self % alignment;
        if remainder == 0 {
            0
        } else {
            alignment - remainder
        }
    }

    fn align_for_copy(&self) -> usize {
        self + &self.calculate_alignment(Self::COPY_BUFFER_ALIGNMENT)
    }

    fn align_for_offset(&self) -> usize {
        self + &self.calculate_alignment(Self::STORAGE_BUFFER_OFFSET_ALIGNMENT)
    }
}

pub trait Padding {
    //Pad the vector to the next multiple of 256.
    fn pad_to_offset(&mut self) -> usize;

    //Pad the vector to the next multiple of 4.
    fn pad_to_copy(&mut self) -> usize;
}

impl<T: Clone + Default> Padding for Vec<T> {
    fn pad_to_copy(&mut self) -> usize {
        let length = &self.len();
        let alignment = length.calculate_alignment(4);
        if alignment != 0 {
            let default_value: T = Default::default();
            let mut padding = vec![default_value; alignment];
            self.append(&mut padding);
            alignment
        } else {
            0
        }
    }

    fn pad_to_offset(&mut self) -> usize {
        let length = &self.len();
        let alignment = length.calculate_alignment(256);
        if alignment != 0 {
            let default_value: T = Default::default();
            let mut padding = vec![default_value; alignment];
            self.append(&mut padding);
            alignment
        } else {
            0
        }
    }
}
