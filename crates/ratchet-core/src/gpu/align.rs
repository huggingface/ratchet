//Handy trait for WebGPU buffer alignment
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

pub trait Padding {
    fn align_standard(&mut self) -> usize;
}

impl<T: Clone + Default> Padding for Vec<T> {
    fn align_standard(&mut self) -> usize {
        let length = &self.len();
        let alignment = length.calculate_alignment();
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
