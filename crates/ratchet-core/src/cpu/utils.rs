use crate::{CPUBuffer, Shape, Storage, Strides, Tensor};
use bytemuck::NoUninit;

pub struct StridedIterator<'a> {
    shape: &'a Shape,
    strides: &'a Strides,
    next_index: Option<usize>,
    multi_index: Vec<usize>,
}

impl<'a> StridedIterator<'a> {
    pub fn new(shape: &'a Shape, strides: &'a Strides, start_offset: usize) -> Self {
        Self {
            shape,
            strides,
            next_index: if shape.numel() == 0 {
                None
            } else {
                Some(start_offset)
            },
            multi_index: vec![0; shape.len()],
        }
    }
}

impl<'a> Iterator for StridedIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let storage_index = match self.next_index {
            None => return None,
            Some(storage_index) => storage_index,
        };
        let mut updated = false;
        let mut next_storage_index = storage_index;
        for ((multi_i, max_i), stride_i) in self
            .multi_index
            .iter_mut()
            .zip(self.shape.iter())
            .zip(self.strides.iter())
            .rev()
        {
            let next_i = *multi_i + 1;
            if next_i < *max_i {
                *multi_i = next_i;
                updated = true;
                next_storage_index += *stride_i as usize;
                break;
            } else {
                next_storage_index -= *multi_i * *stride_i as usize;
                *multi_i = 0
            }
        }
        self.next_index = if updated {
            Some(next_storage_index)
        } else {
            None
        };
        Some(storage_index)
    }
}

impl<'a> From<(&'a Shape, &'a Strides)> for StridedIterator<'a> {
    fn from((shape, strides): (&'a Shape, &'a Strides)) -> Self {
        StridedIterator::new(shape, strides, 0)
    }
}

impl<'a> From<(&'a Shape, &'a Strides, usize)> for StridedIterator<'a> {
    fn from((shape, strides, offset): (&'a Shape, &'a Strides, usize)) -> Self {
        StridedIterator::new(shape, strides, offset)
    }
}

pub fn cpu_store_result<T: NoUninit>(dst: &Tensor, data: &[T]) {
    dst.update_storage(Storage::CPU(CPUBuffer::from_slice(data, dst.shape())));
}
