use crate::{CPUBuffer, Storage, Tensor};
use bytemuck::NoUninit;

pub fn cpu_store_result<T: NoUninit>(dst: &Tensor, data: &[T]) {
    dst.update_storage(Storage::CPU(CPUBuffer::from_slice(data, dst.shape())));
}
