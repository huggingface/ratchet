use crate::{CPUBuffer, OperationError, Storage, Tensor, TensorDType};
use bytemuck::NoUninit;

#[inline]
fn apply_fn_helper<T: TensorDType, U: TensorDType>(src: &[T], dst: &mut [U], f: fn(T) -> U) {
    assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().copied().zip(dst.iter_mut()) {
        *d = f(s);
    }
}

#[inline]
pub fn apply_fn<T: TensorDType, U: TensorDType>(
    input: &Tensor,
    dst: &Tensor,
    f: fn(T) -> U,
) -> Result<(), OperationError> {
    let input = input.to_vec::<T>()?;
    let mut result = vec![U::zero(); dst.shape().numel()];
    apply_fn_helper(&input, &mut result, f);
    cpu_store_result(dst, &result);
    Ok(())
}

pub fn cpu_store_result<T: NoUninit>(dst: &Tensor, data: &[T]) {
    dst.update_storage(Storage::CPU(CPUBuffer::from_slice(data, dst.shape())));
}
