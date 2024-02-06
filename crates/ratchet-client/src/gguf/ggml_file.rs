use crate::error::Result;
use ratchet::{Device, Tensor};

use super::{
    ggml::GgmlDType,
    k_quants::{self, GgmlType},
};

fn from_raw_data<T: GgmlType + Send + Sync + 'static>(
    raw_data: &[u8],
    size_in_bytes: usize,
    dims: Vec<usize>,
    device: &Device,
) -> Result<Tensor> {
    let raw_data_ptr = raw_data.as_ptr();
    let n_blocks = size_in_bytes / std::mem::size_of::<T>();
    let data = unsafe { std::slice::from_raw_parts(raw_data_ptr as *const T, n_blocks) };

    todo!("Not yet implemented")
}

/// Creates a [Tensor] from a raw GGML tensor.
pub fn qtensor_from_ggml(
    ggml_dtype: GgmlDType,
    raw_data: &[u8],
    dims: Vec<usize>,
    device: &Device,
) -> Result<Tensor> {
    let tensor_elems = dims.iter().product::<usize>();
    let block_size = ggml_dtype.block_size();
    if tensor_elems % block_size != 0 {
        crate::bail!(
            "the number of elements {tensor_elems} is not divisible by the block size {block_size}"
        )
    }
    let size_in_bytes = tensor_elems / block_size * ggml_dtype.type_size();

    match ggml_dtype {
        GgmlDType::F32 => from_raw_data::<f32>(raw_data, size_in_bytes, dims, device),
        GgmlDType::F16 => from_raw_data::<half::f16>(raw_data, size_in_bytes, dims, device),
        GgmlDType::Q4_0 => {
            from_raw_data::<k_quants::BlockQ4_0>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Q4_1 => {
            from_raw_data::<k_quants::BlockQ4_1>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Q5_0 => {
            from_raw_data::<k_quants::BlockQ5_0>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Q5_1 => {
            from_raw_data::<k_quants::BlockQ5_1>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Q8_0 => {
            from_raw_data::<k_quants::BlockQ8_0>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Q2K => {
            from_raw_data::<k_quants::BlockQ2K>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Q3K => {
            from_raw_data::<k_quants::BlockQ3K>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Q4K => {
            from_raw_data::<k_quants::BlockQ4K>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Q5K => {
            from_raw_data::<k_quants::BlockQ5K>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Q6K => {
            from_raw_data::<k_quants::BlockQ6K>(raw_data, size_in_bytes, dims, device)
        }
        _ => crate::bail!("quantized type {ggml_dtype:?} is not supported yet"),
    }
}
