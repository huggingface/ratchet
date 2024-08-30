use crate::{
    dtype::Quantized, gpu::STORAGE_BUFFER_ALIGN, DType, Device, Tensor, Q4_KF, Q4_KH, Q8_0F, Q8_0H,
};
use num::integer::div_floor;
use num_traits::{AsPrimitive, Float, FromPrimitive, Zero};

#[inline]
fn storage_align<T>(n: usize) -> usize {
    let size_t = core::mem::size_of::<T>();
    let nbytes = n * size_t;
    let aligned = if nbytes % STORAGE_BUFFER_ALIGN != 0 {
        nbytes + STORAGE_BUFFER_ALIGN - nbytes % STORAGE_BUFFER_ALIGN
    } else {
        nbytes
    };
    aligned / size_t
}

pub fn quantize_inner<Q: Quantized>(matrix: &[Q::FP], elements: usize) -> Vec<u32> {
    assert_eq!(elements % Q::PACK_SIZE, 0);
    assert_eq!(elements % Q::GROUP_SIZE, 0);

    let qmatrix_len = elements / Q::PACK_SIZE;
    let amatrix_len = elements / Q::GROUP_SIZE;

    let mut quantized_matrix = vec![0u32; storage_align::<u32>(qmatrix_len)];
    let mut d_matrix = vec![Q::FP::zero(); storage_align::<Q::FP>(amatrix_len)];
    let mut d = Q::FP::zero();

    for i in (0..elements).step_by(Q::PACK_SIZE) {
        if i % Q::GROUP_SIZE == 0 {
            d = matrix[i..i + Q::GROUP_SIZE]
                .iter()
                .fold(Q::FP::zero(), |acc, &x| acc.max(x.abs()))
                / Q::SF;
            d_matrix[i / Q::GROUP_SIZE] = d;
        }

        let mut packed_value: i32 = 0;
        for j in 0..Q::PACK_SIZE {
            packed_value |= ((matrix[i + j] / d).round().as_() & Q::MASK) << (j * Q::LSHIFT);
        }
        quantized_matrix[i / Q::PACK_SIZE] = packed_value as u32;
    }

    quantized_matrix.append(&mut unsafe { std::mem::transmute(d_matrix) });

    quantized_matrix
}

pub fn quantize<Q: Quantized>(tensor: &Tensor) -> Tensor {
    match (tensor.dt(), Q::dt()) {
        (DType::F32, DType::Q8_0F(_)) => {
            let matrix = tensor.to_vec::<Q::FP>().unwrap();
            unsafe {
                Tensor::from_quantized(
                    quantize_inner::<Q>(&matrix, tensor.shape().numel()),
                    DType::Q8_0F(Q8_0F::default()),
                    tensor.shape().clone(),
                    Device::CPU,
                )
            }
        }
        (DType::F32, DType::Q4_KF(_)) => {
            let matrix = tensor.to_vec::<Q::FP>().unwrap();
            unsafe {
                Tensor::from_quantized(
                    quantize_inner::<Q>(&matrix, tensor.shape().numel()),
                    DType::Q4_KF(Q4_KF::default()),
                    tensor.shape().clone(),
                    Device::CPU,
                )
            }
        }
        (DType::F16, DType::Q8_0H(_)) => {
            let matrix = tensor.to_vec::<Q::FP>().unwrap();
            unsafe {
                Tensor::from_quantized(
                    quantize_inner::<Q>(&matrix, tensor.shape().numel()),
                    DType::Q8_0H(Q8_0H::default()),
                    tensor.shape().clone(),
                    Device::CPU,
                )
            }
        }
        (DType::F16, DType::Q4_KH(_)) => {
            let matrix = tensor.to_vec::<Q::FP>().unwrap();
            unsafe {
                Tensor::from_quantized(
                    quantize_inner::<Q>(&matrix, tensor.shape().numel()),
                    DType::Q4_KH(Q4_KH::default()),
                    tensor.shape().clone(),
                    Device::CPU,
                )
            }
        }
        (dt, qdt) => panic!("Unsupported dtype combination {dt}, {qdt}"),
    }
}

pub fn dequantize_inner<Q: Quantized>(quantized: &[u8], elements: usize) -> Vec<Q::FP> {
    assert_eq!(elements % Q::PACK_SIZE, 0);
    assert_eq!(elements % Q::GROUP_SIZE, 0);

    let num_q = elements / Q::PACK_SIZE;
    let num_q_bytes = num_q * core::mem::size_of::<u32>();
    let aligned_q_bytes = storage_align::<u32>(num_q) * core::mem::size_of::<u32>();

    let num_absmax = elements / Q::GROUP_SIZE;
    let num_absmax_bytes = num_absmax * std::mem::size_of::<Q::FP>();
    let quantized_matrix = bytemuck::cast_slice::<u8, u32>(&quantized[..num_q_bytes]);
    let absmax_matrix = bytemuck::cast_slice::<u8, Q::FP>(
        &quantized[aligned_q_bytes..aligned_q_bytes + num_absmax_bytes],
    );

    let mut dequantized = vec![Q::FP::zero(); elements];
    for i in (0..elements).step_by(Q::PACK_SIZE) {
        let absmax = absmax_matrix[div_floor(i, Q::GROUP_SIZE)];
        let packed_value = quantized_matrix[div_floor(i, Q::PACK_SIZE)] as i32;
        for j in 0..Q::PACK_SIZE {
            dequantized[i + j] = Q::FP::from_i32(
                (packed_value << (Q::LSHIFT * (Q::PACK_SIZE - j - 1))) >> Q::RSHIFT,
            )
            .unwrap()
                * absmax;
        }
    }

    dequantized
}

pub fn dequantize(quantized: Tensor) -> Tensor {
    return match quantized.dt() {
        DType::Q8_0F(_) => {
            let elements = quantized.shape().numel();
            let original_shape = quantized.shape().clone();
            let raw_bytes = unsafe { quantized.into_bytes().unwrap() };
            let dequantized = dequantize_inner::<Q8_0F>(&raw_bytes, elements);
            Tensor::from_data(&dequantized, original_shape, Device::CPU)
        }
        DType::Q4_KF(_) => {
            let elements = quantized.shape().numel();
            let original_shape = quantized.shape().clone();
            let raw_bytes = unsafe { quantized.into_bytes().unwrap() };
            let dequantized = dequantize_inner::<Q4_KF>(&raw_bytes, elements);
            Tensor::from_data(&dequantized, original_shape, Device::CPU)
        }
        DType::Q8_0H(_) => {
            let elements = quantized.shape().numel();
            let original_shape = quantized.shape().clone();
            let raw_bytes = unsafe { quantized.into_bytes().unwrap() };
            let dequantized = dequantize_inner::<Q8_0H>(&raw_bytes, elements);
            Tensor::from_data(&dequantized, original_shape, Device::CPU)
        }
        DType::Q4_KH(_) => {
            let elements = quantized.shape().numel();
            let original_shape = quantized.shape().clone();
            let raw_bytes = unsafe { quantized.into_bytes().unwrap() };
            let dequantized = dequantize_inner::<Q4_KH>(&raw_bytes, elements);
            Tensor::from_data(&dequantized, original_shape, Device::CPU)
        }
        dt => panic!("Unsupported dtype {dt}"),
    };
}

#[cfg(test)]
mod tests {
    use crate::{
        dequantize, quantize, shape, Device, Quantized, Tensor, Q4_KF, Q4_KH, Q8_0F, Q8_0H,
    };
    use half::f16;

    // Verify that quantize -> dequantize is a (lossy) identity operation
    fn check_qd_reflexive<Q: Quantized>(atol: Q::FP, rtol: Q::FP)
    where
        Q::FP: std::fmt::Display + num_traits::Float + Default,
    {
        let ground = Tensor::randn::<Q::FP>(shape![64, 64], Device::CPU);
        let q = quantize::<Q>(&ground);
        let dq = dequantize(q);
        ground.all_close(&dq, atol, rtol).unwrap();
    }

    #[test]
    fn test_quantization_reflexivity() {
        check_qd_reflexive::<Q8_0F>(0.1, 0.1);
        check_qd_reflexive::<Q8_0H>(f16::from_f32(0.1), f16::from_f32(0.1));
        check_qd_reflexive::<Q4_KF>(0.3, 0.3);
        check_qd_reflexive::<Q4_KH>(f16::from_f32(0.3), f16::from_f32(0.3));
    }
}
