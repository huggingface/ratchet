use num::integer::div_floor;
use num_traits::{AsPrimitive, Float};

use std::fmt::Debug;

use crate::{DType, Device, Tensor};

/// Quantizer
///
/// Packs weights into our custom quantization formats.
#[derive(Debug, derive_new::new)]
pub struct Quantizer {
    format: Quantization,
}

impl Quantizer {
    /// Quantizes a float 32 tensor into a packed uint32 tensor.
    /// This is the rust equivalent of: https://www.w3.org/TR/WGSL/#pack4x8snorm-builtin
    /// This allows us to call `unpack4x8snorm` in the shader.
    /// It's a pretty naive quantization scheme, more to come.
    pub fn sint8_quantize(&self, tensor: Tensor) -> Tensor {
        let numel = tensor.shape().numel();
        assert!(numel % 4 == 0 && numel % 16 == 0);
        assert!(tensor.dt() == DType::F32); //TODO: f16, bf16
                                            //TODO: check if tensor is contiguous

        let pack_size = self.format.pack_size();
        let group_size = self.format.group_size();

        let mut quantized_matrix = vec![0u32; numel / pack_size];
        let mut absmax_matrix = vec![0f32; numel / group_size];

        let sf = 127.0f32;
        let mut block_absmax = f32::NEG_INFINITY;

        let matrix = tensor.to_vec::<f32>().unwrap();

        for i in (0..numel).step_by(pack_size) {
            if i % group_size == 0 {
                block_absmax = matrix[i..i + group_size]
                    .iter()
                    .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x.abs()));
            }
            let packed_value: i32 = ((matrix[i] / block_absmax * sf).round() as i32 & 0xFF)
                | (((matrix[i + 1] / block_absmax * sf).round() as i32 & 0xFF) << 8)
                | (((matrix[i + 2] / block_absmax * sf).round() as i32 & 0xFF) << 16)
                | (((matrix[i + 3] / block_absmax * sf).round() as i32 & 0xFF) << 24);
            quantized_matrix[i / pack_size] = packed_value as u32;
            absmax_matrix[i / group_size] = block_absmax;
        }
        quantized_matrix.append(&mut unsafe { std::mem::transmute(absmax_matrix) });
        unsafe {
            Tensor::from_quantized(
                quantized_matrix,
                tensor.shape().clone(),
                DType::WQ8,
                Device::CPU,
            )
        }
    }

    pub fn sint8_dequantize(&self, quantized: Tensor) -> Tensor {
        assert!(quantized.dt() == DType::WQ8);
        let numel = quantized.shape().numel();
        let packed_numel = numel / self.format.pack_size() + numel / self.format.group_size();
        let pack_size = self.format.pack_size();
        let group_size = self.format.group_size();
        let quantized_matrix = quantized.to_vec::<u32>().unwrap();
        let mut dequantized = vec![0.0f32; numel];

        let absmax_start = packed_numel / group_size;

        for i in (0..packed_numel).step_by(pack_size) {
            let block_absmax = quantized_matrix[absmax_start + div_floor(i, group_size)] as f32;
            let packed_value = quantized_matrix[div_floor(i, pack_size)] as i32;
            dequantized[i] = ((packed_value << 24) >> 24) as f32 / 127.0 * block_absmax;
            dequantized[i + 1] = ((packed_value << 16) >> 24) as f32 / 127.0 * block_absmax;
            dequantized[i + 2] = ((packed_value << 8) >> 24) as f32 / 127.0 * block_absmax;
            dequantized[i + 3] = (packed_value >> 24) as f32 / 127.0 * block_absmax;
        }

        Tensor::from_data(dequantized, quantized.shape().clone(), Device::CPU)
    }

    pub fn sint4_quantize<F: Float + AsPrimitive<i32> + Debug>(
        matrix: &[F],
        K: usize,
        N: usize,
    ) -> (Vec<u32>, F) {
        assert!(matrix.len() == K * N);
        assert!(matrix.len() % 4 == 0);
        let pack_size = 8;
        let mut quantized_matrix = vec![0u32; K * N / pack_size];

        let absmax = matrix.iter().fold(F::zero(), |acc, &x| acc.max(x.abs()));
        let sf = F::from(7.).unwrap();

        for i in (0..(K * N)).step_by(pack_size) {
            let packed_value: i32 = ((matrix[i] / absmax * sf).round().as_() & 0xF)
                | (((matrix[i + 1] / absmax * sf).round().as_() & 0xF) << 4)
                | (((matrix[i + 2] / absmax * sf).round().as_() & 0xF) << 8)
                | (((matrix[i + 3] / absmax * sf).round().as_() & 0xF) << 12)
                | (((matrix[i + 4] / absmax * sf).round().as_() & 0xF) << 16)
                | (((matrix[i + 5] / absmax * sf).round().as_() & 0xF) << 20)
                | (((matrix[i + 6] / absmax * sf).round().as_() & 0xF) << 24)
                | (((matrix[i + 7] / absmax * sf).round().as_() & 0xF) << 28);
            quantized_matrix[i / pack_size] = packed_value as u32
        }
        (quantized_matrix, absmax)
    }

    pub fn sint4_dequantize(quantized_matrix: &[u32], absmax: f32, K: usize, N: usize) -> Vec<f32> {
        let pack_size = 8;
        let mut matrix = vec![0.0; K * N];

        for i in (0..(K * N)).step_by(pack_size) {
            let packed_value = quantized_matrix[div_floor(i, pack_size)] as i32;
            matrix[i] = ((packed_value << 28) >> 28) as f32 / 7.0 * absmax;
            matrix[i + 1] = ((packed_value << 24) >> 28) as f32 / 7.0 * absmax;
            matrix[i + 2] = ((packed_value << 20) >> 28) as f32 / 7.0 * absmax;
            matrix[i + 3] = ((packed_value << 16) >> 28) as f32 / 7.0 * absmax;
            matrix[i + 4] = ((packed_value << 12) >> 28) as f32 / 7.0 * absmax;
            matrix[i + 5] = ((packed_value << 8) >> 28) as f32 / 7.0 * absmax;
            matrix[i + 6] = ((packed_value << 4) >> 28) as f32 / 7.0 * absmax;
            matrix[i + 7] = (packed_value >> 28) as f32 / 7.0 * absmax;
        }

        matrix
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Quantization {
    None,
    SInt8,
    SInt4,
}

impl Quantization {
    pub fn pack_size(&self) -> usize {
        match self {
            Quantization::None => 1,
            Quantization::SInt8 => 4,
            Quantization::SInt4 => 8,
        }
    }

    pub fn group_size(&self) -> usize {
        match self {
            Quantization::None => 1,
            Quantization::SInt8 => 16,
            Quantization::SInt4 => 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{shape, Device, Quantization, Quantizer, Tensor};
    #[test]
    pub fn test_sint8_qdq() {
        let ground = Tensor::randn::<f32>(shape![64, 64], Device::CPU);
        let quantizer = Quantizer::new(Quantization::SInt8);
        let _quantized = quantizer.sint8_quantize(ground.deep_clone());
    }
}
