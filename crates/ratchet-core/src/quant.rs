use num::integer::div_floor;
use num_traits::{AsPrimitive, Float};

use std::fmt::Debug;

#[derive(Debug, Clone, Copy)]
pub enum Quantization {
    None,
    SInt8,
    SInt4,
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

/// Quantize a matrix of floats to 8-bit signed integers.
/// The AsPrimitive<i32> may seem confusing, we be need to do the bit masking
/// using signed integers, then cast to unsigned, to avoid losing negative values
pub fn sint8_quantize<F: Float + AsPrimitive<i32> + Debug>(
    matrix: &[F],
    K: usize,
    N: usize,
) -> (Vec<u32>, Vec<F>) {
    assert!(matrix.len() == K * N);
    assert!(matrix.len() % 4 == 0);
    let pack_size = 4;
    let group_size = 16;
    let mut quantized_matrix = vec![0u32; K * N / pack_size];
    let mut absmax_matrix = vec![F::zero(); K * N / group_size];

    let sf = F::from(127.).unwrap();

    let mut local_absmax = F::neg_infinity();
    for i in (0..(K * N)).step_by(pack_size) {
        if i % group_size == 0 {
            local_absmax = matrix[i..i + group_size]
                .iter()
                .fold(F::neg_infinity(), |acc, &x| acc.max(x.abs()));
        }
        let packed_value: i32 = ((matrix[i] / local_absmax * sf).round().as_() & 0xFF)
            | (((matrix[i + 1] / local_absmax * sf).round().as_() & 0xFF) << 8)
            | (((matrix[i + 2] / local_absmax * sf).round().as_() & 0xFF) << 16)
            | (((matrix[i + 3] / local_absmax * sf).round().as_() & 0xFF) << 24);
        quantized_matrix[i / pack_size] = packed_value as u32;
        absmax_matrix[i / group_size] = local_absmax;
    }
    (quantized_matrix, absmax_matrix)
}

pub fn sint8_dequantize(
    quantized_matrix: &[u32],
    absmax_matrix: &[f32],
    K: usize,
    N: usize,
) -> Vec<f32> {
    let pack_size = 4;
    let group_size = 16;
    let mut matrix = vec![0.0; K * N];

    for i in (0..(K * N)).step_by(pack_size) {
        let local_absmax = absmax_matrix[div_floor(i, group_size)];
        let packed_value = quantized_matrix[div_floor(i, pack_size)] as i32;
        matrix[i] = ((packed_value << 24) >> 24) as f32 / 127.0 * local_absmax;
        matrix[i + 1] = ((packed_value << 16) >> 24) as f32 / 127.0 * local_absmax;
        matrix[i + 2] = ((packed_value << 8) >> 24) as f32 / 127.0 * local_absmax;
        matrix[i + 3] = (packed_value >> 24) as f32 / 127.0 * local_absmax;
    }

    matrix
}

#[cfg(test)]
mod tests {
    use rand::{distributions::Uniform, Rng};
    #[test]
    pub fn test_sint8_qdq() {
        let M = 32;
        let N = 32;
        let mut rng = rand::thread_rng();
        let range = Uniform::new(-0.2, 0.2);
        let matrix: Vec<f32> = (0..M * N).map(|_| rng.sample(range)).collect();
        println!("Original matrix: {:?}", matrix);

        let (quantized_matrix, absmax) = super::sint8_quantize(&matrix, M, N);
        println!("Absmax: {:?}", absmax);
        let dequantized_matrix = super::sint8_dequantize(&quantized_matrix, &absmax, M, N);
        println!("Dequantized matrix: {:?}", dequantized_matrix);
        for i in 0..matrix.len() {
            assert!((matrix[i] - dequantized_matrix[i]).abs() < 0.001);
        }
    }

    #[test]
    pub fn test_sint4_qdq() {
        let matrix = vec![
            0.1, -0.1, 0.6, -0.5, 1.0, -1.0, 1.2, -1.2, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 1.2, -1.2,
        ];
        println!("{:?}", matrix);
        let (quantized_matrix, absmax) = super::sint4_quantize(&matrix, 4, 4);
        assert_eq!(quantized_matrix.len(), 2);
        assert_eq!(quantized_matrix, vec![2544293105, 2544292849]);
        let dequantized_matrix = super::sint4_dequantize(&quantized_matrix, absmax, 4, 4);
        println!("{:?}", dequantized_matrix);
        for i in 0..matrix.len() {
            assert!((matrix[i] - dequantized_matrix[i]).abs() < 0.1);
        }
    }
}
