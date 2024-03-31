// Adapted from https://github.com/huggingface/candle/blob/5ebcfeaf0f5af69bb2f74385e8d6b020d4a3b8df/candle-core/src/quantized/mod.rs
//

use crate::error::Result;

use super::new_k_quants;
use super::new_k_quants::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ5_0, BlockQ5_1, BlockQ6K,
    BlockQ8K, BlockQ8_0, BlockQ8_1, GgmlType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

impl GgmlDType {
    pub(crate) fn from_u32(u: u32) -> Result<Self> {
        let dtype = match u {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            _ => crate::bail!("unknown dtype for tensor {u}"),
        };
        Ok(dtype)
    }

    pub(crate) fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
        }
    }

    //     // /// The block dtype
    //     // pub fn cpu_zeros(&self, elem_count: usize) -> Box<dyn QuantizedType> {
    //     //     match self {
    //     //         Self::F32 => Box::new(vec![f32::zeros(); elem_count]),
    //     //         Self::F16 => Box::new(vec![f16::zeros(); elem_count]),
    //     //         Self::Q4_0 => Box::new(vec![BlockQ4_0::zeros(); elem_count / BlockQ4_0::BLCK_SIZE]),
    //     //         Self::Q4_1 => Box::new(vec![BlockQ4_1::zeros(); elem_count / BlockQ4_1::BLCK_SIZE]),
    //     //         Self::Q5_0 => Box::new(vec![BlockQ5_0::zeros(); elem_count / BlockQ5_0::BLCK_SIZE]),
    //     //         Self::Q5_1 => Box::new(vec![BlockQ5_1::zeros(); elem_count / BlockQ5_1::BLCK_SIZE]),
    //     //         Self::Q8_0 => Box::new(vec![BlockQ8_0::zeros(); elem_count / BlockQ8_0::BLCK_SIZE]),
    //     //         Self::Q8_1 => Box::new(vec![BlockQ8_1::zeros(); elem_count / BlockQ8_1::BLCK_SIZE]),
    //     //         Self::Q2K => Box::new(vec![BlockQ2K::zeros(); elem_count / BlockQ2K::BLCK_SIZE]),
    //     //         Self::Q3K => Box::new(vec![BlockQ3K::zeros(); elem_count / BlockQ3K::BLCK_SIZE]),
    //     //         Self::Q4K => Box::new(vec![BlockQ4K::zeros(); elem_count / BlockQ4K::BLCK_SIZE]),
    //     //         Self::Q5K => Box::new(vec![BlockQ5K::zeros(); elem_count / BlockQ5K::BLCK_SIZE]),
    //     //         Self::Q6K => Box::new(vec![BlockQ6K::zeros(); elem_count / BlockQ6K::BLCK_SIZE]),
    //     //         Self::Q8K => Box::new(vec![BlockQ8K::zeros(); elem_count / BlockQ8K::BLCK_SIZE]),
    //     //     }
    //     // }
    /// The type size for blocks in bytes.
    pub fn type_size(&self) -> usize {
        use new_k_quants::*;
        match self {
            Self::F32 => 4,
            Self::F16 => 4, // 2, [TODO] Think about this. Currently WASM doesn't support F16
            Self::Q4_0 => BlockQ4_0::TYPE_SIZE,
            Self::Q4_1 => BlockQ4_1::TYPE_SIZE,
            Self::Q5_0 => BlockQ5_0::TYPE_SIZE,
            Self::Q5_1 => BlockQ5_1::TYPE_SIZE,
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L932
            Self::Q8_0 => BlockQ8_0::TYPE_SIZE,
            Self::Q8_1 => BlockQ8_1::TYPE_SIZE,
            Self::Q2K => BlockQ2K::TYPE_SIZE,
            Self::Q3K => BlockQ3K::TYPE_SIZE,
            Self::Q4K => BlockQ4K::TYPE_SIZE,
            Self::Q5K => BlockQ5K::TYPE_SIZE,
            Self::Q6K => BlockQ6K::TYPE_SIZE,
            Self::Q8K => BlockQ8K::TYPE_SIZE,
        }
    }

    /// The block size, i.e. the number of elements stored in each block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 => 1,
            Self::Q4_0 => new_k_quants::QK4_0,
            Self::Q4_1 => new_k_quants::QK4_1,
            Self::Q5_0 => new_k_quants::QK5_0,
            Self::Q5_1 => new_k_quants::QK5_1,
            Self::Q8_0 => new_k_quants::QK8_0,
            Self::Q8_1 => new_k_quants::QK8_1,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => {
                new_k_quants::QK_K
            }
        }
    }
}
