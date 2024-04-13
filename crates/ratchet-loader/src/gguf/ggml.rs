// Adapted from https://github.com/huggingface/candle/blob/5ebcfeaf0f5af69bb2f74385e8d6b020d4a3b8df/candle-core/src/quantized/mod.rs
//

use ratchet::gguf::{self, GGUFSize, Q4K, Q6K, Q8_0};

use crate::error::Result;

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

impl From<GgmlDType> for ratchet::DType {
    fn from(val: GgmlDType) -> Self {
        match val {
            GgmlDType::F32 => ratchet::DType::F32,
            GgmlDType::F16 => ratchet::DType::F16,
            _ => unimplemented!(),
        }
    }
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

    /// The type size for blocks in bytes.
    pub fn type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2, // 2, [TODO] Think about this. Currently WASM doesn't support F16
            Self::Q4K => Q4K::TYPE_SIZE,
            Self::Q6K => Q6K::TYPE_SIZE,
            Self::Q8_0 => Q8_0::TYPE_SIZE,
            dt => todo!("{:?} not yet supported", dt),
        }
    }

    /// The block size, i.e. the number of elements stored in each block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 => 1,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => gguf::QK_K,
            dt => todo!("{:?} not yet supported", dt),
        }
    }
}
