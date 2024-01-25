use half::{bf16, f16};

use crate::{rvec, RVec};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default, Hash)]
pub enum DType {
    Q8,
    F16,
    BF16,
    #[default]
    F32,
    I32,
    U32,
    WQ8, //Packed Q8 (|--4xQ8(u32)--| |--f32--|)
}

impl DType {
    /// Returns the size of the type in bytes.
    pub fn size_of(self) -> usize {
        match self {
            DType::Q8 => 1,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::F32 => 4,
            DType::I32 => 4,
            DType::U32 => 4,
            DType::WQ8 => 4, //Only works because they're both 4 bytes
        }
    }
}

pub trait TensorDType:
    Clone + std::fmt::Debug + PartialEq + 'static + num_traits::Zero + Send + Sync + bytemuck::Pod
{
    fn dt() -> DType;

    fn one() -> Self;
}

macro_rules! map_type {
    ($t:ty, $v:ident) => {
        impl TensorDType for $t {
            fn dt() -> DType {
                DType::$v
            }

            fn one() -> Self {
                1 as Self
            }
        }
    };
}

macro_rules! map_half_type {
    ($t:ty, $v:ident) => {
        impl TensorDType for $t {
            fn dt() -> DType {
                DType::$v
            }

            fn one() -> Self {
                Self::ONE
            }
        }
    };
}

map_type!(f32, F32);
map_type!(i32, I32);
map_type!(u32, U32);
map_half_type!(f16, F16);
map_half_type!(bf16, BF16);

/// 4xQ8(u32) + f32
pub struct WQ8Container {
    pub quantized: Vec<u32>,
    pub absmax: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockWQ8 {
    pub(crate) d: f32,
    pub(crate) qs: [u32; 4],
}
const _: () = assert!(std::mem::size_of::<BlockWQ8>() == 20);

/// # QContainer
///
/// Quantized weight container, used for packing and unpacking
pub trait QContainer {
    // Returns the size of each of the segments in **bytes**
    fn segment_sizes(total_bytes: usize) -> RVec<usize>;
    fn from_bytes(bytes: &[u8]) -> Self;
}

impl QContainer for WQ8Container {
    fn segment_sizes(total_bytes: usize) -> RVec<usize> {
        let num_quantized = (total_bytes / 5) * 4;
        let num_absmax = total_bytes / 5;
        rvec![num_quantized, num_absmax]
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let segment_sizes = Self::segment_sizes(bytes.len());
        WQ8Container {
            quantized: bytemuck::cast_slice::<u8, u32>(&bytes[..segment_sizes[0]]).to_vec(),
            absmax: bytemuck::cast_slice::<u8, f32>(&bytes[segment_sizes[0]..]).to_vec(),
        }
    }
}
