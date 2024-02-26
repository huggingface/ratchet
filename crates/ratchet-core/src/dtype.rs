use std::num::NonZeroU64;

use half::{bf16, f16};
use wgpu::{BufferAddress, BufferSize};

use crate::{gpu::MIN_STORAGE_BUFFER_SIZE, rvec, RVec};

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
    pub fn to_u32(self) -> u32 {
        match self {
            DType::F32 => 0,
            DType::F16 => 1,
            DType::WQ8 => 64,
            _ => unimplemented!(),
        }
    }

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

    //TODO: this is stupid
    // We take in the total_bytes of the buffer
    // but if the buffer is not completely filled with data
    // then the calculation of the segments is wrong
    pub fn segments(&self, total_bytes: usize) -> RVec<BufferSegment> {
        let total_bytes = if total_bytes < MIN_STORAGE_BUFFER_SIZE {
            MIN_STORAGE_BUFFER_SIZE
        } else {
            total_bytes
        };

        match self {
            DType::WQ8 => {
                println!("WQ8 Total Bytes: {}", total_bytes);
                let weights_size = total_bytes / 5 * 4;
                assert!(weights_size % 256 == 0); //storage buffer alignment
                let weights = BufferSegment::new(0, Some(weights_size as u64), true);
                println!("WQ8 Weights Segment: {:?}", weights);

                let absmax_size = total_bytes - weights_size;
                assert!(absmax_size % 256 == 0); //storage buffer alignment
                let absmax =
                    BufferSegment::new(weights_size as u64, Some(absmax_size as u64), true);
                println!("WQ8 Absmax Segment: {:?}", absmax);
                rvec![weights, absmax]
            }
            _ => {
                rvec![BufferSegment::new(0, Some(total_bytes as u64), false)]
            }
        }
    }
}

#[cfg(feature = "testing")]
impl DType {
    fn handle_type_str(ts: npyz::TypeStr) -> DType {
        match ts.endianness() {
            npyz::Endianness::Little => match (ts.type_char(), ts.size_field()) {
                (npyz::TypeChar::Float, 4) => DType::F32,
                (npyz::TypeChar::Int, 4) => DType::I32,
                (npyz::TypeChar::Uint, 4) => DType::U32,
                (t, s) => unimplemented!("{} {}", t, s),
            },
            _ => unimplemented!(),
        }
    }
}

#[cfg(feature = "testing")]
impl From<npyz::DType> for DType {
    fn from(dtype: npyz::DType) -> Self {
        match dtype {
            npyz::DType::Plain(ts) => Self::handle_type_str(ts),
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug)]
pub struct BufferSegment {
    pub offset: BufferAddress,
    pub size: Option<BufferSize>,
}

impl BufferSegment {
    pub fn new(offset: BufferAddress, size: Option<u64>, aligned: bool) -> Self {
        if let Some(size) = size {
            if aligned {
                assert!(size % 256 == 0); //storage buffer alignment
            }
        }
        let size = size.map(NonZeroU64::new).unwrap();
        Self { offset, size }
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
