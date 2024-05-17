use half::{bf16, f16};
use std::{cmp::max, num::NonZeroU64};
use wgpu::{BufferAddress, BufferSize};

use crate::{
    gpu::{dtype::WgslDType, MIN_STORAGE_BUFFER_SIZE},
    rvec, RVec, RenderFragment, WgslFragment,
};

pub mod gguf;
mod segments;

pub use segments::*;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default, Hash)]
pub enum DType {
    Q8,
    F16,
    BF16,
    #[default]
    F32,
    I32,
    U32,
    GGUF(gguf::GGUFDType),
}

impl DType {
    pub fn to_u32(self) -> u32 {
        match self {
            DType::F32 => 0,
            DType::F16 => 1,
            DType::GGUF(g) => g.to_u32(),
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
            DType::GGUF(g) => g.size_of(),
        }
    }

    pub fn is_quantized(self) -> bool {
        match self {
            DType::GGUF(_) => true,
            _ => false,
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct BufferSegment {
    pub offset: BufferAddress,
    pub size: BufferSize,
    pub dtype: DType,
}

impl BufferSegment {
    pub fn new(offset: BufferAddress, size: u64, dtype: DType) -> Self {
        Self {
            offset,
            size: NonZeroU64::new(size).expect("Invalid u64"),
            dtype,
        }
    }
}

impl RenderFragment for BufferSegment {
    fn render(&self) -> crate::WgslFragment {
        let mut fragment = WgslFragment::new(16);
        let wgsl_dt: WgslDType = self.dtype.into();
        fragment.write(&format!("array<{}>;\n", wgsl_dt));
        fragment
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

//Handy trait for WebGPU buffer alignment
pub trait Align {
    fn calculate_alignment(&self) -> usize;
    fn align(&self) -> usize;
}

impl Align for usize {
    fn calculate_alignment(&self) -> usize {
        let remainder = self % 256;
        if remainder == 0 {
            0
        } else {
            256 - remainder
        }
    }

    fn align(&self) -> usize {
        self + &self.calculate_alignment()
    }
}

pub trait Padding {
    fn align_standard(&mut self) -> usize;
}

impl<T: Clone + Default> Padding for Vec<T> {
    fn align_standard(&mut self) -> usize {
        let length = &self.len();
        let alignment = length.calculate_alignment();
        if alignment != 0 {
            let default_value: T = Default::default();
            let mut padding = vec![default_value; alignment];
            self.append(&mut padding);
            alignment
        } else {
            0
        }
    }
}
