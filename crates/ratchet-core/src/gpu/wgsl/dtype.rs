/// Supported data types in WGSL.
///
/// This can be mapped to and from the Ratchet DType.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgslDType {
    F32,
    F16,
    I32,
    U32,
}

impl std::fmt::Display for WgslDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WgslDType::F32 => write!(f, "f32"),
            WgslDType::F16 => write!(f, "f16"),
            WgslDType::I32 => write!(f, "i32"),
            WgslDType::U32 => write!(f, "u32"),
        }
    }
}

impl WgslDType {
    pub fn size(&self) -> usize {
        match self {
            WgslDType::F32 => 4,
            WgslDType::F16 => 2,
            WgslDType::I32 => 4,
            WgslDType::U32 => 4,
        }
    }
}

impl From<crate::DType> for WgslDType {
    fn from(dtype: crate::DType) -> Self {
        match dtype {
            crate::DType::F32 => WgslDType::F32,
            crate::DType::F16 => WgslDType::F16,
            crate::DType::I32 => WgslDType::I32,
            crate::DType::U32 => WgslDType::U32,
            _ => panic!("Attempted to convert unsupported DType to WGSL DType"),
        }
    }
}
