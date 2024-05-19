use half::f16;

/// Supported data types in WGSL.
///
/// This can be mapped to and from the Ratchet DType.

pub trait WgslDType {
    const DT: &'static str;
}

impl WgslDType for f32 {
    const DT: &'static str = "f32";
}

impl WgslDType for f16 {
    const DT: &'static str = "f16";
}

impl WgslDType for i32 {
    const DT: &'static str = "i32";
}

impl WgslDType for u32 {
    const DT: &'static str = "u32";
}
