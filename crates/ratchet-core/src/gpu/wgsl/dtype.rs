use half::f16;
use std::fmt::{Debug, Display};

/// Supported data types in WGSL.
///
/// This can be mapped to and from the Ratchet DType.
pub trait WgslDType: Debug + Display + Default + Copy {
    const DT: &'static str;
    const MIN: Self;

    fn render(&self) -> String;
}
//RENDER IS CONFUSING HERE

impl WgslDType for f32 {
    const DT: &'static str = "f32";
    const MIN: Self = -3e10; //ranges for wgsl and rust are diff

    fn render(&self) -> String {
        format!("{}f", self)
    }
}

impl WgslDType for f16 {
    const DT: &'static str = "f16";
    const MIN: Self = f16::MIN;

    fn render(&self) -> String {
        format!("{}h", self)
    }
}

impl WgslDType for i32 {
    const DT: &'static str = "i32";
    const MIN: Self = i32::MIN;

    fn render(&self) -> String {
        format!("{}i", self)
    }
}

impl WgslDType for u32 {
    const DT: &'static str = "u32";
    const MIN: Self = u32::MIN;

    fn render(&self) -> String {
        format!("{}u", self)
    }
}
