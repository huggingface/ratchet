use half::f16;

/// Supported data types in WGSL.
///
/// This can be mapped to and from the Ratchet DType.
pub trait WgslDType: std::fmt::Display + Default + Copy {
    const DT: &'static str;
    const NEG_INF: Self;

    fn render(&self) -> String;
}
//RENDER IS CONFUSING HERE

impl WgslDType for f32 {
    const DT: &'static str = "f32";
    const NEG_INF: Self = -3.402_823e38;

    fn render(&self) -> String {
        format!("{}f", self)
    }
}

impl WgslDType for f16 {
    const DT: &'static str = "f16";
    const NEG_INF: Self = f16::NEG_INFINITY;

    fn render(&self) -> String {
        format!("{}h", self)
    }
}

impl WgslDType for i32 {
    const DT: &'static str = "i32";
    const NEG_INF: Self = i32::MIN;

    fn render(&self) -> String {
        format!("{}i", self)
    }
}

impl WgslDType for u32 {
    const DT: &'static str = "u32";
    const NEG_INF: Self = u32::MIN;

    fn render(&self) -> String {
        format!("{}u", self)
    }
}
