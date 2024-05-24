use half::f16;

/// Supported data types in WGSL.
///
/// This can be mapped to and from the Ratchet DType.
pub trait WgslDType: std::fmt::Display {
    const DT: &'static str;

    fn render(&self) -> String;
}
//RENDER IS CONFUSING HERE

impl WgslDType for f32 {
    const DT: &'static str = "f32";

    fn render(&self) -> String {
        format!("{}f", self)
    }
}

impl WgslDType for f16 {
    const DT: &'static str = "f16";

    fn render(&self) -> String {
        format!("{}h", self)
    }
}

impl WgslDType for i32 {
    const DT: &'static str = "i32";

    fn render(&self) -> String {
        format!("{}i", self)
    }
}

impl WgslDType for u32 {
    const DT: &'static str = "u32";

    fn render(&self) -> String {
        format!("{}u", self)
    }
}
