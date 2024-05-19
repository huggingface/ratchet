use half::f16;

/// Supported data types in WGSL.
///
/// This can be mapped to and from the Ratchet DType.

pub trait WgslDType {
    fn render_dt() -> &'static str {
        unimplemented!()
    }
}

impl WgslDType for f32 {
    fn render_dt() -> &'static str {
        "f32"
    }
}

impl WgslDType for f16 {
    fn render_dt() -> &'static str {
        "f16"
    }
}

impl WgslDType for i32 {
    fn render_dt() -> &'static str {
        "i32"
    }
}

impl WgslDType for u32 {
    fn render_dt() -> &'static str {
        "u32"
    }
}
