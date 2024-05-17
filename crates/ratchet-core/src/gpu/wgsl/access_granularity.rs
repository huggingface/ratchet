use super::dtype::WgslDType;

pub struct Vec4;
pub struct Vec2;
pub struct Scalar;

pub trait AccessGranularity {
    fn as_size(&self) -> usize;
    fn render(&self, dtype: WgslDType) -> String {
        let size = self.as_size();
        match size {
            4 => format!("vec4<{}>", dtype),
            2 => format!("vec2<{}>", dtype),
            1 => format!("{}", dtype),
            _ => panic!("Invalid access granularity"),
        }
    }
}

impl AccessGranularity for Vec4 {
    fn as_size(&self) -> usize {
        4
    }
}
impl AccessGranularity for Vec2 {
    fn as_size(&self) -> usize {
        2
    }
}
impl AccessGranularity for Scalar {
    fn as_size(&self) -> usize {
        1
    }
}
