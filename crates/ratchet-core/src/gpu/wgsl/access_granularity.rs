use super::dtype::WgslDType;

/// WGSL types which are used to access buffers.
pub trait WgslPrimitive<T: WgslDType, const W: usize>: std::fmt::Display {
    fn render_type() -> String;
}

#[derive(Default)]
pub struct Vec4<T: WgslDType> {
    inner: [T; 4],
}

impl<T: WgslDType> std::fmt::Display for Vec4<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}, {}, {}, {}",
            self.inner[0].render(),
            self.inner[1].render(),
            self.inner[2].render(),
            self.inner[3].render()
        )
    }
}

#[derive(Default)]
pub struct Vec3<T: WgslDType> {
    inner: [T; 3],
}

impl<T: WgslDType> std::fmt::Display for Vec3<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}, {}, {}",
            self.inner[0].render(),
            self.inner[1].render(),
            self.inner[2].render()
        )
    }
}

#[derive(Default)]
pub struct Vec2<T: WgslDType> {
    inner: [T; 2],
}

impl<T: WgslDType> std::fmt::Display for Vec2<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}, {}", self.inner[0].render(), self.inner[1].render())
    }
}

#[derive(Default, derive_new::new)]
pub struct Scalar<T: WgslDType> {
    inner: T,
}

impl<T: WgslDType> std::fmt::Display for Scalar<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner.render())
    }
}

impl<T: WgslDType> WgslPrimitive<T, 4> for Vec4<T> {
    fn render_type() -> String {
        format!("vec4<{}>", T::DT)
    }
}

impl<T: WgslDType> WgslPrimitive<T, 3> for Vec3<T> {
    fn render_type() -> String {
        format!("vec3<{}>", T::DT)
    }
}

impl<T: WgslDType> WgslPrimitive<T, 2> for Vec2<T> {
    fn render_type() -> String {
        format!("vec2<{}>", T::DT)
    }
}

impl<T: WgslDType> WgslPrimitive<T, 1> for Scalar<T> {
    fn render_type() -> String {
        T::DT.to_string()
    }
}

#[derive(Default)]
pub struct Array<T: WgslDType, const N: usize, P: WgslPrimitive<T, N>, const AN: usize> {
    _p1: std::marker::PhantomData<P>,
    _p2: std::marker::PhantomData<T>,
}

pub trait WgslArray {
    fn render() -> String;
}

impl<T: WgslDType, const N: usize, P: WgslPrimitive<T, N>, const AN: usize> WgslArray
    for Array<T, N, P, AN>
{
    fn render() -> String {
        format!("array<{}, {}>", P::render_type(), AN)
    }
}
