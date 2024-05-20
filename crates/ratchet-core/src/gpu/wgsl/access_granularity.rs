use super::dtype::WgslDType;

/// WGSL types which are used to access buffers.
pub trait WgslPrimitive<T: WgslDType, const W: usize> {
    fn render() -> String;
}

#[derive(Default)]
pub struct Vec4<T> {
    _phantom: std::marker::PhantomData<T>,
}

#[derive(Default)]
pub struct Vec3<T> {
    _phantom: std::marker::PhantomData<T>,
}

#[derive(Default)]
pub struct Vec2<T> {
    _phantom: std::marker::PhantomData<T>,
}

#[derive(Default)]
pub struct Scalar<T> {
    _phantom: std::marker::PhantomData<T>,
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
        format!("array<{}, {}>", P::render(), AN)
    }
}

impl<T: WgslDType> WgslPrimitive<T, 4> for Vec4<T> {
    fn render() -> String {
        format!("vec4<{}>", T::DT)
    }
}

impl<T: WgslDType> WgslPrimitive<T, 3> for Vec3<T> {
    fn render() -> String {
        format!("vec3<{}>", T::DT)
    }
}

impl<T: WgslDType> WgslPrimitive<T, 2> for Vec2<T> {
    fn render() -> String {
        format!("vec2<{}>", T::DT)
    }
}

impl<T: WgslDType> WgslPrimitive<T, 1> for Scalar<T> {
    fn render() -> String {
        T::DT.to_string()
    }
}
