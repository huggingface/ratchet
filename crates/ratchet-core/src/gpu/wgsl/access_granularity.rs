use super::dtype::WgslDType;

/// WGSL types which are used to access buffers.
pub trait Accessor<T: WgslDType, const W: usize> {
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

impl<T: WgslDType> Accessor<T, 4> for Vec4<T> {
    fn render() -> String {
        format!("vec4<{}>", T::DT)
    }
}

impl<T: WgslDType> Accessor<T, 3> for Vec3<T> {
    fn render() -> String {
        format!("vec3<{}>", T::DT)
    }
}

impl<T: WgslDType> Accessor<T, 2> for Vec2<T> {
    fn render() -> String {
        format!("vec2<{}>", T::DT)
    }
}

impl<T: WgslDType> Accessor<T, 1> for Scalar<T> {
    fn render() -> String {
        format!("{}", T::DT)
    }
}
