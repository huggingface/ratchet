use super::dtype::WgslDType;

/// WGSL types which are used to access buffers.
pub trait WgslPrimitive: std::fmt::Display {
    type T: WgslDType;
    const W: usize;

    fn render_type() -> String;
}

pub struct WgslVec<T: WgslDType, const N: usize> {
    inner: [T; N],
}

impl<T: WgslDType, const N: usize> Default for WgslVec<T, N> {
    fn default() -> Self {
        WgslVec {
            inner: [T::default(); N],
        }
    }
}

impl<T: WgslDType, const N: usize> std::fmt::Display for WgslVec<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..N - 2 {
            write!(f, "{}, ", self.inner[i].render())?;
        }
        write!(f, "{}", self.inner[N - 1].render())?;
        Ok(())
    }
}

pub type Scalar<T> = WgslVec<T, 1>;
pub type Vec2<T> = WgslVec<T, 2>;
pub type Vec3<T> = WgslVec<T, 3>;
pub type Vec4<T> = WgslVec<T, 4>;

impl<T: WgslDType> WgslPrimitive for Vec4<T> {
    type T = T;
    const W: usize = 4;

    fn render_type() -> String {
        format!("vec4<{}>", T::DT)
    }
}

impl<T: WgslDType> WgslPrimitive for Vec3<T> {
    type T = T;
    const W: usize = 3;
    fn render_type() -> String {
        format!("vec3<{}>", T::DT)
    }
}

impl<T: WgslDType> WgslPrimitive for Vec2<T> {
    type T = T;
    const W: usize = 2;
    fn render_type() -> String {
        format!("vec2<{}>", T::DT)
    }
}

impl<T: WgslDType> WgslPrimitive for Scalar<T> {
    type T = T;
    const W: usize = 1;
    fn render_type() -> String {
        T::DT.to_string()
    }
}

#[derive(Default)]
pub struct Array<const N: usize, P: WgslPrimitive, const AN: usize> {
    _p1: std::marker::PhantomData<P>,
}

pub trait WgslArray {
    fn render() -> String;
}

impl<const N: usize, P: WgslPrimitive, const AN: usize> WgslArray for Array<N, P, AN> {
    fn render() -> String {
        format!("array<{}, {}>", P::render_type(), AN)
    }
}
