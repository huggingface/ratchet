use encase::impl_wrapper;

use crate::{rvec, RVec, Shape};

#[derive(Clone, PartialEq, Eq, Default, Hash)]
pub struct Strides(RVec<isize>);

impl_wrapper!(Strides; using);

impl Strides {
    pub fn to_vec(&self) -> Vec<isize> {
        self.0.to_vec()
    }
}

impl std::fmt::Debug for Strides {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut shape = String::from("[");
        for (i, dim) in self.0.iter().enumerate() {
            if i == 0 {
                shape.push_str(&format!("{}", dim));
            } else {
                shape.push_str(&format!("x{}", dim));
            }
        }
        write!(f, "{}]", shape)
    }
}

impl From<&Shape> for Strides {
    fn from(shape: &Shape) -> Self {
        let shape = shape.iter().map(|x| *x as isize).collect::<RVec<_>>();

        let mut strides = rvec![];
        let mut stride = 1;
        for size in shape.iter().rev() {
            strides.push(stride);
            stride *= *size;
        }
        strides.reverse();
        Self(strides)
    }
}
