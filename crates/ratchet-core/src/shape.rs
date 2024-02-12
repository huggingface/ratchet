use crate::{shape, RVec};
use encase::impl_wrapper;
use std::ops::RangeTo;

#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct Shape(RVec<usize>);

impl_wrapper!(Shape; using);

impl Shape {
    pub fn new(shape: RVec<usize>) -> Self {
        Self(shape)
    }

    pub fn inner(&self) -> &RVec<usize> {
        &self.0
    }

    pub fn get(&self, index: usize) -> Option<&usize> {
        self.0.get(index)
    }

    pub fn insert(&mut self, index: usize, dim: usize) {
        self.0.insert(index, dim);
    }

    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    pub fn to_vec(&self) -> Vec<usize> {
        self.0.to_vec()
    }

    pub fn iter(&self) -> impl Iterator<Item = &usize> {
        self.0.iter()
    }

    pub fn into_iter(self) -> impl Iterator<Item = usize> {
        self.0.into_iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn rank(&self) -> usize {
        self.len()
    }

    pub fn push(&mut self, dim: usize) {
        self.0.push(dim);
    }

    pub fn is_scalar(&self) -> bool {
        self.0.iter().all(|&x| x == 1)
    }

    #[inline]
    pub fn left_pad_to(&mut self, scalar: usize, rank: usize) {
        while self.0.len() < rank {
            self.0.insert(0, scalar);
        }
    }

    #[inline]
    pub fn right_pad_to(&mut self, scalar: usize, rank: usize) {
        while self.0.len() < rank {
            self.0.push(scalar);
        }
    }

    pub fn drain<R>(&mut self, range: R) -> smallvec::Drain<'_, [usize; 4]>
    where
        R: std::ops::RangeBounds<usize>,
    {
        self.0.drain(range)
    }

    pub fn slice(&self, range: std::ops::Range<usize>) -> Self {
        Shape(self.0[range].to_vec().into())
    }

    pub fn multi_broadcast(shapes: &[&Shape]) -> Option<Shape> {
        let max_rank = shapes.iter().map(|shape| shape.rank()).max()?;
        let mut shape: Shape = shape![];
        for i in 0..max_rank {
            let mut current_dim_size = 1;
            for shape in shapes {
                let len = shape.rank();
                let dim = if i < len { &shape[len - i - 1] } else { &1 };
                if dim != &1 {
                    if current_dim_size != 1 && dim != &current_dim_size {
                        return None;
                    }
                    current_dim_size = *dim;
                }
            }
            shape.0.insert(0, current_dim_size)
        }
        Some(shape)
    }
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut shape = format!("[{}", self.0.first().unwrap_or(&0));
        for dim in self.0.iter().skip(1) {
            shape.push_str(&format!("x{}", dim));
        }
        write!(f, "{}]", shape)
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl std::ops::Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl std::ops::Index<RangeTo<usize>> for Shape {
    type Output = [usize];

    fn index(&self, index: RangeTo<usize>) -> &Self::Output {
        &self.0[index]
    }
}

impl From<Vec<usize>> for Shape {
    fn from(shape: Vec<usize>) -> Self {
        Self(shape.into())
    }
}

impl From<Vec<u32>> for Shape {
    fn from(shape: Vec<u32>) -> Self {
        Self(shape.into_iter().map(|x| x as usize).collect())
    }
}

impl From<&[usize]> for Shape {
    fn from(slice: &[usize]) -> Self {
        Shape(slice.into())
    }
}

impl From<&Shape> for glam::UVec4 {
    fn from(shape: &Shape) -> Self {
        glam::UVec4::new(
            shape[0] as u32,
            shape[1] as u32,
            shape[2] as u32,
            shape[3] as u32,
        )
    }
}

macro_rules! impl_try_into_for_shape {
    ($($N:expr),*) => {
        $(
            impl TryInto<[usize; $N]> for &Shape {
                type Error = anyhow::Error;

                fn try_into(self) -> Result<[usize; $N], Self::Error> {
                    if self.0.len() == $N {
                        let mut arr = [0; $N];
                        for (i, &item) in self.0.iter().enumerate().take($N) {
                            arr[i] = item;
                        }
                        Ok(arr)
                    } else {
                        Err(anyhow::anyhow!("Shape has length {} but expected {}", self.0.len(), $N))
                    }
                }
            }
        )*
    };
}

impl_try_into_for_shape!(0, 1, 2, 3, 4);

#[cfg(test)]
mod tests {
    use crate::Shape;
    use proptest::prelude::*;
    use proptest::strategy::{BoxedStrategy, Strategy};
    use std::ops::Range;

    impl Arbitrary for Shape {
        type Parameters = Vec<Range<usize>>;
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(mut args: Self::Parameters) -> Self::Strategy {
            args.prop_map(move |shape| Into::<Shape>::into(shape))
                .boxed()
        }
    }

    impl Shape {
        pub fn as_torch(&self) -> String {
            let mut shape = format!("({}", self[0]);
            for dim in self.iter().skip(1) {
                shape.push_str(&format!(", {}", dim));
            }
            shape.push(')');
            shape
        }
    }
}
