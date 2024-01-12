use crate::RVec;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape(RVec<usize>);

impl Shape {
    pub fn new(shape: &[usize]) -> Self {
        Self(shape.into())
    }

    pub fn to_vec(&self) -> Vec<usize> {
        self.0.to_vec()
    }

    pub fn iter(&self) -> impl Iterator<Item = &usize> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl std::fmt::Debug for Shape {
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

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
