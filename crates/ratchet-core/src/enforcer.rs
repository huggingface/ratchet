use std::ops::RangeInclusive;

use crate::{DType, RVec, Tensor};

#[derive(Debug, thiserror::Error)]
pub enum InvariantError {
    #[error("Shape mismatch at {left},{right}, {a} != {b}.")]
    ShapeMismatch {
        left: usize,
        right: usize,
        a: usize, //TODO: RDim
        b: usize,
    },
    #[error("Rank mismatch. {accepted:?} != {actual}.")]
    RankMismatch {
        accepted: RangeInclusive<usize>,
        actual: usize,
    },
    #[error("Wrong input arity. Allowed range is {accepted:?}, node has {actual}.")]
    InputArity {
        accepted: RangeInclusive<usize>,
        actual: usize,
    },
    #[error("Wrong output arity. Allowed is {accepted:?}, node has {actual}.")]
    OutputArity {
        accepted: RangeInclusive<usize>,
        actual: usize,
    },
    #[error("DType mismatch, expected {expected:?}, got {actual:?}.")]
    DTypeMismatch { expected: DType, actual: DType },
    #[error("Unsupported DType {0:?}.")]
    UnsupportedDType(DType),
    #[error("Duplicate dims in permutation.")]
    DuplicateDims,
}

/// # Enforcer
///
/// Enforcer enforces common invariants on tensors.
pub struct Enforcer;

impl Enforcer {
    pub fn check_input_arity(inputs: &[&Tensor], expected: usize) -> Result<(), InvariantError> {
        Self::check_input_arity_range(inputs, expected..=expected + 1)
    }

    pub fn check_input_arity_range(
        inputs: &[&Tensor],
        accepted: RangeInclusive<usize>,
    ) -> Result<(), InvariantError> {
        let actual = inputs.len();
        if !accepted.contains(&actual) {
            return Err(InvariantError::InputArity { accepted, actual });
        }
        Ok(())
    }

    pub fn check_shape_pair(
        at: &Tensor,
        bt: &Tensor,
        left: usize,
        right: usize,
    ) -> Result<(), InvariantError> {
        let a = at.shape()[left];
        let b = bt.shape()[right];
        if a != b {
            return Err(InvariantError::ShapeMismatch { left, right, a, b });
        }
        Ok(())
    }

    pub fn match_shapes_at_index(
        tensors: &RVec<Tensor>,
        index: usize,
    ) -> Result<(), InvariantError> {
        let shape = tensors[0].shape();
        for tensor in tensors.iter().skip(1) {
            if shape[index] != tensor.shape()[index] {
                return Err(InvariantError::ShapeMismatch {
                    left: index,
                    right: index,
                    a: shape[index],
                    b: tensor.shape()[index],
                });
            }
        }
        Ok(())
    }

    pub fn assert_rank(tensor: &Tensor, rank: usize) -> Result<(), InvariantError> {
        if tensor.rank() != rank {
            return Err(InvariantError::RankMismatch {
                accepted: rank..=rank + 1,
                actual: tensor.rank(),
            });
        }
        Ok(())
    }

    pub fn assert_dtype(tensor: &Tensor, expected: DType) -> Result<(), InvariantError> {
        let actual = tensor.dt();
        if actual != expected {
            return Err(InvariantError::DTypeMismatch { expected, actual });
        }
        Ok(())
    }

    pub fn assert_rank_range(
        tensor: &Tensor,
        accepted: RangeInclusive<usize>,
    ) -> Result<(), InvariantError> {
        let actual = tensor.rank();
        if !accepted.contains(&actual) {
            return Err(InvariantError::RankMismatch { accepted, actual });
        }
        Ok(())
    }

    pub fn assert_equal_ranks(tensors: &[&Tensor]) -> Result<usize, InvariantError> {
        let rank = tensors[0].rank();
        for tensor in tensors.iter().skip(1) {
            if rank != tensor.rank() {
                return Err(InvariantError::RankMismatch {
                    accepted: rank..=rank + 1,
                    actual: tensor.rank(),
                });
            }
        }
        Ok(rank)
    }

    pub fn check_dtype_match(tensors: &[&Tensor]) -> Result<DType, InvariantError> {
        let dtype = tensors[0].dt();
        for tensor in tensors.iter().skip(1) {
            if dtype != tensor.dt() {
                return Err(InvariantError::DTypeMismatch {
                    expected: dtype,
                    actual: tensor.dt(),
                });
            }
        }
        Ok(dtype)
    }
}
