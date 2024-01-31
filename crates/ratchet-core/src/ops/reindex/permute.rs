use std::collections::HashSet;

use derive_new::new;


use crate::{
    Enforcer, InvariantError, Operation,
    OperationError, StorageView, Strides, Tensor,
};

#[derive(new, Debug, Clone)]
pub struct Permute {
    pub dims: Vec<usize>,
}

impl Operation for Permute {
    fn infer_output(&self, srcs: &[&Tensor]) -> Result<StorageView, OperationError> {
        let input_shape = srcs[0].shape();
        if input_shape.rank() != self.dims.len() {
            return Err(InvariantError::RankMismatch {
                accepted: input_shape.rank()..=input_shape.rank(),
                actual: self.dims.len(),
            })?;
        }
        let dup_set: HashSet<usize> = HashSet::from_iter(self.dims.iter().cloned());
        if dup_set.len() != self.dims.len() {
            return Err(InvariantError::DuplicateDims)?;
        }

        let mut output_shape = input_shape.clone();
        for i in 0..input_shape.rank() {
            output_shape[i] = input_shape[self.dims[i]];
        }
        let strides = Strides::from(&output_shape);
        Ok(StorageView::new(output_shape, srcs[0].dt(), strides))
    }

    fn check_invariants(srcs: &[&Tensor]) -> Result<(), OperationError> {
        Enforcer::check_input_arity(srcs, 1)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::Permute;
    use proptest::prelude::*;

    impl Arbitrary for Permute {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            Just(vec![0, 1, 2, 3])
                .prop_shuffle()
                .prop_map(Permute::new)
                .boxed()
        }
    }
}
