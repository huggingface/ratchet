use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, WorkgroupCount},
    rvec, wgc, Enforcer, InvariantError, KernelElement, MetaOperation, OpMetadata, Operation,
    OperationError, RVec, StorageView, Strides, Tensor,
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

        let mut output_shape = input_shape.clone();
        for i in 0..input_shape.rank() {
            output_shape[i] = input_shape[self.dims[i]].clone();
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
    use {
        proptest::{
            arbitrary::any,
            collection::size_range,
            prelude::Arbitrary,
            strategy::{BoxedStrategy, Strategy},
        },
        test_strategy::proptest,
    };

    impl Arbitrary for Permute {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            (1..=4usize)
                .prop_flat_map(|rank| {
                    let dims = (0..rank).collect::<Vec<_>>();
                    let dims = proptest::collection::vec(
                        proptest::sample::select(dims),
                        size_range(1..=rank),
                    );
                    dims
                })
                .prop_map(|dims| Permute::new(dims))
                .boxed()
        }
    }
}
