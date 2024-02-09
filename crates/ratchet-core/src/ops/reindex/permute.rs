use std::collections::HashSet;

use derive_new::new;

use crate::{Enforcer, InvariantError, Operation, OperationError, StorageView, Strides, Tensor};

#[derive(new, Debug, Clone)]
pub struct Permute {
    pub dims: Vec<usize>,
}

impl Permute {
    pub fn promote(&self) -> Vec<usize> {
        let pad_len = 4 - self.dims.len();

        let mut perm = self.dims.clone();
        for i in 0..perm.len() {
            perm[i] += pad_len;
        }
        (0..pad_len).for_each(|x| perm.insert(0, x));
        perm
    }
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
    use crate::{
        rvec, shape, test_util::run_py_prg, Device, DeviceRequest, Permute, Shape, Tensor,
    };
    use proptest::prelude::*;
    use test_strategy::{proptest, Arbitrary};

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

    impl Arbitrary for PermuteProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            let s = Shape::arbitrary_with(rvec![1..8, 1..8, 1..512, 1..512]);
            let p = Permute::arbitrary_with(());

            todo!()
        }
    }

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    #[derive(Debug)]
    struct PermuteProblem {
        op: Permute,
        shape: Shape,
    }

    fn ground_truth(a: &Tensor, args: &str) -> anyhow::Result<Tensor> {
        let prg = format!(
            r#"
import torch
import numpy as np
def permute(a):
    return np.ascontiguousarray(torch.permute(torch.from_numpy(a), {}).numpy())
"#,
            args
        );
        run_py_prg(prg.to_string(), &[a])
    }

    fn run_reindex_trial(prob: PermuteProblem) -> anyhow::Result<()> {
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let PermuteProblem { op, shape } = prob;
        println!("Permute: {:?} {:?}", op, shape);
        let input = Tensor::randn::<f32>(shape, cpu_device.clone());
        let device = GPU_DEVICE.with(|d| d.clone());

        let ground = ground_truth(&input, format!("{:?}", op.dims).as_str())?;
        let ours = input.to(&device)?.permute(&op.dims)?;
        ours.resolve()?;
        let cpu_result = ours.to(&Device::CPU)?;
        ground.all_close(&cpu_result, 1e-5, 1e-5)?;
        Ok(())
    }

    #[proptest(cases = 16)]
    fn test_permute(prob: PermuteProblem) {
        run_reindex_trial(prob).unwrap();
    }
}
