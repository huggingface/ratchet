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
    use crate::{shape, test_util::run_py_prg, Device, DeviceRequest, Permute, Tensor};
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

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct PermuteProblem {
        op: Permute,
        #[strategy(1..=2usize)]
        B: usize,
        #[strategy(1..=4usize)]
        M: usize,
        #[strategy(1..=256usize)]
        N: usize,
        #[strategy(1..=256usize)]
        K: usize,
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
        let PermuteProblem { op, B, M, N, K } = prob;
        println!("Permute: {:?}, B: {}, M: {}, N: {}, K: {}", op, B, M, N, K);
        let a = Tensor::randn::<f32>(shape![B, M, N, K], cpu_device.clone());
        let device = GPU_DEVICE.with(|d| d.clone());

        let a_gpu = a.to(&device)?;
        let ground = ground_truth(&a, format!("{:?}", op.dims).as_str())?;
        let ours = a_gpu.permute(&op.dims)?;
        ours.resolve()?;
        let d_gpu = ours.to(&Device::CPU)?;
        ground.all_close(&d_gpu, 1e-5, 1e-5)?;
        Ok(())
    }

    #[proptest(cases = 16)]
    fn test_permute(prob: PermuteProblem) {
        run_reindex_trial(prob).unwrap();
    }
}
