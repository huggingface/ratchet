use crate::{prelude::*, OperationError, StorageView, Strides};
use crate::{Enforcer, Operation, RVec};
use std::ops::Range;

/// # Slice
///
/// This is a temporary, user hostile implementation.
#[derive(derive_new::new, Debug, Clone)]
pub struct Slice {
    indices: RVec<Range<usize>>,
}

impl Slice {
    pub fn indices(&self) -> &[Range<usize>] {
        &self.indices
    }
}

impl Operation for Slice {
    fn check_invariants(srcs: &[&Tensor]) -> Result<(), OperationError> {
        Enforcer::check_input_arity(srcs, 1)?;
        Ok(())
    }

    fn infer_output(&self, srcs: &[&Tensor]) -> Result<StorageView, OperationError> {
        //TODO: Check if slice is valid
        let mut output_shape = shape![0, 0, 0, 0];
        for (idx, range) in self.indices.iter().enumerate() {
            output_shape[idx] = range.end - range.start;
        }
        let strides = Strides::from(&output_shape);
        Ok(StorageView::new(output_shape, srcs[0].dt(), strides))
    }
}

#[cfg(test)]
mod tests {
    use crate::{rvec, Slice};
    use crate::{shape, test_util::run_py_prg, Device, DeviceRequest, Tensor};
    use proptest::prelude::*;
    use test_strategy::proptest;

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    impl Slice {
        fn as_torch(&self) -> String {
            let mut s = String::from("[");
            for (idx, range) in self.indices.iter().enumerate() {
                if idx > 0 {
                    s.push_str(", ");
                }
                s.push_str(&format!("{}:{}", range.start, range.end));
            }
            s.push(']');
            s
        }
    }

    impl Arbitrary for Slice {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            todo!()
        }
    }

    #[derive(Debug)]
    struct SliceProblem {
        op: Slice,
        B: usize,
        M: usize,
        N: usize,
        K: usize,
    }

    impl Arbitrary for SliceProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            let gen_range = |max: usize| {
                // Ensure the range end is always less than max
                (0..max, 0..max).prop_map(move |(start, end)| {
                    let (start, end) = match (start, end) {
                        (start, end) if start == end => (start, end + 1),
                        (start, end) if start > end => (end, start),
                        (start, end) => (start, end),
                    };
                    start..end
                })
            };

            let B_range = gen_range(4);
            let M_range = gen_range(4);
            let N_range = gen_range(512);
            let K_range = gen_range(512);

            (B_range, M_range, N_range, K_range)
                .prop_map(|(Br, Mr, Nr, Kr)| {
                    //Adding 10 to ensure it works without matching range end
                    let (B, M, N, K) = (Br.end, Mr.end, Nr.end + 10, Kr.end + 10);
                    let op = Slice {
                        indices: rvec![Br, Mr, Nr, Kr],
                    };
                    SliceProblem { op, B, M, N, K }
                })
                .boxed()
        }
    }

    fn ground_truth(a: &Tensor, args: &str) -> anyhow::Result<Tensor> {
        let prg = format!(
            r#"
import torch
import numpy as np
def slice(a):
    torch_a = torch.from_numpy(a)
    return np.ascontiguousarray(torch_a{})
"#,
            args
        );
        run_py_prg(prg.to_string(), &[a])
    }

    fn run_reindex_trial(prob: SliceProblem) -> anyhow::Result<()> {
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let SliceProblem { op, B, M, N, K } = prob;
        println!("Slice: {:?}, B: {}, M: {}, N: {}, K: {}", op, B, M, N, K);
        let a = Tensor::randn::<f32>(shape![B, M, N, K], cpu_device.clone());
        let device = GPU_DEVICE.with(|d| d.clone());

        let a_gpu = a.to(&device)?;
        let ground = ground_truth(&a, &op.as_torch())?;
        let ours = a_gpu.slice(&op.indices)?;
        ours.resolve()?;
        let d_gpu = ours.to(&Device::CPU)?;
        ground.all_close(&d_gpu, 1e-5, 1e-5)?;
        Ok(())
    }

    #[proptest(cases = 64)]
    fn test_slice(prob: SliceProblem) {
        run_reindex_trial(prob).unwrap();
    }
}
