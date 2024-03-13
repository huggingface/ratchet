use crate::{prelude::*, OpGuards, OperationError, StorageView, Strides};
use crate::{Operation, RVec};
use std::ops::Range;

/// # Slice
///
/// This is a temporary, user hostile implementation.
#[derive(derive_new::new, Debug, Clone)]
pub struct Slice {
    pub src: Tensor,
    indices: RVec<Range<usize>>,
}

impl Slice {
    pub fn indices(&self) -> &[Range<usize>] {
        &self.indices
    }
}

impl OpGuards for Slice {
    fn check_shapes(&self) {
        self.indices.iter().for_each(|range| {
            assert!(range.start <= range.end);
        });
        self.indices
            .iter()
            .zip(self.src.shape().iter())
            .for_each(|(range, &dim)| {
                assert!(range.end <= dim);
            });
    }

    fn check_dtypes(&self) {}
}

impl Operation for Slice {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let output_shape = self
            .indices
            .iter()
            .map(|range| range.end - range.start)
            .collect::<RVec<usize>>()
            .into();
        let strides = Strides::from(&output_shape);
        Ok(StorageView::new(output_shape, self.src.dt(), strides))
    }
}

#[cfg(test)]
mod tests {
    use crate::{rvec, Shape, Slice};
    use crate::{test_util::run_py_prg, Device, DeviceRequest, Tensor};
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

    #[derive(Debug)]
    struct SliceProblem {
        op: Slice,
    }

    impl Arbitrary for SliceProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            Shape::arbitrary_with(vec![0..=4, 0..=4, 0..=128, 0..=128])
                .prop_map(|shape| {
                    let indices = rvec![0..shape[0], 0..shape[1], 0..shape[2], 0..shape[3]];
                    (shape, indices)
                })
                .prop_map(|(shape, indices)| SliceProblem {
                    op: Slice::new(Tensor::randn::<f32>(shape, Device::CPU), indices),
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
        run_py_prg(prg.to_string(), &[a], &[])
    }

    fn run_reindex_trial(prob: SliceProblem) -> anyhow::Result<()> {
        let SliceProblem { op } = prob;
        let device = GPU_DEVICE.with(|d| d.clone());
        let a = op.src.clone();

        let a_gpu = a.to(&device)?;
        let ground = ground_truth(&a, &op.as_torch())?;
        let ours = a_gpu.slice(&op.indices)?.resolve()?;
        let d_gpu = ours.to(&Device::CPU)?;
        ground.all_close(&d_gpu, 1e-5, 1e-5)?;
        Ok(())
    }

    #[proptest(cases = 16)]
    fn test_slice(prob: SliceProblem) {
        run_reindex_trial(prob).unwrap();
    }
}
