use derive_new::new;

use crate::{Enforcer, Operation, OperationError, Shape, StorageView, Strides, Tensor};

#[derive(new, Debug, Clone)]
pub struct Broadcast {
    to: Shape,
}

impl Broadcast {
    pub fn to(&self) -> &Shape {
        &self.to
    }
}

impl Operation for Broadcast {
    //For rules, see https://numpy.org/doc/stable/user/basics.broadcasting.html
    fn infer_output(&self, srcs: &[&Tensor]) -> Result<StorageView, OperationError> {
        let src = srcs[0];
        let src_shape = src.shape();

        //Check if shapes are compatible
        if *src_shape == self.to {
            return Ok(src.storage_view().clone());
        }

        //TODO: actually validate the shapes, currently faith based system
        let strides = Strides::from(&self.to);
        Ok(StorageView::new(self.to.clone(), src.dt(), strides))
    }

    fn check_invariants(srcs: &[&Tensor]) -> Result<(), OperationError> {
        Enforcer::check_input_arity(srcs, 1)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use proptest::{
        arbitrary::Arbitrary,
        strategy::{BoxedStrategy, Just, Strategy},
    };
    use test_strategy::proptest;

    use crate::{rvec, test_util::run_py_prg, Broadcast, Device, DeviceRequest, Shape, Tensor};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    impl Arbitrary for BroadcastProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: ()) -> Self::Strategy {
            let original_ranges = rvec![1..2, 1..8, 1..2, 1..128];

            Shape::arbitrary_with(original_ranges)
                .prop_flat_map(|original_shape| {
                    let create_broadcast_range = |dim: usize| {
                        if original_shape[dim] == 1 {
                            1..8
                        } else {
                            original_shape[dim]..original_shape[dim] + 1
                        }
                    };

                    let to = Shape::arbitrary_with(rvec![
                        create_broadcast_range(0),
                        create_broadcast_range(1),
                        create_broadcast_range(2),
                        create_broadcast_range(3)
                    ]);
                    (Just(original_shape), to)
                })
                .prop_map(|(original_shape, to)| BroadcastProblem {
                    original_shape,
                    op: Broadcast::new(to),
                })
                .boxed()
        }
    }

    #[derive(Debug, Clone)]
    struct BroadcastProblem {
        original_shape: Shape,
        op: Broadcast,
    }

    fn ground_truth(a: &Tensor, args: &str) -> anyhow::Result<Tensor> {
        let prg = format!(
            r#"
import torch
import numpy as np
def slice(a):
    torch_a = torch.from_numpy(a)
    return np.ascontiguousarray(torch_a.broadcast_to({}).numpy())
"#,
            args
        );
        run_py_prg(prg.to_string(), &[a])
    }

    fn run_reindex_trial(prob: BroadcastProblem) -> anyhow::Result<()> {
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        println!("Broadcast problem: {:?}", prob);
        let BroadcastProblem { original_shape, op } = prob;
        let a = Tensor::randn::<f32>(original_shape, cpu_device.clone());
        let device = GPU_DEVICE.with(|d| d.clone());

        let a_gpu = a.to(&device)?;
        let ground = ground_truth(&a, &op.to.as_torch())?;
        let ours = a_gpu.broadcast_to(op.to.clone())?;
        ours.resolve()?;
        let d_gpu = ours.to(&Device::CPU)?;
        ground.all_close(&d_gpu, 1e-5, 1e-5)?;
        Ok(())
    }

    #[proptest(cases = 16)]
    fn test_broadcast(prob: BroadcastProblem) {
        run_reindex_trial(prob).unwrap();
    }
}
