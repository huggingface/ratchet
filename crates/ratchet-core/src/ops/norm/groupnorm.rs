use derive_new::new;

use super::*;
use crate::{DType, OpGuards, Operation, OperationError, StorageView, Tensor};

#[derive(new, Debug, Clone)]
pub struct GroupNorm {
    pub norm: Norm,
    pub num_groups: usize,
}

impl OpGuards for GroupNorm {
    fn check_shapes(&self) {
        assert!(self.norm.input.rank() >= 3);
        let channels = self.norm.input.shape()[1];
        assert!(channels % self.num_groups == 0);
    }

    fn check_dtypes(&self) {
        assert!(self.norm.input.dt() == DType::F32);
        assert!(self.norm.scale.dt() == DType::F32);
        if self.norm.bias.is_some() {
            assert!(self.norm.bias.as_ref().unwrap().dt() == DType::F32);
        }
    }
}

impl Operation for GroupNorm {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        Ok(self.norm.input.storage_view().clone())
    }
}
#[cfg(all(test, feature = "testing"))]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::{rvec, shape, Device, DeviceRequest, Tensor};

    fn ground_truth(
        input: &Tensor,
        scale: &Tensor,
        bias: Option<&Tensor>,
        num_groups: usize,
    ) -> anyhow::Result<Tensor> {
        let input = input.to_tch::<f32>()?;
        let scale = scale.to_tch::<f32>()?;
        let bias = match bias {
            Some(b) => Some(b.to_tch::<f32>()?),
            None => None,
        };
        let result =
            input.f_group_norm(num_groups as i64, Some(&scale), bias.as_ref(), 1e-5, false)?;
        Tensor::try_from(&result)
    }

    fn run_norm_trial(device: &Device, problem: GroupNormProblem) -> anyhow::Result<()> {
        let GroupNormProblem {
            num_groups,
            B,
            C,
            N,
        } = problem;

        let input = Tensor::randn::<f32>(shape![B, C, N], Device::CPU);
        let scale = Tensor::randn::<f32>(shape![C], Device::CPU);
        let bias = Some(Tensor::randn::<f32>(shape![C], Device::CPU));

        let ground = ground_truth(&input, &scale, bias.as_ref(), num_groups)?;

        let input_gpu = input.to(device)?;
        let scale_gpu = scale.to(device)?;
        let bias_gpu = bias.map(|b| b.to(device)).transpose()?;

        let result = input_gpu
            .group_norm(num_groups, scale_gpu, bias_gpu, 1e-5)?
            .resolve()?;

        let ours = result.to(&Device::CPU)?;

        ground.all_close(&ours, 1e-4, 1e-4)?;
        Ok(())
    }

    #[derive(Arbitrary, Debug)]
    struct GroupNormProblem {
        #[map(|num_groups: u32| #C/2 )]
        num_groups: usize,
        #[strategy(1..=1usize)]
        B: usize,
        #[strategy(2..=4usize)]
        #[filter(#C % 2 != 0)]
        C: usize,
        #[strategy(1..=1usize)]
        N: usize,
    }

    #[proptest(cases = 64)]
    fn test_groupnorm(prob: GroupNormProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        println!("prob = {:#?}", prob);
        run_norm_trial(&device, prob).unwrap();
    }
}
