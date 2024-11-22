use derive_new::new;

use super::*;

#[derive(new, Debug, Clone)]
pub struct GroupNorm {
    pub norm: Norm,
    pub num_groups: usize,
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::test_util::run_py_prg;
    use crate::{rvec, shape, Device, DeviceRequest, Tensor};

    fn ground_truth(
        input: &Tensor,
        scale: &Tensor,
        bias: Option<&Tensor>,
        num_groups: usize,
    ) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
import torch.nn.functional as F

def manual_group_norm(input, scale, bias, num_groups):
    (input, scale, bias) = (torch.from_numpy(input), torch.from_numpy(scale), torch.from_numpy(bias))
    return F.group_norm(input, num_groups, weight=scale, bias=bias).numpy()
"#;

        let inputs = match bias {
            Some(bias) => rvec![input, scale, bias],
            None => rvec![input, scale],
        };
        run_py_prg(prg.to_string(), &inputs, &[&num_groups], input.dt())
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

        let input = input.to(device)?;
        let scale = scale.to(device)?;
        let bias = bias.map(|b| b.to(device)).transpose()?;

        let result = input.group_norm(num_groups, scale, bias, 1e-5)?.resolve()?;

        let ours = result.to(&Device::CPU)?;
        println!("GROUND: {ground:?}");
        println!("OURS: {ours:?}");
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
    fn test_groupnorm_gpu(prob: GroupNormProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_norm_trial(&device, prob).unwrap();
    }

    #[proptest(cases = 64)]
    fn test_groupnorm_cpu(prob: GroupNormProblem) {
        let device = Device::request_device(DeviceRequest::CPU).unwrap();
        run_norm_trial(&device, prob).unwrap();
    }
}
