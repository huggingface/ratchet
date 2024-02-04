//TODO: move this to a custom operation
use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, WorkgroupCount},
    rvec, shape, wgc, Enforcer, KernelElement, MetaOperation, OpMetadata, Operation,
    OperationError, RVec, StorageView, Strides, Tensor,
};

#[derive(new, Debug, Clone)]
pub struct Conv {
    input: Tensor,
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    //dilation: usize, TODO: implement dilation
}

impl Conv {
    pub fn name(&self) -> &'static str {
        "conv"
    }
}

#[derive(Debug, derive_new::new, ShaderType)]
pub struct ConvMeta {
    padding: u32,
    stride: u32,
    Cin: u32,
    Lin: u32,
    KS: u32,
    F_numel: u32,
    Lout: u32,
    Fperthread: u32,
}

impl OpMetadata for ConvMeta {}

impl Operation for Conv {
    fn infer_output(&self, srcs: &[&Tensor]) -> Result<StorageView, OperationError> {
        let (input_t, weight_t) = (srcs[0], &self.weight);
        let (input_shape, weight_shape) = (input_t.shape(), weight_t.shape());
        let calc_dim = |i_size, k_size, pad, dil, stride| {
            ((i_size + (2 * pad) - dil * (k_size - 1) - 1) / stride) + 1 //TODO: Missing floor
        };
        let [N, C_in, L_in] = input_shape.try_into()?;
        let [C_out, _, KS] = weight_shape.try_into()?;

        let L_out = calc_dim(L_in, KS, self.padding, 1, self.stride);
        let out_shape = shape![N, C_out, L_out];
        let out_strides = Strides::from(&out_shape);
        Ok(StorageView::new(out_shape, input_t.dt(), out_strides))
    }

    fn check_invariants(srcs: &[&Tensor]) -> Result<(), OperationError> {
        Enforcer::check_input_arity_range(srcs, 2..=3)?;
        Enforcer::assert_rank(srcs[0], 3)?;
        //TODO: exhaustive checks
        Ok(())
    }
}

impl MetaOperation for Conv {
    type Meta = ConvMeta;

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input]
    }

    fn kernel_name(&self) -> &'static str {
        "conv"
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, _dst: &Tensor) -> Result<WorkgroupCount, OperationError> {
        let input = &self.input;
        let [N, Cin, Lin] = input.shape().try_into()?;
        let [Cout, _, KS] = self.weight.shape().try_into()?;
        let F_numel = Cin * KS;
        let padded_strided_Lin = (Lin + 2 * self.padding) / self.stride;
        let wgcx = WorkgroupCount::div_ceil(F_numel, 256);
        Ok(wgc![wgcx as _, Cout as _, 1])
    }

    fn storage_bind_group_layout(
        &self,
        _: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        Ok(BindGroupLayoutDescriptor::ternary())
    }

    fn metadata(
        &self,
        dst: &Tensor,
        _kernel_element: &KernelElement,
    ) -> Result<Self::Meta, OperationError> {
        let [N, Cin, Lin] = self.input.shape().try_into()?;
        let [Cout, _, KS] = self.weight.shape().try_into()?;
        let [_, _, Lout] = dst.shape().try_into()?;
        let F_numel = Cin * KS;
        let Fperthread = F_numel / 256;

        Ok(ConvMeta::new(
            self.padding as _,
            self.stride as _,
            Cin as _,
            Lin as _,
            KS as _,
            F_numel as _,
            Lout as _,
            Fperthread as _,
        ))
    }
}

#[cfg(test)]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::test_util::run_py_prg;
    use crate::{shape, Device, DeviceRequest, Tensor};

    fn ground_truth(a: &Tensor) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
import torch.nn.functional as F
def conv(a):
    return F.conv1d(torch.from_numpy(a), dim=-1).numpy()
"#;
        run_py_prg(prg.to_string(), &[a])
    }

    fn run_conv_trial(device: &Device, problem: ConvProblem) {
        let ConvProblem { B, M, N } = problem;
        let a = Tensor::randn::<f32>(shape![B, M, N], Device::CPU);
        let ground = ground_truth(&a).unwrap();

        let a_gpu = a.to(device).unwrap();
        let b = a_gpu.conv(2).unwrap();
        b.resolve().unwrap();

        let ours = b.to(&Device::CPU).unwrap();
        println!("ours = {:?}", ours);
        println!("ground = {:?}", ground);
        ground.all_close(&ours, 1e-6, 1e-6).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct ConvProblem {
        #[strategy(1..=4usize)]
        B: usize,
        #[strategy(1..=512usize)]
        M: usize,
        #[strategy(1..=512usize)]
        N: usize,
    }

    #[proptest(cases = 16)]
    fn test_conv(prob: ConvProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let ConvProblem { B, M, N } = prob;
        println!("B = {}, M = {}, N = {}", B, M, N);
        run_conv_trial(&device, prob);
    }
}
