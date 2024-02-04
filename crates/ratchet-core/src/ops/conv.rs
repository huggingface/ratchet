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
    stride: u32,
    padding: u32,
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
        let (input_t, weight_t) = (srcs[0], srcs[1]);
        let (input_shape, weight_shape) = (input_t.shape(), weight_t.shape());
        let calc_dim = |i_size, k_size, pad, dil, stride| {
            ((i_size + (2 * pad) - dil * (k_size - 1) - 1) / stride) + 1 //TODO: Missing floor
        };
        let [N, C_in, L_in] = input_shape.try_into()?;
        let [C_out, _, KS] = weight_shape.try_into()?;
        assert!(KS == 3, "Only 3 kernel size is supported");

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
        rvec![&self.input, &self.weight, self.bias.as_ref().unwrap()]
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
        let wgcx = WorkgroupCount::div_ceil(padded_strided_Lin, 256);
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
        let Fperthread = WorkgroupCount::div_ceil(F_numel, 256);

        Ok(ConvMeta::new(
            self.stride as _,
            self.padding as _,
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
    use crate::{shape, Conv, Device, DeviceRequest, Tensor};

    fn ground_truth(
        input: &Tensor,
        filters: &Tensor,
        bias: &Tensor,
        stride: usize,
        padding: usize,
    ) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
import torch.nn.functional as F
def conv(input, filters, bias, stride, padding):
    input = torch.from_numpy(input)
    filters = torch.from_numpy(filters)
    bias = torch.from_numpy(bias)
    stride = int(stride.item())
    padding = int(padding.item())
    return F.conv1d(input, filters, bias, stride=stride, padding=padding).numpy()
"#;
        let stride = Tensor::from_data(&[stride as f32], shape![1], Device::CPU);
        let padding = Tensor::from_data(&[padding as f32], shape![1], Device::CPU);
        run_py_prg(prg.to_string(), &[input, filters, bias, &stride, &padding])
    }

    fn run_conv_trial(device: &Device, problem: ConvProblem) {
        let ConvProblem {
            Cin,
            Lin,
            Cout,
            stride,
        } = problem;
        let input = Tensor::randn::<f32>(shape![1, Cin, Lin], Device::CPU);
        let weight = Tensor::randn::<f32>(shape![Cout, Cin, 3], Device::CPU);
        let bias = Tensor::randn::<f32>(shape![Cout], Device::CPU);
        let ground = ground_truth(&input, &weight, &bias, stride, 1).unwrap();

        let input = input.to(device).unwrap();
        let weight = weight.to(device).unwrap();
        let bias = bias.to(device).unwrap();
        let ours = input.conv1d(&weight, Some(&bias), stride, 1).unwrap();
        ours.resolve().unwrap();
        let ours = ours.to(&Device::CPU).unwrap();

        println!("ours = {:?}", ours);
        println!("ground = {:?}", ground);
        ground.all_close(&ours, 1e-3, 1e-3).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct ConvProblem {
        #[strategy(16..=2048usize)]
        Cin: usize,
        #[strategy(16..=2048usize)]
        #[filter(#Lin % 3 == 0)]
        Lin: usize,
        #[strategy(16..=2048usize)]
        Cout: usize,
        #[strategy(1..=2usize)]
        stride: usize,
    }

    #[proptest(cases = 1)]
    fn test_conv(prob: ConvProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let ConvProblem {
            Cin,
            Lin,
            Cout,
            stride,
        } = prob;
        println!(
            "Cin = {}, Lin = {}, Cout = {}, stride = {}",
            Cin, Lin, Cout, stride
        );
        run_conv_trial(&device, prob);
    }
}
