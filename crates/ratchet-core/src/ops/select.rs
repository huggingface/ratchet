use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, WorkgroupCount},
    rvec, wgc, DType, Enforcer, KernelElement, MetaOperation, OpMetadata, Operation,
    OperationError, RVec, StorageView, Strides, Tensor,
};

#[derive(new, Debug, Clone)]
pub struct IndexSelect {
    input: Tensor,
    indices: Tensor,
    dim: usize,
}

impl IndexSelect {
    pub fn name(&self) -> &'static str {
        "index_select"
    }
}

#[derive(Debug, derive_new::new, ShaderType)]
pub struct IndexSelectMeta {
    dst_numel: u32,
    left_numel: u32,
    right_numel: u32,
    ids_numel: u32,
    src_dim_numel: u32,
}

impl OpMetadata for IndexSelectMeta {}

impl Operation for IndexSelect {
    fn infer_output(&self, srcs: &[&Tensor]) -> Result<StorageView, OperationError> {
        let (input, indices) = (srcs[0], srcs[1]);
        let (indices_shape, input_shape) = (indices.shape(), input.shape());

        let mut output_shape = input_shape.clone();
        output_shape[self.dim] = indices_shape[0];
        let strides = Strides::from(&output_shape);
        Ok(StorageView::new(output_shape, DType::F32, strides))
    }

    fn check_invariants(srcs: &[&Tensor]) -> Result<(), OperationError> {
        let (input, indices) = (srcs[0], srcs[1]);
        Enforcer::assert_dtype(indices, DType::I32)?;
        Enforcer::assert_rank(input, 2)?;
        Enforcer::assert_rank(indices, 1)?;
        Ok(())
    }
}

impl MetaOperation for IndexSelect {
    type Meta = IndexSelectMeta;

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input, &self.indices]
    }

    fn kernel_name(&self) -> &'static str {
        match self.input.dt() {
            DType::F32 => "f32_index_select",
            DType::WQ8 => "wq8_index_select",
            _ => unimplemented!(),
        }
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<WorkgroupCount, OperationError> {
        let wgcx = WorkgroupCount::div_ceil(dst.shape().numel(), 64);
        Ok(wgc![wgcx as _, 1, 1])
    }

    fn storage_bind_group_layout(
        &self,
        _: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        match self.input.dt() {
            DType::F32 => Ok(BindGroupLayoutDescriptor::binary()),
            DType::WQ8 => Ok(BindGroupLayoutDescriptor::ternary()),
            _ => unimplemented!(),
        }
    }

    fn metadata(
        &self,
        dst: &Tensor,
        _kernel_element: &KernelElement,
    ) -> Result<Self::Meta, OperationError> {
        let dst_numel = dst.shape().numel() as u32;
        let left_numel = self.input.shape()[..self.dim].iter().product::<usize>() as u32;
        let right_numel = self.input.shape()[(self.dim + 1)..]
            .iter()
            .product::<usize>() as u32;
        let ids_numel = self.indices.shape().numel() as u32;
        let src_dim_numel = self.input.shape()[self.dim] as u32;
        Ok(IndexSelectMeta {
            dst_numel,
            left_numel,
            right_numel,
            ids_numel,
            src_dim_numel,
        })
    }
}

#[cfg(test)]
mod tests {
    use proptest::arbitrary::Arbitrary;
    use proptest::strategy::{BoxedStrategy, Just, Strategy};
    use test_strategy::proptest;

    use crate::test_util::run_py_prg;
    use crate::{rvec, shape, Device, DeviceRequest, Quantization, Quantizer, Shape, Tensor};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    impl Arbitrary for IndexSelectProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            Shape::arbitrary_with(vec![1..512usize, 1..16usize])
                .prop_flat_map(|input_shape| (Just(input_shape), 1..64usize))
                .prop_map(|(input_shape, num_indices)| {
                    let indices =
                        Tensor::randint(0, input_shape[0] as i32, shape![num_indices], Device::CPU);
                    IndexSelectProblem {
                        input_shape,
                        indices,
                    }
                })
                .boxed()
        }
    }

    fn ground_truth(input: &Tensor, indices: &Tensor, dim: usize) -> anyhow::Result<Tensor> {
        let prg = format!(
            r#"
import torch
def index_select(input, indices):
    return torch.index_select(torch.from_numpy(input),{},torch.from_numpy(indices)).numpy()
"#,
            dim
        );
        run_py_prg(prg.to_string(), &[input, indices], &[])
    }

    fn run_index_select_trial(problem: IndexSelectProblem, quantize: bool) {
        let device = GPU_DEVICE.with(|d| d.clone());
        let IndexSelectProblem {
            input_shape,
            indices,
        } = problem;
        let mut input = Tensor::randn::<f32>(input_shape, Device::CPU);

        let dim = 0;
        let ground_truth = ground_truth(&input, &indices, dim).unwrap();
        println!("ground_truth: {:?}", ground_truth);
        if quantize {
            let quantizer = Quantizer::new(Quantization::SInt8);
            input = quantizer.quantize(input);
        }

        let input = input.to(&device).unwrap();
        let indices = indices.to(&device).unwrap();

        let result = input
            .index_select(&indices, dim)
            .unwrap()
            .resolve()
            .unwrap();
        let x = result.to(&Device::CPU).unwrap();
        println!("result: {:?}", x);
        ground_truth.all_close(&x, 1e-2, 1e-2).unwrap();
    }

    #[test]
    fn qindex_select() {
        let prob = IndexSelectProblem {
            input_shape: shape![1024, 128],
            indices: Tensor::from_data(vec![0i32, 1, 2, 3], shape![4], Device::CPU),
        };
        run_index_select_trial(prob, true);
    }

    #[derive(Debug, Clone)]
    struct IndexSelectProblem {
        input_shape: Shape,
        indices: Tensor,
    }

    #[proptest(cases = 16)]
    fn test_index_select(prob: IndexSelectProblem) {
        run_index_select_trial(prob, false);
    }
}
