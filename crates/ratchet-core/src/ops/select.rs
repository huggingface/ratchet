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
    dim_len: i32,
}

impl OpMetadata for IndexSelectMeta {}

impl Operation for IndexSelect {
    fn infer_output(&self, srcs: &[&Tensor]) -> Result<StorageView, OperationError> {
        let (input, indices) = (srcs[0], srcs[1]);
        let (indices_shape, input_shape) = (indices.shape(), input.shape());

        let mut output_shape = input_shape.clone();
        output_shape[self.dim] = indices_shape[0];
        let strides = Strides::from(&output_shape);
        Ok(StorageView::new(output_shape, input.dt(), strides))
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
        self.name()
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, _dst: &Tensor) -> Result<WorkgroupCount, OperationError> {
        let embedding_dim = self.input.shape()[self.dim] as _;
        let wgc_y = self.indices.shape().numel() as _;
        Ok(wgc![wgc_y, embedding_dim, 1])
    }

    fn storage_bind_group_layout(
        &self,
        _: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        Ok(BindGroupLayoutDescriptor::binary())
    }

    fn metadata(
        &self,
        _dst: &Tensor,
        _kernel_element: &KernelElement,
    ) -> Result<Self::Meta, OperationError> {
        let meta = IndexSelectMeta {
            dim_len: self.input.shape()[self.dim + 1] as _,
        };
        println!("meta: {:?}", meta);
        Ok(meta)
    }
}

#[cfg(test)]
mod tests {
    use proptest::arbitrary::Arbitrary;
    use proptest::strategy::{BoxedStrategy, Just, Strategy};
    use test_strategy::proptest;

    use crate::test_util::run_py_prg;
    use crate::{rvec, shape, Device, DeviceRequest, Shape, Tensor};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    impl Arbitrary for IndexSelectProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            Shape::arbitrary_with(vec![1..20000, 1..1024])
                .prop_flat_map(|input_shape| (Just(input_shape), 1..1024usize))
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
        run_py_prg(prg.to_string(), &[input, indices])
    }

    fn run_index_select_trial(problem: IndexSelectProblem) {
        let device = GPU_DEVICE.with(|d| d.clone());
        let IndexSelectProblem {
            input_shape,
            indices,
        } = problem;
        let input = Tensor::randn::<f32>(input_shape, Device::CPU);

        let ground_truth = ground_truth(&input, &indices, 0).unwrap();
        println!("ground_truth: {:?}", ground_truth);

        let input = input.to(&device).unwrap();
        let indices = indices.to(&device).unwrap();
        let dim = 0;

        let result = input.index_select(&indices, dim).unwrap();
        result.resolve().unwrap();
        let x = result.to(&Device::CPU).unwrap();
        println!("result: {:?}", x);
        ground_truth.all_close(&x, 1e-6, 1e-6).unwrap();
    }

    #[derive(Debug, Clone)]
    struct IndexSelectProblem {
        input_shape: Shape,
        indices: Tensor,
    }

    #[test]
    fn debug_index_select() {
        let problem = IndexSelectProblem {
            input_shape: shape![3, 4],
            indices: Tensor::from_data(vec![0, 2], shape![2], Device::CPU),
        };
        run_index_select_trial(problem);
    }

    #[proptest(cases = 1)]
    fn test_index_select(prob: IndexSelectProblem) {
        run_index_select_trial(prob);
    }
}
