use derive_new::new;
use encase::ShaderType;
use ratchet_macros::WgslMetadata;

use crate::{
    gguf::GGUFDType,
    gpu::{BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, Array, BindingMode, BuiltIn, DType, KernelElement, KernelKey, KernelSource,
    MetaOperation, OpGuards, Operation, OperationError, RVec, Scalar, StorageView, Strides, Tensor,
    Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize,
};
use inline_wgsl::wgsl;

#[derive(new, Debug, Clone)]
pub struct IndexSelect {
    input: Tensor,
    indices: Tensor,
    dim: usize,
}

impl IndexSelect {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError> {
        if !inplace {
            panic!("Only inplace softmax is supported");
        }

        let index_arr = Array::<Scalar<i32>>::default();
        match self.input.dt() {
            DType::F16 | DType::F32 => {
                builder.register_storage("E", BindingMode::ReadOnly, Array::<P>::default());
                builder.register_storage("I", BindingMode::ReadWrite, index_arr);
            }
            DType::GGUF(g) => match g {
                GGUFDType::Q8_0(_) => {
                    let packed_arr = Array::<Scalar<u32>>::default();
                    let scale_arr = Array::<Scalar<f32>>::default();
                    builder.register_storage("E", BindingMode::ReadOnly, packed_arr);
                    builder.register_storage("S", BindingMode::ReadWrite, scale_arr);
                    builder.register_storage("I", BindingMode::ReadOnly, index_arr);
                }
                _ => unimplemented!(),
            },
            _ => unimplemented!(),
        }
        builder.register_uniform();
        Ok(())
    }

    fn build_index_select<P: WgslPrimitive>(
        &self,
        inplace: bool,
        _: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let device = self.input.device().try_gpu().unwrap();
        let mut kernel_builder = WgslKernelBuilder::new(
            workgroup_size.clone(),
            rvec![
                BuiltIn::GlobalInvocationId,
                BuiltIn::LocalInvocationId,
                BuiltIn::WorkgroupId,
            ],
            device.compute_features().clone(),
        );
        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        kernel_builder.write_metadata::<IndexSelectMeta>();

        kernel_builder.write_main(wgsl! {
            let tid = (group_id.x * 64u + local_index);
            let right_numel = metadata.right_numel/ 4u;
            let src_dim_numel = metadata.src_dim_numel/ 4u;

            if (tid >= metadata.dst_numel / 4u) {
                return;
            }

            let id_i = (tid / right_numel) % metadata.ids_numel;
            let input_i = min(u32(I[id_i]), (src_dim_numel * 4u) - 1u);
            let right_rank_i = tid % right_numel;
            let left_rank_i = tid / (right_numel * metadata.ids_numel);

            let src_i = left_rank_i * src_dim_numel * right_numel + input_i * right_numel + right_rank_i;
            Y[tid] = unpack4x8snorm_gguf(E[src_i]) * S[src_i / 8u];

        });

        Ok(kernel_builder.build()?)
    }
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct IndexSelectMeta {
    dst_numel: u32,
    right_numel: u32,
    ids_numel: u32,
    src_dim_numel: u32,
}

impl Operation for IndexSelect {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let (input, indices) = (&self.input, &self.indices);
        let (indices_shape, input_shape) = (indices.shape(), input.shape());

        let mut output_shape = input_shape.clone();
        output_shape[self.dim] = indices_shape[0];
        let strides = Strides::from(&output_shape);
        Ok(StorageView::new(output_shape, DType::F32, strides))
    }
}

impl OpGuards for IndexSelect {
    fn check_shapes(&self) {
        let (input, indices) = (&self.input, &self.indices);
        assert_eq!(input.rank(), 2);
        assert_eq!(indices.rank(), 1);
    }

    fn check_dtypes(&self) {
        let indices = &self.indices;
        //TODO: support others
        assert_eq!(indices.dt(), DType::I32);
    }
}

impl MetaOperation for IndexSelect {
    fn kernel_name(&self) -> String {
        "index_select".to_string()
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input, &self.indices]
    }

    fn kernel_key(&self, _: bool, dst: &Tensor) -> KernelKey {
        let op_key = match self.input.dt() {
            DType::F32 => "f32_index_select",
            DType::GGUF(_) => "wq8_index_select",
            _ => unimplemented!(),
        };
        KernelKey::new(format!("{}_{}", op_key, self.kernel_element(dst).as_str()))
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<WorkgroupCount, OperationError> {
        let numel = match self.input.dt() {
            DType::F32 => dst.shape().numel(),
            DType::GGUF(_) => dst.shape().numel() / 4,
            _ => unimplemented!(),
        };
        let wgcx = WorkgroupCount::div_ceil(numel, 64);
        Ok(wgc![wgcx as _, 1, 1])
    }

    fn storage_bind_group_layout(
        &self,
        _: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        match self.input.dt() {
            DType::F32 => Ok(BindGroupLayoutDescriptor::binary()),
            DType::GGUF(_) => Ok(BindGroupLayoutDescriptor::ternary()),
            _ => unimplemented!(),
        }
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        dst: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        let dst_numel = dst.shape().numel() as u32;
        let right_numel = self.input.shape()[(self.dim + 1)..]
            .iter()
            .product::<usize>() as u32;
        let ids_numel = self.indices.shape().numel() as u32;
        let src_dim_numel = self.input.shape()[self.dim] as u32;

        let meta = IndexSelectMeta {
            dst_numel,
            right_numel,
            ids_numel,
            src_dim_numel,
        };
        Ok(uniform.write(&meta)?)
    }
}

#[cfg(all(test, feature = "pyo3"))]
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
            Shape::arbitrary_with(vec![1..=512usize, 1..=16usize])
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

        let ground_truth = ground_truth(&input, &indices, 0).unwrap();
        if quantize {
            let quantizer = Quantizer::new(Quantization::SInt8);
            input = quantizer.quantize(input);
        }

        let input = input.to(&device).unwrap();
        let indices = indices.to(&device).unwrap();

        let result = input.index_select(indices, 0).unwrap().resolve().unwrap();
        let x = result.to(&Device::CPU).unwrap();
        ground_truth.all_close(&x, 1e-1, 1e-1).unwrap();
    }

    #[test]
    fn qindex_select() {
        let prob = IndexSelectProblem {
            input_shape: shape![4000, 384],
            indices: Tensor::from_data(vec![3i32, 4i32, 1000i32], shape![3], Device::CPU),
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
