use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use ratchet_macros::WgslMetadata;

use crate::{
    gpu::{dtype::WgslDType, BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, wgs, Array, BindingMode, BuiltIn, CompiledOp, ComputePipelineDescriptor, DType,
    KernelElement, KernelKey, KernelSource, KernelSourceDesc, MetaOperation, OpGuards, Operation,
    OperationError, PipelineLayoutDescriptor, RVec, Scalar, StorageView, Tensor, Vec2, Vec4,
    WgpuDevice, WgslKernelBuilder, WgslPrimitive, WorkgroupSize,
};

#[derive(new, Debug, Clone)]
pub struct Softmax {
    input: Tensor,
    dim: usize,
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct SoftmaxMeta {
    M: u32,
    N: u32,
    ND2: u32,
    ND4: u32,
}

impl OpGuards for Softmax {
    fn check_shapes(&self) {
        let input = &self.input;
        assert!(input.rank() >= 2);
        assert!(self.dim < input.rank());
    }

    fn check_dtypes(&self) {
        let input = &self.input;
        assert!(input.dt().is_float());
    }
}

impl Softmax {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError> {
        let arr = Array::<P>::default();
        builder.register_storage("X", BindingMode::ReadWrite, arr);
        builder.register_uniform();
        Ok(())
    }

    fn build_softmax<P: WgslPrimitive>(
        &self,
        inplace: bool,
        _: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError>
    where
        P::T: num_traits::Float,
    {
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
        kernel_builder.write_metadata::<SoftmaxMeta>();

        let dt = P::T::DT;
        let accessor = P::render_type();

        let BLOCK_SIZE = workgroup_size.x.render();
        let minFloat = P::T::NEG_INF;
        let workgroup_size_x = workgroup_size.x;

        kernel_builder.write_global(wgsl! {
            var<workgroup> smem: array<'accessor, 'workgroup_size_x>;
            var<workgroup> maximum: 'dt;
            var<workgroup> sum: 'dt;
        });

        kernel_builder.write_global(wgsl! {
            fn block_sum(index: u32, stride: u32) {
                if index < stride {
                    smem[index] += smem[index + stride];
                }
                workgroupBarrier();
            }

            fn block_max(index: u32, stride: u32) {
                if index < stride {
                    smem[index] = max(smem[index], smem[index + stride]);
                }
                workgroupBarrier();
            }
        });

        let reduce_var = match P::W {
            1 => "metadata.N",
            2 => "metadata.ND2",
            4 => "metadata.ND4",
            _ => {
                return Err(OperationError::CompileError(
                    "Invalid dimension".to_string(),
                ))?
            }
        };

        let offsets = wgsl! {
            let batch_stride = workgroup_id.y * metadata.M * 'reduce_var;
            let row_start = batch_stride + workgroup_id.x * 'reduce_var;
            let index = local_invocation_id.x;
        };
        kernel_builder.write_main(offsets);

        kernel_builder.write_main(wgsl! {
            smem[index] = 'accessor('minFloat);
            for (var i: u32 = index; i < 'reduce_var; i += 'BLOCK_SIZE) {
                smem[index] = max(smem[index], X[row_start + i]);
            }
            workgroupBarrier();
        });

        let steps = (workgroup_size.x - 1).ilog2();
        for i in (0..=steps).rev().map(|x| 2u32.pow(x)) {
            let v = i.render();
            kernel_builder.write_main(wgsl! { block_max(index, 'v); });
        }

        let finalize_max = match P::W {
            1 => wgsl! { maximum = smem[0]; },
            2 => wgsl! { maximum = max(smem[0].x, smem[0].y); },
            4 => wgsl! { maximum = max(smem[0].x, max(smem[0].y, max(smem[0].z, smem[0].w))); },
            _ => unreachable!(),
        };
        kernel_builder.write_main(wgsl! {
            if index == 0 {
                'finalize_max
            }
            workgroupBarrier();
        });

        kernel_builder.write_main(wgsl! {
            smem[index] = 'accessor(0.);
            for (var i: u32 = index; i < 'reduce_var; i += 'BLOCK_SIZE) {
                smem[index] += exp(X[row_start + i] - maximum);
            }
            workgroupBarrier();
        });

        for i in (0..=steps).rev().map(|x| 2u32.pow(x)) {
            let v = i.render();
            kernel_builder.write_main(wgsl! { block_sum(index, 'v); });
        }

        let finalize_sum = match P::W {
            1 => wgsl! { sum = smem[0]; },
            2 => wgsl! { sum = dot(smem[0], 'accessor(1.0, 1.0)); },
            4 => wgsl! { sum = dot(smem[0], 'accessor(1.0, 1.0, 1.0, 1.0)); },
            _ => unreachable!(),
        };
        kernel_builder.write_main(wgsl! {
            if index == 0 {
                'finalize_sum
            }
            workgroupBarrier();
        });

        let finalize = wgsl! {
            for(var i: u32 = index; i < 'reduce_var; i += 'BLOCK_SIZE) {
                var val = X[row_start + i];
                X[row_start + i] = exp(val - maximum) / sum;
            }
        };
        kernel_builder.write_main(finalize);
        Ok(kernel_builder.build()?)
    }
}

impl Operation for Softmax {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        Ok(self.input.storage_view().clone())
    }
}

impl MetaOperation for Softmax {
    fn kernel_name(&self) -> String {
        "softmax".to_string()
    }

    fn kernel_key(&self, _: bool, dst: &Tensor) -> KernelKey {
        KernelKey::new(format!("softmax_{}", self.kernel_element(dst).as_str()))
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input]
    }

    fn supports_inplace(&self) -> bool {
        true
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        let input = &self.input;
        let N = input.shape()[self.dim] as u32;
        if N % 4 == 0 {
            KernelElement::Vec4
        } else if N % 2 == 0 {
            KernelElement::Vec2
        } else {
            KernelElement::Scalar
        }
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = self.kernel_element(dst);
        match (self.input.dt(), &kernel_element) {
            (DType::F32, KernelElement::Scalar) => {
                self.build_softmax::<Scalar<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec2) => {
                self.build_softmax::<Vec2<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec4) => {
                self.build_softmax::<Vec4<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Scalar) => {
                self.build_softmax::<Scalar<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec2) => {
                self.build_softmax::<Vec2<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec4) => {
                self.build_softmax::<Vec4<f16>>(inplace, dst, workgroup_size)
            }
            _ => Err(OperationError::CompileError(format!(
                "Unsupported dtype {:?} or kernel element {:?}",
                self.input.dt(),
                kernel_element
            ))),
        }
    }

    fn calculate_dispatch(&self, _dst: &Tensor) -> Result<WorkgroupCount, OperationError> {
        let input = &self.input;
        let stacks = input.shape().slice(0..self.dim - 1).numel();
        let M = input.shape()[self.dim - 1] as u32;
        Ok(wgc![M as _, stacks as _, 1])
    }

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if !inplace {
            panic!("Only inplace softmax is supported");
        }
        Ok(BindGroupLayoutDescriptor::unary_inplace())
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        _: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        let input = &self.input;
        let M = input.shape()[self.dim - 1] as u32;
        let N = input.shape()[self.dim] as u32;
        let ND2 = N / 2;
        let ND4 = N / 4;
        let meta = SoftmaxMeta { M, N, ND2, ND4 };
        Ok(uniform.write(&meta)?)
    }

    fn compile(
        &self,
        dst: &Tensor,
        uniform: &mut CpuUniform,
        device: &WgpuDevice,
        can_inplace: bool,
    ) -> Result<CompiledOp, OperationError> {
        let kernel_element = self.kernel_element(dst);
        let offset = self.write_metadata(uniform, dst, &kernel_element)? as usize;

        let workgroup_count = self.calculate_dispatch(dst)?;

        let storage_layout = device
            .get_or_create_bind_group_layout(&self.storage_bind_group_layout(can_inplace)?)?;
        let uniform_layout =
            device.get_or_create_bind_group_layout(&BindGroupLayoutDescriptor::uniform())?;
        let pipeline_layout = device.get_or_create_pipeline_layout(&PipelineLayoutDescriptor {
            entries: rvec![storage_layout, uniform_layout],
        })?;

        let kernel_key = self.kernel_key(can_inplace, dst); //TODO: needs DTYPES

        let source_descriptor = KernelSourceDesc {
            key: kernel_key.clone(),
        };

        let compute_module = device.get_or_create_compute_module(
            &source_descriptor,
            self,
            can_inplace,
            dst,
            &wgs![128, 1, 1],
        );

        let pipeline_descriptor = ComputePipelineDescriptor {
            pipeline_layout,
            kernel_key: kernel_key.clone(),
            compute_module: Some(compute_module),
        };
        let pipeline_handle = device.get_or_create_compute_pipeline(&pipeline_descriptor)?;

        //TODO: Not sure i like this call here
        let storage_bind_groups = CompiledOp::create_storage_bind_groups(
            &self.srcs(),
            dst,
            rvec![storage_layout],
            device,
            can_inplace,
        )?;

        Ok(CompiledOp::new(
            pipeline_handle,
            workgroup_count,
            storage_bind_groups,
            offset as _,
            kernel_key,
        ))
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::test_util::run_py_prg;
    use crate::{shape, wgs, Device, DeviceRequest, MetaOperation, Softmax, Tensor};
    use half::f16;

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    fn ground_truth(a: &Tensor) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
import torch.nn.functional as F
def softmax(a):
    return F.softmax(torch.from_numpy(a), dim=-1).numpy()
"#;
        run_py_prg(prg.to_string(), &[a], &[])
    }

    fn run_softmax_trial(problem: SoftmaxProblem) {
        let device = GPU_DEVICE.with(|d| d.clone());
        let SoftmaxProblem { B, M, N } = problem;
        let a = Tensor::randn::<f32>(shape![B, M, N], Device::CPU);
        let ground = ground_truth(&a).unwrap();

        let a_gpu = a.to(&device).unwrap();
        let b = a_gpu.softmax(2).unwrap().resolve().unwrap();

        let ours = b.to(&Device::CPU).unwrap();
        println!("ours = {:?}", ours);
        println!("ground = {:?}", ground);
        ground.all_close(&ours, 1e-6, 1e-6).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct SoftmaxProblem {
        #[strategy(1..=3usize)]
        B: usize,
        #[strategy(1..=256usize)]
        M: usize,
        #[strategy(1..=256usize)]
        N: usize,
    }

    #[proptest(cases = 8)]
    fn test_softmax(prob: SoftmaxProblem) {
        let SoftmaxProblem { B, M, N } = prob;
        println!("B = {}, M = {}, N = {}", B, M, N);
        run_softmax_trial(prob);
    }

    #[test]
    fn dbg_softmax() {
        let problem = SoftmaxProblem { B: 1, M: 2, N: 128 };
        run_softmax_trial(problem);
    }

    #[test]
    fn test_render_softmax() {
        let device = GPU_DEVICE.with(|d| d.clone());
        let a = Tensor::randn::<f16>(shape![1, 2, 128], device.clone());
        let dst = Tensor::zeros::<f16>(&shape![1, 2, 128], &device);
        let op = Softmax::new(a, 2);
        let wgs = wgs![128, 1, 1];
        let _ = op.build_kernel(true, &dst, &wgs);
    }
}
