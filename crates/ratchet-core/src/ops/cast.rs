use anyhow::anyhow;
use derive_new::new;
use encase::ShaderType;
use half::{bf16, f16};
use inline_wgsl::wgsl;
use ratchet_macros::WgslMetadata;

use crate::{
    cpu_store_result, gpu::BindGroupLayoutDescriptor, rvec, Array, BindingMode, BuiltIn,
    CPUOperation, DType, GPUOperation, Kernel, KernelElement, KernelRenderable, KernelSource,
    OpGuards, Operation, OperationError, RVec, Scalar, StorageView, Strides, Tensor, Vec2, Vec4,
    WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
};

#[derive(new, Debug, Clone)]
pub struct Cast {
    input: Tensor,
    dst_dt: DType,
}

impl KernelRenderable for CastKernels {
    fn register_bindings<SP: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        builder.register_storage("X", BindingMode::ReadOnly, Array::<SP>::default());
        let CastKernels::Standard(inner) = self;
        //TODO: This is a bit of a hack, this match should be formalized
        let dst_accessor = match SP::W {
            1 => inner.dst_dt.as_wgsl().to_string(),
            2 | 4 => format!("vec{}<{}>", SP::W, inner.dst_dt.as_wgsl()),
            _ => unimplemented!(),
        };

        unsafe {
            builder.register_storage_raw(
                "Y",
                BindingMode::ReadWrite,
                format!("array<{}>", dst_accessor),
            )
        };
        builder.register_uniform();
        Ok(())
    }

    fn render<SP: WgslPrimitive>(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let device = dst.device().try_gpu()?;
        let mut kernel_builder = WgslKernelBuilder::new(
            workgroup_size.clone(),
            rvec![
                BuiltIn::WorkgroupId,
                BuiltIn::LocalInvocationIndex,
                BuiltIn::NumWorkgroups
            ],
            device.compute_features().clone(),
        );

        self.register_bindings::<SP>(&mut kernel_builder, inplace)?;
        kernel_builder.render_metadata(&self.metadata(dst, &self.kernel_element(dst))?);

        let n = SP::W;
        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let index = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (index >= metadata.numel / 'n) {
                return;
            }
        });

        let CastKernels::Standard(inner) = self;

        //Bit of a hack
        let dst_accessor = match n {
            1 => inner.dst_dt.as_wgsl().to_string(),
            2 | 4 => format!("vec{}<{}>", n, inner.dst_dt.as_wgsl()),
            _ => unimplemented!(),
        };

        kernel_builder.write_main(wgsl! {
            Y[index] = 'dst_accessor(X[index]);
        });

        Ok(kernel_builder.build()?)
    }
}

#[derive(Debug, ShaderType, WgslMetadata)]
pub struct CastMeta {
    numel: u32,
}

impl OpGuards for Cast {
    fn check_shapes(&self) {}

    fn check_dtypes(&self) {}
}

impl Operation for Cast {
    fn name(&self) -> &'static str {
        "Cast"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let shape = self.input.shape().clone();
        let strides = Strides::from(&shape);
        Ok(StorageView::new(shape, self.dst_dt, strides))
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input]
    }

    fn supports_inplace(&self) -> bool {
        //CANNOT BE DONE INPLACE
        false
    }
}

pub enum CastKernels {
    Standard(Cast),
}

impl GPUOperation for Cast {
    type KernelEnum = CastKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        CastKernels::Standard(self.clone())
    }
}

impl Kernel for CastKernels {
    type Metadata = CastMeta;

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if inplace {
            panic!("Cast cannot be done in place on GPU");
        }
        Ok(BindGroupLayoutDescriptor::unary())
    }

    fn kernel_name(&self) -> String {
        match self {
            CastKernels::Standard(_) => "cast".to_string(),
        }
    }

    fn metadata(&self, dst: &Tensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let numel = dst.shape().numel() as u32;
        Ok(CastMeta { numel })
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError> {
        Ok(Workload::std(dst.shape().numel(), self.kernel_element(dst)))
    }

    fn kernel_element(&self, dst: &Tensor) -> KernelElement {
        let numel = dst.shape().numel();
        if numel % 4 == 0 {
            KernelElement::Vec4
        } else if numel % 2 == 0 {
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
        let CastKernels::Standard(inner) = self;
        match (inner.input.dt(), &kernel_element) {
            (DType::F32, KernelElement::Scalar) => {
                self.render::<Scalar<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec2) => {
                self.render::<Vec2<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec4) => {
                self.render::<Vec4<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Scalar) => {
                self.render::<Scalar<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec2) => {
                self.render::<Vec2<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec4) => {
                self.render::<Vec4<f16>>(inplace, dst, workgroup_size)
            }
            _ => unimplemented!("Cannot cast {:?} -> {:?}", inner.input.dt(), inner.dst_dt),
        }
    }
}

use crate::{TensorDType, TensorError, Q8_0H};

#[inline]
fn apply_fn_helper<T: TensorDType, U: TensorDType>(src: &[T], dst: &mut [U], f: fn(T) -> U) {
    assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().copied().zip(dst.iter_mut()) {
        *d = f(s);
    }
}

#[inline]
fn apply_fn<T: TensorDType, U: TensorDType>(
    input: &Tensor,
    dst: &Tensor,
    f: fn(T) -> U,
) -> Result<(), OperationError> {
    let input = input.to_vec::<T>()?;
    let mut result = vec![U::zero(); dst.shape().numel()];
    apply_fn_helper(&input, &mut result, f);
    cpu_store_result(dst, &result);
    Ok(())
}

fn direct_cast<T: TensorDType, U: TensorDType>(
    input: &Tensor,
    dst: &Tensor,
) -> Result<(), OperationError> {
    let input = input.to_vec::<T>()?;
    let result =
        bytemuck::try_cast_slice::<T, U>(&input).map_err(|_| anyhow!("Failed direct cast"))?;
    cpu_store_result(dst, &result);
    Ok(())
}

impl CPUOperation for Cast {
    fn apply(&self, dst: Tensor) -> Result<Tensor, OperationError> {
        if self.input.dt() == self.dst_dt {
            return Ok(self.input.clone());
        }
        match (self.input.dt(), self.dst_dt) {
            // F32 ->
            (DType::F32, DType::F16) => apply_fn::<f32, f16>(&self.input, &dst, f16::from_f32)?,
            (DType::F32, DType::BF16) => apply_fn::<f32, bf16>(&self.input, &dst, bf16::from_f32)?,
            (DType::F32, DType::I32) => direct_cast::<f32, i32>(&self.input, &dst)?,
            (DType::F32, DType::U32) => direct_cast::<f32, u32>(&self.input, &dst)?,

            // F16 ->
            (DType::F16, DType::F32) => apply_fn::<f16, f32>(&self.input, &dst, f32::from)?,

            // BF16 ->
            (DType::BF16, DType::F32) => apply_fn::<bf16, f32>(&self.input, &dst, f32::from)?,

            // I32 ->
            (DType::I32, DType::F32) => direct_cast::<i32, f32>(&self.input, &dst)?,

            // U32 ->
            (DType::U32, DType::F32) => direct_cast::<u32, f32>(&self.input, &dst)?,

            _ => unimplemented!("Cannot cast {:?} -> {:?}", self.input.dt(), self.dst_dt),
        };

        Ok(dst)
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use half::f16;
    use test_strategy::{proptest, Arbitrary};

    use crate::{shape, test_util::run_py_prg, DType, Device, DeviceRequest, Tensor};

    #[derive(Arbitrary, Debug)]
    struct CastProblem {
        dst_dt: DType,
        #[strategy(1..=2usize)]
        B: usize,
        #[strategy(1..=128usize)]
        M: usize,
        #[strategy(1..=128usize)]
        N: usize,
    }

    fn ground_truth(input: &Tensor, dst_dt: DType) -> anyhow::Result<Tensor> {
        let prg = format!(
            r#"
import torch
def cast(a):
    return torch.from_numpy(a).to({}).numpy()
"#,
            dst_dt.as_torch()
        );

        run_py_prg(prg.to_string(), &[input], &[], dst_dt)
    }

    fn run_cast_trial(prob: CastProblem, device: Device) -> anyhow::Result<()> {
        let CastProblem { dst_dt, B, M, N } = prob;
        let input = Tensor::randn::<f32>(shape![B, M, N], Device::CPU);
        let ground = ground_truth(&input, dst_dt)?;

        let input = input.to(&device)?;
        let casted = input.cast(dst_dt)?.resolve()?;

        let casted = casted.to(&Device::CPU)?;
        match dst_dt {
            DType::F16 => {
                ground.all_close::<f16>(&casted, f16::from_f32(1e-4), f16::from_f32(1e-4))?
            }
            DType::F32 => ground.all_close::<f32>(&casted, 1e-4, 1e-4)?,
            _ => return Ok(()), //all_close doesn't support integers
        }
        Ok(())
    }

    #[proptest(cases = 256)]
    fn test_type_cast_gpu(prob: CastProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_cast_trial(prob, device).unwrap();
    }

    #[proptest(cases = 256)]
    fn test_type_cast_cpu(prob: CastProblem) {
        let device = Device::request_device(DeviceRequest::CPU).unwrap();
        run_cast_trial(prob, device).unwrap();
    }
}
