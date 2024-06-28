use num_traits::One;
use std::borrow::Cow;

use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use ratchet_macros::WgslMetadata;

use crate::{
    gpu::{dtype::WgslDType, BindGroupLayoutDescriptor, CpuUniform},
    rvec, Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel, KernelElement,
    KernelRenderable, KernelSource, MetaOperation, OpGuards, Operation, OperationError, RVec,
    Scalar, StorageView, Tensor, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize,
    Workload,
};

#[cfg(test)]
use test_strategy::Arbitrary;

#[cfg_attr(test, derive(Arbitrary))]
#[derive(Debug, Clone)]
pub enum UnaryOp {
    Gelu,
    Tanh,
    Exp,
    Log,
    Sin,
    Cos,
    Abs,
    Sqrt,
    Relu,
    Floor,
    Ceil,
    Neg,
    Silu,
    Sigmoid,
}

impl UnaryOp {
    pub fn kernel_name(&self) -> Cow<'static, str> {
        match self {
            UnaryOp::Gelu => "gelu".into(),
            UnaryOp::Tanh => "tanh".into(),
            UnaryOp::Exp => "exp".into(),
            UnaryOp::Log => "log".into(),
            UnaryOp::Sin => "sin".into(),
            UnaryOp::Cos => "cos".into(),
            UnaryOp::Abs => "abs".into(),
            UnaryOp::Sqrt => "sqrt".into(),
            UnaryOp::Relu => "relu".into(),
            UnaryOp::Floor => "floor".into(),
            UnaryOp::Ceil => "ceil".into(),
            UnaryOp::Neg => "neg".into(),
            UnaryOp::Silu => "silu".into(),
            UnaryOp::Sigmoid => "sigmoid".into(),
        }
    }

    pub fn kernel_operation(&self) -> Cow<'static, str> {
        match self {
            UnaryOp::Tanh => "safe_tanh".into(),
            UnaryOp::Neg => "-".into(),
            _ => self.kernel_name(),
        }
    }
}

#[derive(new, Debug, Clone)]
pub struct Unary {
    input: Tensor,
    op: UnaryOp,
}

impl KernelRenderable for Unary {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError> {
        if inplace {
            builder.register_storage("X", BindingMode::ReadWrite, Array::<P>::default());
        } else {
            builder.register_storage("X", BindingMode::ReadOnly, Array::<P>::default());
            builder.register_storage("Y", BindingMode::ReadWrite, Array::<P>::default());
        }
        builder.register_uniform();
        Ok(())
    }

    fn render<P: WgslPrimitive>(
        &self,
        inplace: bool,
        _: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let device = self.input.device().try_gpu().unwrap();
        let mut kernel_builder = WgslKernelBuilder::new(
            workgroup_size.clone(),
            rvec![
                BuiltIn::WorkgroupId,
                BuiltIn::LocalInvocationIndex,
                BuiltIn::NumWorkgroups
            ],
            device.compute_features().clone(),
        );

        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        kernel_builder.render_metadata::<UnaryMeta>();

        let accessor = P::render_type();
        //Write global functions
        match self.op {
            UnaryOp::Gelu => {
                kernel_builder.write_global(Unary::render_tanh::<P>());
                kernel_builder.write_global(Unary::render_gelu::<P>());
            }
            UnaryOp::Tanh => {
                kernel_builder.write_global(Unary::render_tanh::<P>());
            }
            UnaryOp::Sigmoid => {
                kernel_builder.write_global(Unary::render_sigmoid::<P>());
            }
            UnaryOp::Silu => {
                kernel_builder.write_global(Unary::render_sigmoid::<P>());
                kernel_builder.write_global(Unary::render_silu::<P>());
            }
            UnaryOp::Relu => {
                kernel_builder.write_global(Unary::render_relu::<P>());
            }
            _ => {}
        };

        let n = P::W;

        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let index = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (index >= metadata.numel / 'n) {
                return;
            }
        });

        let func = self.op.kernel_operation();
        if inplace {
            kernel_builder.write_main(wgsl! {
                let val = X[index];
                X[index] = 'func(val);
            });
        } else {
            kernel_builder.write_main(wgsl! {
                Y[index] = 'func(X[index]);
            });
        }

        Ok(kernel_builder.build()?)
    }
}

impl Unary {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const SCALED_SQRT_2_OVER_PI: f32 = 0.035_677_407;

    pub fn op(&self) -> &UnaryOp {
        &self.op
    }

    fn render_gelu<P: WgslPrimitive>() -> String {
        let accessor = P::render_type();
        let SQRT_2_OVER_PI = Self::SQRT_2_OVER_PI;
        let SCALED_SQRT_2_OVER_PI = Self::SCALED_SQRT_2_OVER_PI;

        wgsl! {
            fn gelu(val: 'accessor) -> 'accessor {
                let cdf = 'accessor(0.5) + 'accessor(0.5) * safe_tanh(val * ('accessor('SCALED_SQRT_2_OVER_PI)
                        * (val * val) + 'accessor('SQRT_2_OVER_PI)));
                return val * cdf;
            }
        }
    }

    fn render_tanh<P: WgslPrimitive>() -> String {
        let accessor = P::render_type();

        wgsl! {
            fn safe_tanh(x: 'accessor) -> 'accessor {
                return select(tanh(x), sign(x), abs(x) >= 'accessor(10.));
            }
        }
    }

    fn render_relu<P: WgslPrimitive>() -> String {
        let accessor = P::render_type();

        wgsl! {
            fn relu(val: 'accessor) -> 'accessor {
                return max(val, 'accessor(0.0));
            }
        }
    }

    fn render_silu<P: WgslPrimitive>() -> String {
        let accessor = P::render_type();

        wgsl! {
            fn silu(val: 'accessor) -> 'accessor {
                return val * sigmoid(val);
            }
        }
    }

    fn render_sigmoid<P: WgslPrimitive>() -> String {
        let accessor = P::render_type();
        let one = P::T::one().render();

        wgsl! {
            fn sigmoid(val: 'accessor) -> 'accessor {
                return select('one / ('one + exp(-val)), exp(val) / ('one + exp(val)), val >= 'accessor(0.));
            }
        }
    }
}

#[derive(Debug, ShaderType, WgslMetadata)]
pub struct UnaryMeta {
    numel: u32,
}

impl OpGuards for Unary {
    fn check_shapes(&self) {}

    fn check_dtypes(&self) {}
}

impl Operation for Unary {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        Ok(self.input.storage_view().clone())
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input]
    }

    fn supports_inplace(&self) -> bool {
        true
    }
}

impl GPUOperation for Unary {
    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if inplace {
            Ok(BindGroupLayoutDescriptor::unary_inplace())
        } else {
            Ok(BindGroupLayoutDescriptor::unary())
        }
    }
}

impl Kernel for Unary {
    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        let a_rank = &self.input.shape().rank();
        let N = &self.input.shape()[a_rank - 1];

        if N % 4 == 0 {
            KernelElement::Vec4
        } else if N % 2 == 0 {
            KernelElement::Vec2
        } else {
            KernelElement::Scalar
        }
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError> {
        Ok(Workload::std(dst.shape().numel(), self.kernel_element(dst)))
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        _: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        let a = &self.input;
        let numel = a.shape().numel() as u32;
        Ok(uniform.write(&UnaryMeta { numel })?)
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
            _ => Err(OperationError::CompileError(format!(
                "Unsupported dtype {:?} or kernel element {:?}",
                self.input.dt(),
                kernel_element
            ))),
        }
    }
}

impl MetaOperation for Unary {}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::{
        shape, test_util::run_py_prg, Device, DeviceRequest, MetaOperation, Tensor, UnaryOp,
    };

    #[derive(Arbitrary, Debug)]
    struct UnaryProblem {
        op: UnaryOp,
        #[strategy(1..=2usize)]
        B: usize,
        #[strategy(1..=128usize)]
        M: usize,
        #[strategy(1..=128usize)]
        N: usize,
    }

    fn ground_truth(a: &Tensor, op: &UnaryOp, args: &str) -> anyhow::Result<Tensor> {
        let kn = op.kernel_name();
        let func_prg = format!(
            r#"
import torch
import torch.nn.functional as F
def {}(a):
    return F.{}(torch.from_numpy(a), {}).numpy()
"#,
            kn, kn, args,
        );

        let imp_prg = format!(
            r#"
import torch
def {}(a):
    return torch.{}(torch.from_numpy(a), {}).numpy()
"#,
            kn, kn, args,
        );

        let prg = match op {
            UnaryOp::Gelu | UnaryOp::Silu | UnaryOp::Sigmoid => func_prg,
            _ => imp_prg,
        };

        run_py_prg(prg.to_string(), &[a], &[], a.dt())
    }

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    fn run_unary_trial(prob: UnaryProblem) -> anyhow::Result<()> {
        let device = GPU_DEVICE.with(|d| d.clone());
        let UnaryProblem { op, B, M, N } = prob;
        println!("op: {:?}, B: {}, M: {}, N: {}", op, B, M, N);
        let a = Tensor::randn::<f32>(shape![B, M], Device::CPU);

        let args = match op {
            UnaryOp::Gelu => "approximate=\"tanh\"",
            _ => "",
        };
        let ground = ground_truth(&a, &op, args)?;

        let a_gpu = a.to(&device)?;
        let c_gpu = match op {
            UnaryOp::Gelu => a_gpu.gelu()?,
            UnaryOp::Tanh => a_gpu.tanh()?,
            UnaryOp::Exp => a_gpu.exp()?,
            UnaryOp::Log => a_gpu.log()?,
            UnaryOp::Sin => a_gpu.sin()?,
            UnaryOp::Cos => a_gpu.cos()?,
            UnaryOp::Abs => a_gpu.abs()?,
            UnaryOp::Sqrt => a_gpu.sqrt()?,
            UnaryOp::Relu => a_gpu.relu()?,
            UnaryOp::Floor => a_gpu.floor()?,
            UnaryOp::Ceil => a_gpu.ceil()?,
            UnaryOp::Neg => a_gpu.neg()?,
            UnaryOp::Silu => a_gpu.silu()?,
            UnaryOp::Sigmoid => a_gpu.sigmoid()?,
        }
        .resolve()?;

        let (atol, rtol) = match op {
            UnaryOp::Gelu | UnaryOp::Tanh => (5e-2, 5e-2),
            _ => (1e-4, 1e-4),
        };

        let d_gpu = c_gpu.to(&Device::CPU)?;
        ground.all_close(&d_gpu, atol, rtol)?;
        Ok(())
    }

    #[proptest(cases = 256)]
    fn test_unary(prob: UnaryProblem) {
        run_unary_trial(prob).unwrap();
    }
}
