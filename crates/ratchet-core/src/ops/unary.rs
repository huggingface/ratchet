use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use ratchet_macros::WgslMetadata;

use crate::{
    gpu::{dtype::WgslDType, BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, Array, BindingMode, BuiltIn, DType, KernelElement, KernelKey, KernelSource,
    MetaOperation, OpGuards, OpMetadata, Operation, OperationError, RVec, Scalar, StorageView,
    Tensor, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize,
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
    pub fn kernel_name(&self) -> &'static str {
        match self {
            UnaryOp::Gelu => "gelu",
            UnaryOp::Tanh => "tanh",
            UnaryOp::Exp => "exp",
            UnaryOp::Log => "log",
            UnaryOp::Sin => "sin",
            UnaryOp::Cos => "cos",
            UnaryOp::Abs => "abs",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Relu => "relu",
            UnaryOp::Floor => "floor",
            UnaryOp::Ceil => "ceil",
            UnaryOp::Neg => "neg",
            UnaryOp::Silu => "silu",
            UnaryOp::Sigmoid => "sigmoid",
        }
    }
}

#[derive(new, Debug, Clone)]
pub struct Unary {
    input: Tensor,
    op: UnaryOp,
}

impl Unary {
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    const SCALED_SQRT_2_OVER_PI: f32 = 0.035677408136300125;

    pub fn op(&self) -> &UnaryOp {
        &self.op
    }

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
            fn safe_tanh(val: 'accessor) -> 'accessor {
                return select(tanh(x), sign(x), abs(x) >= 10f);
            }
        }
    }

    fn render_sigmoid<P: WgslPrimitive>() -> String {
        let accessor = P::render_type();

        wgsl! {
            fn sigmoid(val: 'accessor) -> 'accessor {
                return select(1.0f / (1.0f + exp(-val)), exp(val) / (1.0f + exp(val)), val >= 'accessor(0.));
            }
        }
    }

    fn build_unary<P: WgslPrimitive>(
        &self,
        inplace: bool,
        _: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let device = self.input.device().try_gpu().unwrap();
        let mut kernel_builder = WgslKernelBuilder::new(
            workgroup_size.clone(),
            rvec![BuiltIn::WorkgroupId, BuiltIn::LocalInvocationIndex,],
            device.compute_features().clone(),
        );

        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        kernel_builder.write_metadata::<UnaryMeta>();

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
                kernel_builder.write_global(wgsl! {
                    fn silu(val: 'accessor) -> 'accessor {
                        return val * sigmoid(val);
                    }
                });
            }
            UnaryOp::Relu => {
                kernel_builder.write_global(wgsl! {
                    fn relu(val: 'accessor) -> 'accessor {
                        return max(val, 'accessor(0.0));
                    }
                });
            }
            _ => todo!(),
        };

        let n = P::W;

        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let index = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (index >= metadata.numel / 'n) {
                return;
            }
        });

        let func = self.op.kernel_name();
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

#[derive(Debug, ShaderType, WgslMetadata)]
pub struct UnaryMeta {
    numel: u32,
}

impl OpGuards for Unary {
    fn check_shapes(&self) {}

    fn check_dtypes(&self) {
        let a = &self.input;
        assert!(matches!(a.dt(), crate::DType::F32));
    }
}

impl Operation for Unary {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        Ok(self.input.storage_view().clone())
    }
}

impl MetaOperation for Unary {
    fn kernel_name(&self) -> String {
        self.op.kernel_name().to_string()
    }

    fn kernel_key(&self, inplace: bool, dst: &Tensor) -> KernelKey {
        let kn = self.kernel_name();
        let ke = self.kernel_element(dst).as_str();
        let key = if inplace {
            format!("{}_inplace_{}", kn, ke)
        } else {
            format!("{}_{}", kn, ke)
        };
        KernelKey::new(key)
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input]
    }

    fn supports_inplace(&self) -> bool {
        true
    }

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

    fn calculate_dispatch(&self, _dst: &Tensor) -> Result<WorkgroupCount, OperationError> {
        let numel = self.input.shape().numel();
        let x_groups = WorkgroupCount::div_ceil(numel as _, 64);
        let (x_groups, y_groups) = if x_groups > WorkgroupCount::MAX_WGS_PER_DIM {
            let y_groups = WorkgroupCount::div_ceil(x_groups, WorkgroupCount::MAX_WGS_PER_DIM);
            (WorkgroupCount::MAX_WGS_PER_DIM, y_groups)
        } else {
            (x_groups, 1)
        };
        Ok(wgc![x_groups as _, y_groups as _, 1])
    }

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

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        _: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        let a = &self.input;
        let numel = a.shape().numel() as u32;
        let meta = UnaryMeta { numel };
        Ok(uniform.write(&meta)?)
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
                self.build_unary::<Scalar<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec2) => {
                self.build_unary::<Vec2<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec4) => {
                self.build_unary::<Vec4<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Scalar) => {
                self.build_unary::<Scalar<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec2) => {
                self.build_unary::<Vec2<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec4) => {
                self.build_unary::<Vec4<f16>>(inplace, dst, workgroup_size)
            }
            _ => Err(OperationError::CompileError(format!(
                "Unsupported dtype {:?} or kernel element {:?}",
                self.input.dt(),
                kernel_element
            ))),
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::{
        shape, test_util::run_py_prg, wgs, Device, DeviceRequest, MetaOperation, Tensor, Unary,
        UnaryOp,
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

        run_py_prg(prg.to_string(), &[a], &[])
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

    #[test]
    fn test_render_unary() {
        let device = GPU_DEVICE.with(|d| d.clone());
        let input = Tensor::randn::<f32>(shape![1, 128], device.clone());
        let op = Unary::new(input, UnaryOp::Gelu);
        let dst = Tensor::randn::<f32>(shape![1, 128], device);
        let kernel = op.build_kernel(true, &dst, &wgs![8, 8, 1]).unwrap();
        println!("{}", kernel);
    }
}
