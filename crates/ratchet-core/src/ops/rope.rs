use derive_new::new;
use encase::ShaderType;
use half::f16;
use ratchet_macros::WgslMetadata;

use crate::gpu::dtype::WgslDType;
use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, wgs, Array, BindingMode, BuiltIn, DType, KernelElement, KernelSource, MetaOperation,
    OpGuards, Operation, OperationError, RVec, Scalar, StorageView, Strides, Tensor, Vec2, Vec4,
    WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
};
use inline_wgsl::wgsl;

#[derive(new, Debug, Clone)]
pub struct RoPE {
    input: Tensor,
    dim: usize,
    base: f32,
    offset: usize,
}

impl RoPE {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError> {
        if !inplace {
            panic!("Only inplace rope is supported");
        }
        let arr = Array::<P>::default();
        builder.register_storage("in", BindingMode::ReadWrite, arr);
        builder.register_uniform();
        Ok(())
    }

    fn build_rope<P: WgslPrimitive>(
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
                BuiltIn::NumWorkgroups,
            ],
            device.compute_features().clone(),
        );
        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        kernel_builder.write_metadata::<RoPEMeta>();

        let dt = P::T::DT;

        kernel_builder.write_main(wgsl! {
            if(global_invocation_id.y >= metadata.seq_len) {
              return;
            }

            let grid = vec3<u32>(num_workgroups.x * 8u, num_workgroups.y * 8u, num_workgroups.z * 1u);

            let out_index_1 = dot(global_invocation_id, vec3<u32>(metadata.out_strides[2], metadata.out_strides[1], metadata.out_strides[0]));
            let out_index_2 = out_index_1 + grid.x * metadata.out_strides[2];

            let in_index_1 = dot(global_invocation_id, vec3<u32>(metadata.in_strides[2], metadata.in_strides[1], metadata.in_strides[0]));
            let in_index_2 = in_index_1 + grid.x * metadata.in_strides[2];

            let L = metadata.scale * f32(global_invocation_id.y + metadata.offset);
            let d = f32(global_invocation_id.x) / f32(grid.x);

            let theta = L * exp2(-d * metadata.base);
            let costheta = 'dt(cos(theta));
            let sintheta = 'dt(sin(theta));

            let x1 = in[in_index_1];
            let x2 = in[in_index_2];

            let rx1 = x1 * costheta - x2 * sintheta;
            let rx2 = x1 * sintheta + x2 * costheta;

            in[out_index_1] = rx1;
            in[out_index_2] = rx2;
        });

        Ok(kernel_builder.build()?)
    }
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct RoPEMeta {
    in_strides: glam::UVec3,
    out_strides: glam::UVec3,
    seq_len: u32,
    offset: u32,
    base: f32,
    scale: f32,
}

impl OpGuards for RoPE {
    fn check_shapes(&self) {
        let input = &self.input;
        //TODO: overly restrictive
        assert!(input.rank() == 4);
        assert!(input.shape()[3] >= self.dim);
        assert!(self.dim % 8 == 0);
    }

    fn check_dtypes(&self) {
        let input = &self.input;
        assert!(input.dt().is_float());
    }
}

impl Operation for RoPE {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        Ok(self.input.storage_view().clone())
    }
}

impl MetaOperation for RoPE {
    fn kernel_name(&self) -> String {
        "rope".to_string()
    }

    fn supports_inplace(&self) -> bool {
        true
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input]
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, _dst: &Tensor) -> Result<Workload, OperationError> {
        const WGSX: usize = 8;
        const WGSY: usize = 8;
        const WGSZ: usize = 1;
        let workgroup_size = wgs![WGSX as _, WGSY as _, WGSZ as _];

        let input = &self.input;
        let [_, _, SL, HD]: [usize; 4] = input.shape().try_into()?;
        let mat_size = SL * HD;

        let total_x = self.dim / 2; //solve pairs
        let total_y = SL;
        let total_z = input.shape().numel() / mat_size;

        let wgcx = WorkgroupCount::div_ceil(total_x, WGSX) as u32;
        let wgcy = WorkgroupCount::div_ceil(total_y, WGSY) as u32;
        let wgcz = WorkgroupCount::div_ceil(total_z, WGSZ) as u32;

        Ok(Workload {
            workgroup_count: wgc![wgcx, wgcy, wgcz],
            workgroup_size,
        })
    }

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if inplace {
            return Ok(BindGroupLayoutDescriptor::unary_inplace());
        }
        panic!("RoPE does not support out-of-place operation");
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        dst: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        let mut input_shape = self.input.shape().clone();
        let SL = input_shape[2];
        let mut out_shape = dst.shape().clone();
        input_shape.remove(0);
        out_shape.remove(0);
        let in_strides = Strides::from(&input_shape);
        let out_strides = Strides::from(&out_shape);
        let meta = RoPEMeta::new(
            (&in_strides).into(),
            (&out_strides).into(),
            SL as u32,
            self.offset as u32,
            self.base,
            1.0,
        );
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
                self.build_rope::<Scalar<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec2) => {
                self.build_rope::<Vec2<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec4) => {
                self.build_rope::<Vec4<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Scalar) => {
                self.build_rope::<Scalar<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec2) => {
                self.build_rope::<Vec2<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec4) => {
                self.build_rope::<Vec4<f16>>(inplace, dst, workgroup_size)
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

    use crate::test_util::run_py_prg;
    use crate::{shape, Device, DeviceRequest, Tensor};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    fn ground_truth(a: &Tensor, dim: usize, offset: usize) -> anyhow::Result<Tensor> {
        let prg = r#"
import mlx.core as mx
import mlx.nn as nn
import numpy as np

def mlx_rope(input, dim, offset):
    rope = nn.RoPE(dim)
    mx_input = mx.array(input)
    y = rope(mx_input, offset)
    mx.eval(y)
    return np.array(y)
"#;
        run_py_prg(prg.to_string(), &[a], &[&dim, &offset], a.dt())
    }

    fn run_rope_trial(problem: RoPEProblem) {
        let device = GPU_DEVICE.with(|d| d.clone());
        let RoPEProblem {
            BS,
            NH,
            SL,
            HD,
            dim,
            offset,
        } = problem;
        let a = Tensor::randn::<f32>(shape![BS, NH, SL, HD], Device::CPU);
        let ground = ground_truth(&a, dim, offset).unwrap();

        let a_gpu = a.to(&device).unwrap();
        let b = a_gpu.rope(dim, 10000.0, offset).unwrap().resolve().unwrap();

        let ours = b.to(&Device::CPU).unwrap();
        //println!("ours = \n{:#?}\n", ours.to_ndarray_view::<f32>());
        //println!("ground = \n{:#?}", ground.to_ndarray_view::<f32>());
        //Weak tolerance because of `ffast-math`
        ground.all_close(&ours, 1e-3, 1e-3).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct RoPEProblem {
        #[strategy(1..=2usize)]
        BS: usize,
        #[strategy(1..=64usize)]
        NH: usize,
        #[strategy(1..=256usize)]
        SL: usize,
        #[strategy(32..=128usize)]
        #[filter(#HD % 16 == 0)]
        HD: usize,
        #[strategy(32..=#HD)]
        #[filter(#dim % 32 == 0)]
        dim: usize,
        #[strategy(0..=#SL)]
        offset: usize,
    }

    #[proptest(cases = 16)]
    fn test_rope(prob: RoPEProblem) {
        let RoPEProblem {
            BS,
            NH,
            SL,
            HD,
            dim,
            offset,
        } = prob;
        println!(
            "BS = {}, NH = {}, SL = {}, HD = {}, rope_dim = {}, offset = {}",
            BS, NH, SL, HD, dim, offset
        );
        run_rope_trial(prob);
    }
}
