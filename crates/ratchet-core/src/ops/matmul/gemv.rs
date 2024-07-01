use encase::ShaderType;
use half::f16;
use ratchet_macros::WgslMetadata;

use crate::{
    gpu::dtype::WgslDType, rvec, Array, BindingMode, BuiltIn, CpuUniform, DType, GEMMMeta,
    InvariantError, KernelElement, KernelSource, Matmul, MatmulSpec, OperationError, Scalar,
    Tensor, Vec4, WgslFragment, WgslKernelBuilder, WgslPrimitive, WorkgroupSize,
};
use glam::IVec3;
use inline_wgsl::wgsl;
use num_traits::Zero;

#[derive(Debug, Clone)]
pub struct GEMV(MatmulInner);

impl WorkgroupGEMVMeta {
    pub(crate) fn write_metadata(
        uniform: &mut CpuUniform,
        spec: &MatmulSpec,
    ) -> Result<u64, OperationError> {
        GEMMMeta::write_metadata(uniform, spec)
    }
}

impl GEMV {
    fn register_bindings_workgroup<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        let (A, _, bias) = (&self.lhs, &self.rhs, &self.bias);

        if A.dt().is_float() {
            let float_arr = Array::<P>::default();
            builder.register_storage("A", BindingMode::ReadOnly, float_arr);
            builder.register_storage("X", BindingMode::ReadOnly, float_arr);
            if bias.is_some() {
                builder.register_storage("bias", BindingMode::ReadOnly, float_arr);
            }
            builder.register_storage("result", BindingMode::ReadWrite, float_arr);
        } else if A.dt().is_quantized() {
            let scalar = Array::<Scalar<P::T>>::default();
            builder.register_storage("A", BindingMode::ReadOnly, Array::<Scalar<u32>>::default());
            builder.register_storage("scale", BindingMode::ReadOnly, scalar);
            builder.register_storage("X", BindingMode::ReadOnly, Array::<Vec4<P::T>>::default());
            if bias.is_some() {
                builder.register_storage("bias", BindingMode::ReadOnly, scalar);
            }
            builder.register_storage("result", BindingMode::ReadWrite, scalar);
        } else {
            return Err(InvariantError::UnsupportedDType(A.dt()).into());
        }

        builder.register_uniform();
        Ok(())
    }

    fn register_bindings_subgroup<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        let A = &self.lhs;
        let bias = &self.bias;
        let float_arr = Array::<P>::default();

        if A.dt().is_float() {
            builder.register_storage("mat", BindingMode::ReadOnly, float_arr);
            builder.register_storage("inVec", BindingMode::ReadOnly, float_arr);
            if bias.is_some() {
                builder.register_storage("bias", BindingMode::ReadOnly, float_arr);
            }
            builder.register_storage("outVec", BindingMode::ReadWrite, float_arr);
        } else if A.dt().is_quantized() {
            let scalar = Array::<Scalar<P::T>>::default();
            let u32_arr = Array::<Scalar<u32>>::default();
            builder.register_storage("mat", BindingMode::ReadOnly, u32_arr);
            builder.register_storage("scale", BindingMode::ReadOnly, scalar);
            builder.register_storage("inVec", BindingMode::ReadOnly, scalar);
            if bias.is_some() {
                builder.register_storage("bias", BindingMode::ReadOnly, scalar);
            }
            builder.register_storage("outVec", BindingMode::ReadWrite, scalar);
        } else {
            return Err(InvariantError::UnsupportedDType(A.dt()).into());
        }

        builder.register_uniform();
        Ok(())
    }

    pub fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
        spec: MatmulSpec,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = KernelElement::Scalar;
        match (self.lhs.dt(), kernel_element) {
            (DType::F32, KernelElement::Scalar) => {
                self.render_gemv::<Scalar<f32>>(inplace, dst, workgroup_size, spec)
            }
            (DType::F16, KernelElement::Scalar) => {
                self.render_gemv::<Scalar<f16>>(inplace, dst, workgroup_size, spec)
            }
            (DType::Q8_0F(_), _) => {
                self.render_gemv::<Vec4<f32>>(inplace, dst, workgroup_size, spec)
            }
            (DType::Q8_0H(_), _) => {
                self.render_gemv::<Vec4<f16>>(inplace, dst, workgroup_size, spec)
            }
            _ => panic!("Unsupported dtype"),
        }
    }

    fn render_gemv<P: WgslPrimitive>(
        &self,
        inplace: bool,
        _: &Tensor,
        workgroup_size: &WorkgroupSize,
        spec: MatmulSpec,
    ) -> Result<KernelSource, OperationError> {
        let device = self.lhs.device().try_gpu().unwrap();
        if device.compute_features().SUBGROUP {
            self.subgroup_gemv::<P>(inplace, &self.lhs, workgroup_size, spec)
        } else {
            self.workgroup_gemv::<P>(inplace, &self.lhs, workgroup_size, spec)
        }
    }
}
