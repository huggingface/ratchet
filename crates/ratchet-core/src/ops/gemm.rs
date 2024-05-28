use encase::ShaderType;
use half::f16;
use ratchet_macros::WgslMetadata;

use crate::{
    gguf::GGUFDType, gpu::dtype::WgslDType, rvec, Array, BindingMode, BuiltIn, DType,
    InvariantError, KernelElement, KernelSource, OperationError, Scalar, Tensor, Vec2, Vec4,
    WgslKernelBuilder, WgslPrimitive, WorkgroupSize,
};
use glam::IVec3;
use inline_wgsl::wgsl;

#[derive(Debug, Clone)]
pub struct GEMM {
    lhs: Tensor,
    rhs: Tensor,
    bias: Option<Tensor>,
    trans_lhs: bool,
    trans_rhs: bool,
    trans_out: bool,
}

#[allow(clippy::too_many_arguments)]
#[derive(Debug, Clone, ShaderType, WgslMetadata)]
pub struct GEMMMeta {
    aShape: IVec3,
    aStrides: IVec3,
    bShape: IVec3,
    bStrides: IVec3,
    outShape: IVec3,
    outStrides: IVec3,
    dimAOuter: i32,
    dimBOuter: i32,
    dimInner: i32,
}

impl GEMM {
    fn write_indexing<P: WgslPrimitive>(&self, builder: &mut WgslKernelBuilder) {
        let accessor = P::render_type();
        let W = P::W;
        builder.write_global(wgsl! {
            fn getAIndexFromCoords3D(coords : vec3<i32>) -> i32 {
                return dot(coords, metadata.aStrides);
            }

            fn getBIndexFromCoords3D(coords : vec3<i32>) -> i32 {
                return dot(coords, metadata.bStrides);
            }

            fn getOutputIndexFromCoords(coords : vec3<i32>) -> i32 {
              return dot(coords, metadata.outStrides);
            }

            fn setOutputAtIndex(flatIndex : i32, value : 'accessor) {
                result[flatIndex] = 'accessor(value);
            }

            fn setOutputAtCoords(d0 : i32, d1 : i32, d2 : i32, value : 'accessor) {
                let flatIndex = getOutputIndexFromCoords(vec3<i32>(d0, d1, d2));
                setOutputAtIndex(flatIndex / 'W, value);
            }
        });
    }

    fn write_getters<P: WgslPrimitive>(
        &self,
        dst: &Tensor,
        builder: &mut WgslKernelBuilder,
    ) -> Result<(), OperationError> {
        let (A, _, _) = (&self.lhs, &self.rhs, &self.bias);
        let accessor = P::render_type();
        let W = P::W;

        let a_getters = match A.dt() {
            DType::F32 | DType::F16 => {
                wgsl! {
                    fn getA(d0 : i32, d1 : i32, d2 : i32) -> 'accessor {
                        return 'accessor(A[getAIndexFromCoords3D(vec3<i32>(d0,d1,d2)) / 'W]);
                    }
                }
            }
            DType::GGUF(GGUFDType::Q8_0(_)) => {
                wgsl! {
                    fn unpack4x8snorm_gguf(x: u32) -> vec4<f32> {
                        return unpack4x8snorm(x) * 127f;
                    }

                    fn getA(d0 : i32, d1 : i32, d2 : i32) -> vec4<f32> {
                        return unpack4x8snorm_gguf(A[getAIndexFromCoords3D(vec3<i32>(d0,d1,d2)) / 4]);
                    }

                    fn getAbsMax(d0 : i32, d1 : i32, d2 : i32) -> f32 {
                        let abs_index = getAIndexFromCoords3D(vec3<i32>(d0,d1,d2)) / 32;
                        return scale[abs_index];
                    }
                }
            }
            _ => return Err(InvariantError::UnsupportedDType(A.dt()).into()),
        };
        builder.write_global(a_getters);

        builder.write_global(wgsl! {
            fn getB(d0 : i32, d1 : i32, d2 : i32) -> 'accessor {
                return 'accessor(B[getBIndexFromCoords3D(vec3<i32>(d0,d1,d2)) / 'W]);
            }
        });

        Ok(())
    }

    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        let (A, _, bias) = (&self.lhs, &self.rhs, &self.bias);

        let float_arr = Array::<P>::default();

        let ro = BindingMode::ReadOnly;
        match A.dt() {
            DType::F32 | DType::F16 => {
                builder.register_storage("A", ro, float_arr);
                builder.register_storage("B", ro, float_arr);
            }
            DType::GGUF(GGUFDType::Q8_0(_)) => {
                builder.register_storage("A", ro, Array::<Scalar<u32>>::default());
                builder.register_storage("scale", ro, float_arr);
                builder.register_storage("B", ro, Array::<Vec4<P::T>>::default());
            }
            _ => return Err(InvariantError::UnsupportedDType(A.dt()).into()),
        }

        if bias.is_some() {
            builder.register_storage("bias", BindingMode::ReadOnly, float_arr);
        }
        builder.register_storage("result", BindingMode::ReadWrite, float_arr);
        builder.register_uniform();
        Ok(())
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = KernelElement::Scalar;
        match (self.lhs.dt(), kernel_element) {
            (DType::F32, KernelElement::Scalar) => {
                self.build_gemm::<Scalar<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec2) => {
                self.build_gemm::<Vec2<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec4) => {
                self.build_gemm::<Vec4<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Scalar) => {
                self.build_gemm::<Scalar<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec2) => {
                self.build_gemm::<Vec2<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec4) => {
                self.build_gemm::<Vec4<f16>>(inplace, dst, workgroup_size)
            }
            (DType::GGUF(g), _) => match g {
                crate::gguf::GGUFDType::Q8_0(_) => todo!(),
                _ => unimplemented!(),
            },
            _ => panic!("Unsupported dtype"),
        }
    }

    fn build_gemm<P: WgslPrimitive>(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let device = self.lhs.device().try_gpu().unwrap();
        let mut kernel_builder = WgslKernelBuilder::new(
            workgroup_size.clone(),
            rvec![
                BuiltIn::GlobalInvocationId,
                BuiltIn::LocalInvocationId,
                BuiltIn::WorkgroupId,
            ],
            device.compute_features().clone(),
        );
        self.register_bindings::<P>(&mut kernel_builder, inplace)
            .unwrap();
        kernel_builder.write_metadata::<GEMMMeta>();
        self.write_indexing::<P>(&mut kernel_builder);
        self.write_getters::<P>(dst, &mut kernel_builder)?;

        Ok(kernel_builder.build()?)
    }
}

#[cfg(test)]
mod tests {
    use crate::{shape, wgs, Device, DeviceRequest, Tensor, GEMM};
    use half::f16;

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    #[test]
    fn render_gemm() {
        let device = GPU_DEVICE.with(|d| d.clone());
        let lhs = Tensor::randn::<f16>(shape![128, 128], device.clone());
        let rhs = Tensor::randn::<f16>(shape![128, 128], device.clone());
        let bias = Tensor::randn::<f16>(shape![128], device.clone());
        let op = GEMM {
            lhs,
            rhs,
            bias: Some(bias),
            trans_lhs: false,
            trans_rhs: false,
            trans_out: false,
        };
        let dst = Tensor::zeros::<f16>(&shape![128, 128], &device);
        let kernel = op.build_kernel(false, &dst, &wgs![16, 16, 1]).unwrap();
        println!("{}", kernel);
    }
}
