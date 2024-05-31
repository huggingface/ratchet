use encase::ShaderType;
use half::f16;
use ratchet_macros::WgslMetadata;

use crate::{
    gguf::GGUFDType, rvec, Array, BindingMode, BuiltIn, DType, InvariantError, KernelElement,
    KernelSource, Matmul, OperationError, Scalar, Tensor, Vec4, WgslKernelBuilder, WgslPrimitive,
    WorkgroupSize,
};
use glam::IVec3;
use inline_wgsl::wgsl;

#[derive(Debug, Clone)]
pub struct GEMV {
    lhs: Tensor,
    rhs: Tensor,
    bias: Option<Tensor>,
    trans_lhs: bool,
    trans_rhs: bool,
    trans_out: bool,
}

impl From<Matmul> for GEMV {
    fn from(matmul: Matmul) -> Self {
        let Matmul {
            lhs,
            rhs,
            bias,
            trans_lhs,
            trans_rhs,
            trans_out,
        } = matmul;
        GEMV {
            lhs,
            rhs,
            bias,
            trans_lhs,
            trans_rhs,
            trans_out,
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[derive(Debug, Clone, ShaderType, WgslMetadata)]
pub struct GEMVMeta {
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

impl GEMV {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        let (A, _, bias) = (&self.lhs, &self.rhs, &self.bias);

        let float_arr = Array::<P>::default();

        if A.dt().is_float() {
            builder.register_storage("A", BindingMode::ReadOnly, float_arr);
            builder.register_storage("X", BindingMode::ReadOnly, float_arr);
        } else if A.dt().is_quantized() {
            builder.register_storage("A", BindingMode::ReadOnly, Array::<Scalar<u32>>::default());
            builder.register_storage("scale", BindingMode::ReadOnly, float_arr);
            builder.register_storage("X", BindingMode::ReadOnly, Array::<Vec4<P::T>>::default());
        } else {
            return Err(InvariantError::UnsupportedDType(A.dt()).into());
        }

        if bias.is_some() {
            builder.register_storage("bias", BindingMode::ReadOnly, float_arr);
        }
        builder.register_storage("result", BindingMode::ReadWrite, float_arr);
        builder.register_uniform();
        Ok(())
    }

    pub fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = KernelElement::Scalar;
        match (self.lhs.dt(), kernel_element) {
            (DType::F32, KernelElement::Scalar) => {
                self.render_gemv::<Scalar<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Scalar) => {
                self.render_gemv::<Scalar<f16>>(inplace, dst, workgroup_size)
            }
            (DType::GGUF(g), _) => match g {
                crate::gguf::GGUFDType::Q8_0(_) => todo!(),
                _ => unimplemented!(),
            },
            _ => panic!("Unsupported dtype"),
        }
    }

    fn render_gemv<P: WgslPrimitive>(
        &self,
        inplace: bool,
        _: &Tensor,
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
        let accessor = P::render_type();

        kernel_builder.write_metadata::<GEMVMeta>();

        let FIT = true;

        let workgroup_size_y = workgroup_size.y;
        let main_loop = match self.lhs.dt() {
            DType::GGUF(g) => match g {
                GGUFDType::Q8_0(_) => {
                    wgsl! {
                        let sIndex = (aOffset / 4) + row * metadata.aStrides.y / 32;
                        for (var k = i32(global_invocation_id.y); k < metadata.dimInner / 4; k+='workgroup_size_y) {
                            sum = fma(unpack4x8snorm_gguf(A[aIndex + k]) * scale[sIndex + (k/8)], X[k], sum);
                        }
                    }
                }
                _ => unimplemented!(),
            },
            _ => {
                wgsl! {
                    for (var k = i32(global_invocation_id.y); k < metadata.dimInner; k+='workgroup_size_y) {
                        sum = fma(A[aIndex + k], X[bOffset + k], sum);
                    }
                }
            }
        };

        kernel_builder.write_main(wgsl! { let row = i32(global_invocation_id.x); });
        if FIT {
            kernel_builder.write_main(wgsl! {
                if (row >= metadata.aShape.y) {
                    return;
                }
            });
        }

        kernel_builder.write_main(wgsl! {
            let batch = i32(global_invocation_id.z);
            let batchA = batch % metadata.aShape.x;
            let batchB = batch % metadata.bShape.x;
        });

        let n = P::W;
        kernel_builder.write_main(wgsl! {
            let aOffset = metadata.aStrides.x * batchA / 'n;
            let bOffset = metadata.bStrides.x * batchB / 'n;
            let outOffset = metadata.outStrides.x * batch / 'n;
        });

        kernel_builder.write_main(wgsl! { var sum = 'accessor(0.0); });
        kernel_builder.write_main(wgsl! { let aIndex = aOffset + row * metadata.aStrides.y / 'n; });

        kernel_builder.write_main(main_loop);

        Ok(kernel_builder.build()?)
    }
}

#[cfg(test)]
mod tests {
    use crate::{shape, wgs, Device, DeviceRequest, Tensor, GEMV};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    #[test]
    fn render_gemv() {
        let device = GPU_DEVICE.with(|d| d.clone());
        let lhs = Tensor::randn::<f32>(shape![128, 128], device.clone());
        let rhs = Tensor::randn::<f32>(shape![128, 1], device.clone());
        let bias = Tensor::randn::<f32>(shape![128], device.clone());
        let op = GEMV {
            lhs,
            rhs,
            bias: Some(bias),
            trans_lhs: false,
            trans_rhs: false,
            trans_out: false,
        };
        let dst = Tensor::zeros::<f32>(&shape![128, 1], &device);
        let kernel = op.render(false, &dst, wgs![16, 16, 1]);
    }
}
