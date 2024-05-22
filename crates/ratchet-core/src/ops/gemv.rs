use encase::ShaderType;
use half::f16;
use ratchet_macros::WgslMetadata;

use crate::{
    gguf::{GGUFDType, Q8_0, QK4_0},
    gpu::dtype::WgslDType,
    rvec, BindGroupLayoutDescriptor, BuiltIn, DType, InvariantError, KernelElement, OpMetadata,
    OperationError, RVec, Scalar, Tensor, Vec2, Vec4, WgslFragment, WgslKernel, WgslKernelBuilder,
    WgslPrimitive, WorkgroupSize,
};
use glam::IVec3;

#[derive(Debug, Clone)]
pub struct GEMV {
    lhs: Tensor,
    rhs: Tensor,
    bias: Option<Tensor>,
    trans_lhs: bool,
    trans_rhs: bool,
    trans_out: bool,
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

impl OpMetadata for GEMVMeta {
    fn render_wgsl() -> WgslFragment {
        //TODO: fix this in proc macro
        GEMVMeta::render()
    }
}

impl GEMV {
    fn bindvars<A: WgslPrimitive<T, N>, T: WgslDType, const N: usize>(
        &self,
        inplace: bool,
        _: &Tensor,
    ) -> RVec<WgslFragment> {
        let mut fragment = WgslFragment::new(32);
        fragment.write(&format!("X: array<{}>;\n", A::render_type()));
        rvec![fragment]
    }

    fn storage_bind_group_layout(
        &self,
        _inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        let (A, B, bias) = (&self.lhs, &self.rhs, &self.bias);
        let layout = match (A.dt(), B.dt(), bias.is_some()) {
            (DType::F32, DType::F32, false) => BindGroupLayoutDescriptor::binary(),
            (DType::F32, DType::F32, true) => BindGroupLayoutDescriptor::ternary(),
            (DType::GGUF(_), DType::F32, false) => BindGroupLayoutDescriptor::ternary(),
            (DType::GGUF(_), DType::F32, true) => BindGroupLayoutDescriptor::nthary(4),
            _ => return Err(InvariantError::UnsupportedDType(B.dt()).into()),
        };
        Ok(layout)
    }

    pub fn render(&self, inplace: bool, dst: &Tensor, workgroup_size: WorkgroupSize) -> WgslKernel {
        let kernel_element = KernelElement::Scalar;
        match (self.lhs.dt(), kernel_element) {
            (DType::F32, KernelElement::Scalar) => {
                self.render_gemv::<Scalar<f32>, _, 1>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec2) => {
                self.render_gemv::<Vec2<f32>, _, 2>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec4) => {
                self.render_gemv::<Vec4<f32>, _, 4>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Scalar) => {
                self.render_gemv::<Scalar<f16>, _, 1>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec2) => {
                self.render_gemv::<Vec2<f16>, _, 2>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec4) => {
                self.render_gemv::<Vec4<f16>, _, 4>(inplace, dst, workgroup_size)
            }
            (DType::GGUF(g), _) => match g {
                crate::gguf::GGUFDType::Q8_0(_) => todo!(),
                _ => unimplemented!(),
            },
            _ => panic!("Unsupported dtype"),
        }
    }

    fn render_gemv<P: WgslPrimitive<T, N>, T: WgslDType + num_traits::Float, const N: usize>(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: WorkgroupSize,
    ) -> WgslKernel {
        let device = self.lhs.device().try_gpu().unwrap();
        let mut kernel_builder = WgslKernelBuilder::new(
            workgroup_size.clone(),
            vec![
                BuiltIn::GlobalInvocationId,
                BuiltIn::LocalInvocationId,
                BuiltIn::WorkgroupId,
            ],
            device.compute_features().clone(),
        );
        let bindings = self.storage_bind_group_layout(inplace).unwrap();
        let bind_vars = self.bindvars::<P, T, N>(inplace, dst);
        let accessor = P::render_type();

        kernel_builder.write_bindings(&bindings, bind_vars);
        kernel_builder.write_metadata::<GEMVMeta>();

        let FIT = true;

        let fit_check = if FIT {
            r#"
            if (row >= metadata.aShape.y) {
                return;
            }
            "#
        } else {
            ""
        };

        let main_loop = match self.lhs.dt() {
            DType::GGUF(g) => match g {
                GGUFDType::Q8_0(_) => {
                    format!(
                        r#"
        let sIndex = (aOffset / 4) + row * metadata.aStrides.y / 32;
        for (var k = i32(global_invocation_id.y); k < metadata.dimInner / 4; k+={}) {{
            sum = fma(unpack4x8snorm_gguf(A[aIndex + k]) * scale[sIndex + (k/8)], X[k], sum);
        }}
"#,
                        workgroup_size.y
                    )
                }
                _ => unimplemented!(),
            },
            _ => {
                format!(
                    r#"
        for (var k = i32(global_invocation_id.y); k < metadata.dimInner; k+={}) {{
            sum = fma(A[aIndex + k], X[bOffset + k], sum);
        }}"#,
                    workgroup_size.y
                )
            }
        };

        let indexing = format!(
            r#"
let row = i32(global_invocation_id.x);
{fit_check}
let batch = i32(global_invocation_id.z);
let batchA = batch % metadata.aShape.x;
let batchB = batch % metadata.bShape.x;

let aOffset = metadata.aStrides.x * batchA / {N}
let bOffset = metadata.bStrides.x * batchB / {N}
let outOffset = metadata.outShapeStrides.x * batch / {N}

var sum = {accessor}(0.0);
let aIndex = aOffset + row * metadata.aStrides.y / {N};
    "#
        );

        kernel_builder.write_main(indexing.into());
        kernel_builder.write_main(main_loop.into());

        kernel_builder.render()
    }
}

#[cfg(test)]
mod tests {
    use crate::{shape, wgs, Device, DeviceRequest, Tensor, GEMV};
    use wgpu::naga::front::wgsl::parse_str;

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
        println!("{}", kernel);
        parse_str(&kernel.to_string()).unwrap();
    }
}
