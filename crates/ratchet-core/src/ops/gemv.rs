use encase::ShaderType;
use half::f16;
use ratchet_macros::WgslMetadata;

use crate::{
    gguf::{GGUFDType, Q8_0, QK4_0},
    gpu::dtype::WgslDType,
    rvec, BindGroupLayoutDescriptor, BuiltIn, DType, InvariantError, KernelElement, OpMetadata,
    OperationError, RVec, Scalar, Tensor, Vec2, Vec4, WgslFragment, WgslKernelBuilder,
    WgslPrimitive, WorkgroupSize,
};
use glam::IVec3;
use inline_wgsl::wgsl;
use wgpu::naga::Module;

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
    //TODO: this is stupid
    fn bindvars<P: WgslPrimitive<T, N>, T: WgslDType, const N: usize>(
        &self,
        _: bool,
        _: &Tensor,
    ) -> RVec<WgslFragment> {
        let dt = T::DT;
        let mut bind_vars = rvec![];
        match self.lhs.dt() {
            DType::F32 | DType::F16 => {
                let A = wgsl! { A: array<'dt>; }.into();
                let X = wgsl! { X: array<'dt>; }.into();
                bind_vars.push(A);
                bind_vars.push(X);

                if self.bias.is_some() {
                    let bias = wgsl! { bias: array<vec4<'dt>>; }.into();
                    bind_vars.push(bias);
                }
            }
            DType::GGUF(g) => match g {
                GGUFDType::Q8_0(_) => {
                    let A = wgsl! { A: array<u32>; }.into();
                    let scale = wgsl! { scale: array<'dt>; }.into();
                    let X = wgsl! { X: array<vec4<'dt>>; }.into();
                    bind_vars.push(A);
                    bind_vars.push(scale);
                    bind_vars.push(X);

                    if self.bias.is_some() {
                        let bias = wgsl! { bias: array<'dt>; }.into();
                        bind_vars.push(bias);
                    }
                }
                _ => unimplemented!(),
            },
            _ => panic!("Unsupported dtype"),
        }

        let result = wgsl! { result: array<'dt>; }.into();
        bind_vars.push(result);
        bind_vars
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

    pub fn render(&self, inplace: bool, dst: &Tensor, workgroup_size: WorkgroupSize) -> Module {
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
    ) -> Module {
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

        //TODO:
        // We need a method that returns a data structure that represents all of our buffer
        // bindings.
        // We shouldn't repeat the match statement.
        // The bind group layout also needs this information, we should amalgamate this.
        //
        // We should not need to call the kernel_builder.write_bindings method, this should be
        // performed upon construction.
        //
        // The only dynamic writing is the kernel body
        // The metadata structure can all be passed when constructing the kernel builder
        // Those are non optional, so should be on constructor.

        let bindings = self.storage_bind_group_layout(inplace).unwrap();
        let bind_vars = self.bindvars::<P, T, N>(inplace, dst);
        let accessor = P::render_type();

        println!("BINDINGS: {:?}", bindings);
        kernel_builder.write_bindings(&bindings, bind_vars);
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

        let row = wgsl! { let row = i32(global_invocation_id.x); };
        kernel_builder.write_main(row);
        if FIT {
            kernel_builder.write_main(wgsl! {
                if (row >= metadata.aShape.y) {
                    return;
                }
            });
        }

        let batches = wgsl! {
            let batch = i32(global_invocation_id.z);
            let batchA = batch % metadata.aShape.x;
            let batchB = batch % metadata.bShape.x;
        };
        kernel_builder.write_main(batches);

        let offset = wgsl! {
            let aOffset = metadata.aStrides.x * batchA / 'N;
            let bOffset = metadata.bStrides.x * batchB / 'N;
            let outOffset = metadata.outStrides.x * batch / 'N;
        };
        kernel_builder.write_main(offset);

        let sum = wgsl! { var sum = 'accessor(0.0); };
        kernel_builder.write_main(sum);
        let aIndex = wgsl! { let aIndex = aOffset + row * metadata.aStrides.y / 'N; };
        kernel_builder.write_main(aIndex);

        kernel_builder.write_main(main_loop);

        kernel_builder.build().unwrap()
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
    }
}
