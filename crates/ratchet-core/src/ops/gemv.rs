use encase::ShaderType;
use half::f16;
use ratchet_macros::WgslMetadata;

use crate::{
    gpu::dtype::WgslDType, rvec, Array, BindingMode, BuiltIn, DType, GEMMSpec, InvariantError,
    KernelElement, KernelSource, Matmul, OperationError, Scalar, Tensor, Vec4, WgslKernelBuilder,
    WgslPrimitive, WorkgroupSize,
};
use glam::IVec3;
use inline_wgsl::wgsl;
use num_traits::Zero;

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

    pub fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
        spec: GEMMSpec,
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
        spec: GEMMSpec,
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
        let n = P::W;
        let accessor = P::render_type();
        let scalar = P::T::DT;
        let zero = P::T::zero().render();

        kernel_builder.write_metadata::<GEMVMeta>();

        let work_size = (workgroup_size.x * workgroup_size.y / (n as u32)).render();
        kernel_builder.write_global(wgsl! {
            var<workgroup> work: array<'accessor, 'work_size>;
        });

        let (TILE_X, _) = spec.heuristic.as_workgroup_size();
        let A_FIT = spec.lhs_shape()[1] % TILE_X == 0;

        let readA = match (A_FIT, self.lhs.dt()) {
            (true, DType::F32) | (true, DType::F16) => {
                wgsl! {
                    fn readA(batch: i32, row: i32, col: i32) -> 'scalar {
                        return A[dot(metadata.aStrides, vec3<i32>(batch, row, col))];
                    }
                }
            }
            (false, DType::F32) | (false, DType::F16) => {
                wgsl! {
                    fn readA(batch: i32, row: i32, col: i32) -> 'scalar {
                        var val = 'zero;
                        if (row <= metadata.aShape.y) {
                            val = A[dot(metadata.aStrides, vec3<i32>(batch, row, col))];
                        }
                        return val;
                    }
                }
            }
            (true, DType::Q8_0F(_)) => {
                wgsl! {
                    fn readA(batch: i32, row: i32, col: i32) -> vec4<f32> {
                        return unpack4x8snorm(A[dot(metadata.aStrides, vec3<i32>(batch, row, col))]) * 127f;
                    }
                }
            }
            _ => unimplemented!(),
        };
        kernel_builder.write_global(readA);

        let workgroup_size_y = workgroup_size.y;
        let main_loop = match self.lhs.dt() {
            DType::Q8_0F(_) => {
                wgsl! {
                    let sIndex = (aOffset / 4) + row * metadata.aStrides.y / 32;
                    for (var k = i32(global_invocation_id.y); k < metadata.dimInner / 4; k+='workgroup_size_y / 4) {
                        sum = fma(unpack4x8snorm(A[aIndex + k]) * 127f * scale[sIndex + (k/8)], X[k], sum);
                    }
                }
            }
            _ => {
                wgsl! {
                    for (var k = i32(global_invocation_id.y); k < metadata.dimInner; k+='workgroup_size_y) {
                        sum = fma(readA(batchA, row, k), X[bOffset + k], sum);
                    }
                }
            }
        };

        kernel_builder.write_main(wgsl! { let row = i32(global_invocation_id.x); });

        kernel_builder.write_main(wgsl! {
            let batch = i32(global_invocation_id.z);
            let batchA = batch % metadata.aShape.x;
            let batchB = batch % metadata.bShape.x;
        });

        kernel_builder.write_main(wgsl! {
            let aOffset = metadata.aStrides.x * batchA / 'n;
            let bOffset = metadata.bStrides.x * batchB / 'n;
            let outOffset = metadata.outStrides.x * batch / 'n;
        });

        kernel_builder.write_main(wgsl! { var sum = 'accessor(0.0); });
        kernel_builder.write_main(wgsl! { let aIndex = aOffset + row * metadata.aStrides.y / 'n; });

        kernel_builder.write_main(main_loop);

        let workgroup_size_x = workgroup_size.x.render();
        let workgroup_size_y = workgroup_size.y.render();
        kernel_builder.write_main(wgsl! {
            let rows = 'workgroup_size_x;
            let cols = 'workgroup_size_y / 'n;

            let ii = u32(local_invocation_id.x);
            let jj = u32(local_invocation_id.y);
            work[ii + rows * jj] = sum;
            workgroupBarrier();

            // Reduce sums in log2(cols) steps
            for (var s = u32(cols) / 2u; s > 0u; s >>= 1u) {
                if (jj < s) {
                    work[ii + rows * jj] += work[ii + rows * (jj + s)];
                }
                workgroupBarrier();
            }
        });

        let bias = if self.bias.is_some() {
            wgsl! { bias[row] }
        } else {
            wgsl! { 0. }
        };

        let finalizer = match P::W {
            4 | 2 => wgsl! { result[outOffset + row] = dot(work[ii], 'accessor(1.0)) + 'bias; },
            1 => wgsl! { result[outOffset + row] = work[ii] + 'bias; },
            _ => unimplemented!(),
        };

        kernel_builder.write_main(wgsl! {
            if (jj == 0) {
                'finalizer
            }
        });

        Ok(kernel_builder.build()?)
    }
}
