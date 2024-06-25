use encase::ShaderType;
use half::f16;
use ratchet_macros::WgslMetadata;

use crate::{
    gpu::dtype::WgslDType, rvec, Array, BindingMode, BuiltIn, DType, GEMMSpec, InvariantError,
    KernelElement, KernelSource, Matmul, OperationError, Scalar, Tensor, Vec2, Vec4, WgslFragment,
    WgslKernelBuilder, WgslPrimitive, WorkgroupSize,
};
use glam::IVec3;
use inline_wgsl::wgsl;

#[allow(clippy::too_many_arguments)]
#[derive(Debug, Clone, ShaderType, WgslMetadata)]
pub struct QGEMMMeta {
    dummy: u32,
}

#[derive(Debug, Clone)]
pub struct Quantized {
    lhs: Tensor,
    rhs: Tensor,
    bias: Option<Tensor>,
    trans_lhs: bool,
    trans_rhs: bool,
    trans_out: bool,
}

impl From<Matmul> for Quantized {
    fn from(matmul: Matmul) -> Self {
        let Matmul {
            lhs,
            rhs,
            bias,
            trans_lhs,
            trans_rhs,
            trans_out,
        } = matmul;
        Quantized {
            lhs,
            rhs,
            bias,
            trans_lhs,
            trans_rhs,
            trans_out,
        }
    }
}

impl Quantized {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        let (A, _, bias) = (&self.lhs, &self.rhs, &self.bias);

        let farr = Array::<P>::default();
        let scalar_farr = Array::<Scalar<P::T>>::default();
        let scalar_u32 = Array::<Scalar<u32>>::default();

        let ro = BindingMode::ReadOnly;
        match A.dt() {
            DType::F32 | DType::F16 => {
                builder.register_storage("A", ro, farr);
                builder.register_storage("B", ro, farr);
                if bias.is_some() {
                    builder.register_storage("bias", BindingMode::ReadOnly, farr);
                }
                builder.register_storage("result", BindingMode::ReadWrite, farr);
            }
            DType::Q8_0F(_) | DType::Q8_0H(_) => {
                builder.register_storage("A", ro, scalar_u32);
                builder.register_storage("scale", ro, farr);
                builder.register_storage("B", ro, scalar_farr);
                if bias.is_some() {
                    builder.register_storage("bias", BindingMode::ReadOnly, farr);
                }
                builder.register_storage("result", BindingMode::ReadWrite, farr);
            }
            DType::Q4_KF(_) | DType::Q4_KH(_) => {
                builder.register_storage("A", ro, scalar_u32);
                builder.register_storage("scales", ro, scalar_u32);
                builder.register_storage("dmin", ro, scalar_farr);
                builder.register_storage("d", ro, scalar_farr);
                builder.register_storage("B", ro, scalar_farr);
                if bias.is_some() {
                    builder.register_storage("bias", BindingMode::ReadOnly, farr);
                }
                builder.register_storage("result", BindingMode::ReadWrite, farr);
            }
            _ => return Err(InvariantError::UnsupportedDType(A.dt()).into()),
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
        let kernel_element = spec.select_kernel_element();
        match (self.lhs.dt(), kernel_element) {
            (DType::Q4_KF(_), _) => {
                self.build_gemm::<Scalar<f32>>(inplace, dst, workgroup_size, spec)
            }
            (DType::Q4_KH(_), _) => {
                self.build_gemm::<Scalar<f16>>(inplace, dst, workgroup_size, spec)
            }
            _ => return Err(InvariantError::UnsupportedDType(self.lhs.dt()).into()),
        }
    }

    fn build_qgemm<P: WgslPrimitive>(
        &self,
        mut kernel_builder: WgslKernelBuilder,
    ) -> Result<KernelSource, OperationError> {
        /* Extract the 16 x 6 bit values scales-mins pairs. The
         * encoding of those values is odd because of performance
         * reasons:
         *
         *  dddddddd dddddddd dddddddd dddddddd mmmmmmmm mmmmmmmm
         *  44000000|55111111|66222222|77333333|44000000|55111111
         *
         *  mmmmmmmm mmmmmmmm mmmmdddd mmmmdddd mmmmdddd mmmmdddd
         *  66222222|77333333|44444444|55555555|66666666|77777777
         *
         * In the above diagram you can see the 12 bytes and the
         * scales/mins 6 bits encodings. */
        kernel_builder.write_global(wgsl! {
           fn extract_subblock_first_four(soffset: u32, pair_idx: u32) -> vec2<u32> {
                let s0 = scales[soffset]; //first 4 bytes
                let s1 = scales[soffset + 1u];//bytes 4-7
                let pair_bit_offset = (8u * pair_idx);
                return vec2<u32>((s0 >> pair_bit_offset) & 63u, (s1 >> pair_bit_offset) & 63u);
           }

           fn extract_subblock_latter_four(soffset: u32, pair_idx: u32) -> vec2<u32> {
                let s0 = scales[soffset]; //first 4 bytes
                let s1 = scales[soffset + 1u];//bytes 4-7
                let s2 = scales[soffset + 2u];//bytes 8-11

                //All of the lower bits are in the last 4 bytes (s2)
                //2bit values are distributed in 1-7 bytes
                //[01][011101] == 29
                let shift = 8u * (pair_idx - 4u);
                let dl = (s2 >> shift & 0xF); //mask 4 bits
                let dh = (s0 >> (6u + shift)) & 0x3; //mask 2 bits

                let ml = (s2 >> (shift + 4u) & 0xF);
                let mh = (s1 >> (6u + shift)) & 0x3;

                return vec2<u32>((dh << 4u) | dl, (mh << 4u) | ml);
           }


           fn get_subblock_scale_min(so: u32, pair_index: u32) -> vec2<u32> {
                return select(
                    extract_subblock_latter_four(so, pair_index),
                    extract_subblock_first_four(so, pair_index),
                    pair_index < 4u
                );
            }
        });

        kernel_builder.write_main(wgsl! {
            var scm = get_subblock_scale_min(0u, 1u);
            var delta = f16(scm.x) * d[0];
            var min = f16(scm.y) * dmin[0];
            result[0] = delta;
            result[1] = min;

            scm = get_subblock_scale_min(0u, 7u);
            delta = f16(scm.x) * d[0];
            min = f16(scm.y) * dmin[0];
            result[2] = delta;
            result[3] = min;
        });

        let x = kernel_builder.build()?;
        println!("{}", x);
        Ok(x)
    }

    fn build_gemm<P: WgslPrimitive>(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
        spec: GEMMSpec,
    ) -> Result<KernelSource, OperationError> {
        let device = self.lhs.device().try_gpu().unwrap();
        let mut kernel_builder = WgslKernelBuilder::new(
            workgroup_size.clone(),
            rvec![
                BuiltIn::GlobalInvocationId,
                BuiltIn::LocalInvocationId,
                BuiltIn::WorkgroupId
            ],
            device.compute_features().clone(),
        );
        kernel_builder.write_metadata::<QGEMMMeta>();
        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        if spec.is_gemv() {
            todo!()
        } else {
            self.build_qgemm::<P>(kernel_builder)
        }
    }
}
