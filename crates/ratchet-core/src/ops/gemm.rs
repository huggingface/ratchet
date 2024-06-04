use encase::ShaderType;
use half::f16;
use ratchet_macros::WgslMetadata;

use crate::{
    gguf::GGUFDType, gpu::dtype::WgslDType, rvec, Array, BindingMode, BuiltIn, DType, GEMMSpec,
    InvariantError, KernelElement, KernelSource, Matmul, OperationError, Scalar, Tensor, Vec2,
    Vec4, WgslFragment, WgslKernelBuilder, WgslPrimitive, WorkgroupSize,
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

impl From<Matmul> for GEMM {
    fn from(matmul: Matmul) -> Self {
        let Matmul {
            lhs,
            rhs,
            bias,
            trans_lhs,
            trans_rhs,
            trans_out,
        } = matmul;
        GEMM {
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
        _: &Tensor,
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
                    fn getA(d0 : i32, d1 : i32, d2 : i32) -> vec4<f32> {
                        return unpack4x8snorm(A[getAIndexFromCoords3D(vec3<i32>(d0,d1,d2)) / 4]) * 127.0;
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

    fn write_readers_and_writers<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        fits: (bool, bool, bool),
    ) -> Result<(), OperationError> {
        let FIT_A_OUTER = fits.0;
        let FIT_INNER = fits.1;
        let FIT_B_OUTER = fits.2;
        let accessor = P::render_type();

        let a_inner = if self.trans_lhs {
            wgsl! { value = getA(batch, col, row); }
        } else {
            wgsl! { value = getA(batch, row, col); }
        };

        let readA = if FIT_A_OUTER && FIT_INNER {
            a_inner
        } else if self.trans_lhs {
            wgsl! {
                if (row < metadata.aShape.z && col < metadata.aShape.y) {
                    'a_inner
                }
            }
        } else {
            wgsl! {
                if (row < metadata.aShape.y && col < metadata.aShape.z) {
                    'a_inner
                }
            }
        };

        let aAccessor = if matches!(self.lhs.dt(), DType::GGUF(GGUFDType::Q8_0(_))) {
            Vec4::<f32>::render_type()
        } else {
            accessor.clone()
        };

        builder.write_global(wgsl! {
            fn mm_readA(batch: i32, row: i32, col: i32) -> 'aAccessor {
                var value = 'aAccessor(0.0);
                'readA
                return value;
            }
        });

        let b_inner = if self.trans_rhs {
            wgsl! { value = getB(batch, col, row); }
        } else {
            wgsl! { value = getB(batch, row, col); }
        };

        let readB = if FIT_INNER && FIT_B_OUTER {
            b_inner
        } else if self.trans_rhs {
            wgsl! {
                if (row < metadata.bShape.z && col < metadata.bShape.y) {
                    'b_inner
                }
            }
        } else {
            wgsl! {
                if (row < metadata.bShape.y && col < metadata.bShape.z) {
                    'b_inner
                }
            }
        };

        builder.write_global(wgsl! {
            fn mm_readB(batch: i32, row: i32, col: i32) -> 'accessor {
                var value = 'accessor(0.0);
                'readB
                return value;
            }
        });

        let write = if FIT_A_OUTER && FIT_B_OUTER {
            wgsl! {
                var value = valueIn;
                let coords = vec3<i32>(batch, row, col);
                setOutputAtCoords(coords[0], coords[1], coords[2], value);
            }
        } else {
            wgsl! {
                if (row < metadata.dimAOuter && col < metadata.dimBOuter) {
                    var value = valueIn;
                    let coords = vec3<i32>(batch, row, col);
                    setOutputAtCoords(coords[0], coords[1], coords[2], value);
                }
            }
        };

        builder.write_global(wgsl! {
            fn mm_write(batch: i32, row: i32, col: i32, valueIn: 'accessor) {
                'write
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
                if bias.is_some() {
                    builder.register_storage("bias", BindingMode::ReadOnly, float_arr);
                }
                builder.register_storage("result", BindingMode::ReadWrite, float_arr);
            }
            DType::GGUF(GGUFDType::Q8_0(_)) => {
                builder.register_storage("A", ro, Array::<Scalar<u32>>::default());
                builder.register_storage("scale", ro, float_arr);
                if self.trans_rhs {
                    builder.register_storage("B", ro, Array::<Scalar<f32>>::default());
                } else {
                    builder.register_storage("B", ro, Array::<Vec4<f32>>::default());
                }
                if bias.is_some() {
                    builder.register_storage("bias", BindingMode::ReadOnly, float_arr);
                }
                builder.register_storage("result", BindingMode::ReadWrite, float_arr);
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
            (DType::F32, KernelElement::Scalar) => {
                self.build_gemm::<Scalar<f32>>(inplace, dst, workgroup_size, spec)
            }
            (DType::F32, KernelElement::Vec2) => {
                self.build_gemm::<Vec2<f32>>(inplace, dst, workgroup_size, spec)
            }
            (DType::F32, KernelElement::Vec4) => {
                self.build_gemm::<Vec4<f32>>(inplace, dst, workgroup_size, spec)
            }
            (DType::F16, KernelElement::Scalar) => {
                self.build_gemm::<Scalar<f16>>(inplace, dst, workgroup_size, spec)
            }
            (DType::F16, KernelElement::Vec2) => {
                self.build_gemm::<Vec2<f16>>(inplace, dst, workgroup_size, spec)
            }
            (DType::F16, KernelElement::Vec4) => {
                self.build_gemm::<Vec4<f16>>(inplace, dst, workgroup_size, spec)
            }
            (DType::GGUF(g), _) => match g {
                crate::gguf::GGUFDType::Q8_0(_) => {
                    self.build_gemm::<Scalar<f32>>(inplace, dst, workgroup_size, spec)
                }
                _ => unimplemented!(),
            },
            _ => panic!("Unsupported dtype"),
        }
    }

    fn build_gemm_scalar<P: WgslPrimitive>(
        &self,
        mut kernel_builder: WgslKernelBuilder,
    ) -> Result<KernelSource, OperationError> {
        const ROW_PER_THREAD: usize = 4;
        const TILE_DIM: usize = 32;

        let accessor = P::render_type();
        let dt = P::T::DT;
        let W = P::W;
        let T_W = TILE_DIM / W;
        kernel_builder.write_global(wgsl! {
            var<workgroup> mm_Asub: array<array<'accessor, 'T_W>, 'TILE_DIM>;
            var<workgroup> mm_Bsub: array<array<'accessor, 'T_W>, 'TILE_DIM>;
        });

        kernel_builder.write_main(wgsl! {
            let batch = i32(global_invocation_id.z);
            let batchA = batch % metadata.aShape[0];
            let batchB = batch % metadata.bShape[0];

            let tileRow = i32(local_invocation_id.y) * 4;
            let tileCol = i32(local_invocation_id.x) * 4;

            let globalRowStart = i32(workgroup_id.y) * 'T_W;
            let globalRow = i32(global_invocation_id.y) * 'ROW_PER_THREAD;
            let globalCol = i32(global_invocation_id.x) * 'ROW_PER_THREAD;

            let numTiles = (metadata.dimInner - 1) / 'TILE_DIM + 1;
            var kStart = 0;

            var acc: array<array<'dt, 'ROW_PER_THREAD>, 'ROW_PER_THREAD>;

            let tileRowA = i32(local_invocation_id.y) * 'ROW_PER_THREAD;
            let tileColA = i32(local_invocation_id.x) * 'ROW_PER_THREAD;
            let tileRowB = i32(local_invocation_id.y) * 'ROW_PER_THREAD;
            // Loop over shared dimension.
        });

        let a_inner = match self.lhs.dt() {
            DType::F32 | DType::F16 => {
                wgsl! {
                    for (var innerCol = 0; innerCol < 'ROW_PER_THREAD; innerCol++) {
                        let inputRow = tileRowA + innerRow;
                        let inputCol = tileColA + innerCol;

                        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
                            globalRowStart + inputRow,
                            kStart + inputCol);
                    }
                }
            }
            DType::GGUF(GGUFDType::Q8_0(_)) => {
                let mut inner = wgsl! {
                    let curRow = globalRow + innerRow;
                    let curCol = kStart + i32(local_invocation_id.x) * 4;

                    let absmax = getAbsMax(batchA, curRow, curCol);
                    let val = mm_readA(batchA, curRow, curCol) * absmax;
                };
                for i in 0..4 {
                    inner.push_str(
                        &wgsl! { mm_Asub[tileRowA + innerRow][tileColA + 'i] = val['i]; },
                    );
                }
                inner
            }
            _ => panic!("Unsupported dtype"),
        };

        let load_a = wgsl! {
            for (var innerRow = 0; innerRow < 'ROW_PER_THREAD; innerRow++) {
                'a_inner
            }
        };

        let load_b = wgsl! {
            // Load one tile of B into local memory.
            for (var innerRow = 0; innerRow < 'ROW_PER_THREAD; innerRow++) {
                for (var innerCol = 0; innerCol < 'ROW_PER_THREAD; innerCol++) {
                    let inputRow = tileRowB + innerRow;
                    let inputCol = tileCol + innerCol;

                    mm_Bsub[inputRow][inputCol] = mm_readB(batchB, kStart + inputRow, globalCol + innerCol);
                }
            }
        };

        let compute_acc = wgsl! {
            // Compute acc values for a single thread.
            for (var k = 0; k < 'T_W; k++) {
              let bidx = k * 'W;
              let BCached0 = mm_Bsub[bidx][tileCol + 0];
              let BCached1 = mm_Bsub[bidx][tileCol + 1];
              let BCached2 = mm_Bsub[bidx][tileCol + 2];
              let BCached3 = mm_Bsub[bidx][tileCol + 3];
              for (var innerRow = 0; innerRow < 'ROW_PER_THREAD; innerRow++) {
                let ACached = mm_Asub[tileRow + innerRow][k];
                acc[innerRow][0] = fma(ACached, BCached0, acc[innerRow][0]);
                acc[innerRow][1] = fma(ACached, BCached1, acc[innerRow][1]);
                acc[innerRow][2] = fma(ACached, BCached2, acc[innerRow][2]);
                acc[innerRow][3] = fma(ACached, BCached3, acc[innerRow][3]);
              }
            }
        };

        kernel_builder.write_main(wgsl! {
            for (var t = 0; t < numTiles; t++) {

                'load_a

                'load_b

                kStart = kStart + 'TILE_DIM;
                workgroupBarrier();

                'compute_acc
                workgroupBarrier();
            }

            var val: 'accessor;
        });

        for row in 0..ROW_PER_THREAD {
            for col in 0..ROW_PER_THREAD {
                let bias_val = if self.bias.is_some() {
                    if self.trans_out {
                        wgsl! { bias[globalRow + 'row] }
                    } else {
                        wgsl! { bias[globalCol + 'col] }
                    }
                } else {
                    wgsl! { 0.0 }
                };

                let writer = if self.trans_out {
                    wgsl! { mm_write(batch, globalCol + 'col, globalRow + 'row, val); }
                } else {
                    wgsl! { mm_write(batch, globalRow + 'row, globalCol + 'col, val); }
                };

                kernel_builder.write_main(wgsl! {
                    val = acc['row]['col] + 'bias_val;
                    'writer
                });
            }
        }

        Ok(kernel_builder.build()?)
    }

    fn build_gemm_vectorized<P: WgslPrimitive>(
        &self,
        mut kernel_builder: WgslKernelBuilder,
    ) -> Result<KernelSource, OperationError> {
        const ROW_PER_THREAD: usize = 4;
        const TILE_DIM: usize = 32;

        let accessor = P::render_type();
        let W = P::W;
        let T_W = TILE_DIM / W;
        kernel_builder.write_global(wgsl! {
            var<workgroup> mm_Asub: array<array<'accessor, 'T_W>, 'TILE_DIM>;
            var<workgroup> mm_Bsub: array<array<'accessor, 'T_W>, 'TILE_DIM>;
        });

        kernel_builder.write_main(wgsl! {
            let batch = i32(global_invocation_id.z);
            let batchA = batch % metadata.aShape[0];
            let batchB = batch % metadata.bShape[0];

            let localRow = i32(local_invocation_id.y);
            let tileRow = localRow * 'ROW_PER_THREAD;
            let tileCol = i32(local_invocation_id.x);

            let globalRow = i32(global_invocation_id.y) * 'ROW_PER_THREAD;
            let globalCol = i32(global_invocation_id.x) * 'W;

            let numTiles = (metadata.dimInner - 1) / 'TILE_DIM + 1;
            var kStart = 0;

            var acc: array<'accessor, 'ROW_PER_THREAD>;

            // Loop over shared dimension.
            let tileRowB = localRow * 'ROW_PER_THREAD;
        });

        let load_a_inner = match self.lhs.dt() {
            DType::F32 | DType::F16 => {
                wgsl! { mm_Asub[inputRow][inputCol] = mm_readA(batchA, globalRow + innerRow, kStart + inputCol * 'W); }
            }
            DType::GGUF(GGUFDType::Q8_0(_)) => {
                wgsl! {
                    let curRow = globalRow + innerRow;
                    let curCol = kStart + inputCol * 'W;

                    let absmax = getAbsMax(batchA, curRow, curCol);
                    mm_Asub[inputRow][inputCol] = mm_readA(batchA, curRow, curCol) * absmax;
                }
            }
            _ => panic!("Unsupported dtype"),
        };

        let load_a = wgsl! {
            // Load one tile of A into local memory.
            for (var innerRow = 0; innerRow < 'ROW_PER_THREAD; innerRow++) {
                let inputRow = tileRow + innerRow;
                let inputCol = tileCol;

                'load_a_inner
            }
        };

        let load_b = wgsl! {
            // Load one tile of B into local memory.
            for (var innerRow = 0; innerRow < 'ROW_PER_THREAD; innerRow++) {
                let inputRow = tileRowB + innerRow;
                let inputCol = tileCol;

                mm_Bsub[inputRow][inputCol] = mm_readB(batchB, kStart + inputRow, globalCol);
            }
        };

        let mut outer_body = WgslFragment::new(128);
        let mut inner_body = WgslFragment::new(128);
        for c in 0..W {
            let ident = format!("BCached{}", c);
            inner_body.write(wgsl! { acc[i] = fma('ident, 'accessor(ACached['c]), acc[i]); });
            outer_body.write(wgsl! { let 'ident = mm_Bsub[bidx + 'c][tileCol]; });
        }

        let compute_acc = wgsl! {
            // Compute acc values for a single thread.
            for (var k = 0; k < 'T_W; k++) {
              let bidx = k * 'W;
              'outer_body
              for (var i = 0; i < 'ROW_PER_THREAD; i++) {
                let ACached = mm_Asub[tileRow + i][k];
                'inner_body
              }
            }
        };

        kernel_builder.write_main(wgsl! {
            for (var t = 0; t < numTiles; t++) {

                'load_a

                'load_b

                kStart = kStart + 'TILE_DIM;
                workgroupBarrier();

                'compute_acc
                workgroupBarrier();
            }

            var val: 'accessor;
        });

        let bias_val = if self.bias.is_some() {
            wgsl! { bias[globalCol / 'W]; }
        } else {
            wgsl! { 0.0; }
        };

        for i in 0..ROW_PER_THREAD {
            kernel_builder.write_main(wgsl! {
                val = acc['i] + 'bias_val
                mm_write(batch, globalRow + 'i, globalCol, val);
            });
        }

        Ok(kernel_builder.build()?)
    }

    fn build_gemm<P: WgslPrimitive>(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
        spec: GEMMSpec,
    ) -> Result<KernelSource, OperationError> {
        println!("GEMM SPEC: {:#?}", spec);
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
        self.register_bindings::<P>(&mut kernel_builder, inplace)
            .unwrap();
        kernel_builder.write_metadata::<GEMMMeta>();
        self.write_indexing::<P>(&mut kernel_builder);
        self.write_getters::<P>(dst, &mut kernel_builder)?;
        self.write_readers_and_writers::<P>(&mut kernel_builder, spec.tile_fit())?;
        if P::W == 1 {
            self.build_gemm_scalar::<P>(kernel_builder)
        } else {
            self.build_gemm_vectorized::<P>(kernel_builder)
        }
    }
}
