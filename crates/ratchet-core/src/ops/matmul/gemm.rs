use encase::ShaderType;
use half::f16;
use ratchet_macros::WgslMetadata;

use crate::{
    gpu::dtype::WgslDType, rvec, wgc, wgs, Array, BindGroupLayoutDescriptor as BGLD, BindingMode,
    BuiltIn, DType, InvariantError, Kernel, KernelElement, KernelKey, KernelRenderable,
    KernelSource, Matmul, MatmulSpec, OperationError, Scalar, Strides, Tensor, Vec2, Vec4,
    WgslFragment, WgslKernelBuilder, WgslPrimitive, WorkgroupCount, WorkgroupSize, Workload,
};
use glam::IVec3;
use inline_wgsl::wgsl;

pub struct GEMM {
    lhs: Tensor,
    rhs: Tensor,
    bias: Option<Tensor>,
    trans_lhs: bool,
    trans_rhs: bool,
    trans_dst: bool,
    spec: MatmulSpec,
}

impl GEMM {
    pub fn from_matmul(matmul: &Matmul, spec: MatmulSpec) -> Self {
        let Matmul {
            lhs,
            rhs,
            bias,
            trans_lhs,
            trans_rhs,
            trans_dst,
        } = matmul.clone();
        Self {
            lhs,
            rhs,
            bias,
            trans_lhs,
            trans_rhs,
            trans_dst,
            spec,
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[derive(Debug, Clone, ShaderType, WgslMetadata)]
pub struct GEMMMeta {
    lhs_shape: IVec3,
    lhs_strides: IVec3,
    rhs_shape: IVec3,
    rhs_strides: IVec3,
    dst_shape: IVec3,
    dst_strides: IVec3,
    dim_lhs_outer: i32,
    dim_rhs_outer: i32,
    dim_inner: i32,
}

impl KernelRenderable for GEMM {
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
            DType::Q8_0F(_) | DType::Q8_0H(_) => {
                builder.register_storage("A", ro, Array::<Scalar<u32>>::default());
                builder.register_storage("scale", ro, float_arr);
                builder.register_storage("B", ro, Array::<Scalar<P::T>>::default());
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

    fn render<P: WgslPrimitive>(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let device = dst.device().try_gpu()?;
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
        kernel_builder.render_metadata(&self.metadata(dst, &self.kernel_element(dst))?);
        self.write_indexing::<P>(&mut kernel_builder);
        self.write_getters::<P>(dst, &mut kernel_builder)?;
        self.write_readers_and_writers::<P>(&mut kernel_builder, self.spec.tile_fit())?;
        if P::W == 1 {
            self.render_scalar::<P>(kernel_builder)
        } else {
            self.render_vectorized::<P>(kernel_builder)
        }
    }
}

impl Kernel for GEMM {
    type Metadata = GEMMMeta;

    fn kernel_name(&self) -> String {
        "gemm".to_string()
    }

    fn kernel_key(
        &self,
        workgroup_size: &WorkgroupSize,
        inplace: bool,
        srcs: &[&Tensor],
        dst: &Tensor,
        kernel_element: &KernelElement,
    ) -> KernelKey {
        let (a_fit, b_fit, out_fit) = self.spec.tile_fit();
        let bias_key = if self.bias.is_some() { "bias" } else { "" };

        let additional = format!(
            "{}_{}_{}_{}_{}_{}_{}",
            if a_fit { "" } else { "a_checked" },
            if b_fit { "" } else { "b_checked" },
            if out_fit { "" } else { "out_checked" },
            if self.trans_lhs { "trans_a" } else { "" },
            if self.trans_rhs { "trans_b" } else { "" },
            if self.trans_dst { "trans_dst" } else { "" },
            bias_key
        );

        KernelKey::new(
            &self.kernel_name(),
            srcs,
            dst,
            workgroup_size,
            inplace,
            kernel_element,
            Some(&additional),
        )
    }

    fn metadata(&self, _: &Tensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let spec = &self.spec;
        let mut lhs_shape = spec.lhs_shape().clone();
        lhs_shape.insert(0, spec.lhs_stack());
        let lhs_strides = Strides::from(&lhs_shape);

        let mut rhs_shape = spec.rhs_shape().clone();
        rhs_shape.insert(0, spec.rhs_stack());
        let rhs_strides = Strides::from(&rhs_shape);

        let mut dst_shape = spec.dst_shape().clone();
        dst_shape.insert(0, spec.stacks());
        let dst_strides = Strides::from(&dst_shape);

        let dim_lhs_outer = spec.dim_lhs_outer() as i32;
        let dim_rhs_outer = spec.dim_rhs_outer() as i32;
        let dim_inner = spec.dim_inner() as i32;

        Ok(GEMMMeta {
            lhs_shape: lhs_shape.into(),
            lhs_strides: lhs_strides.into(),
            rhs_shape: rhs_shape.into(),
            rhs_strides: rhs_strides.into(),
            dst_shape: dst_shape.into(),
            dst_strides: dst_strides.into(),
            dim_lhs_outer,
            dim_rhs_outer,
            dim_inner,
        })
    }

    fn calculate_dispatch(&self, _: &Tensor) -> Result<crate::Workload, OperationError> {
        //GEMM
        let TILE_DIM = 32;
        let lhs_shape = self.spec.lhs_shape();
        let rhs_shape = self.spec.rhs_shape();

        let dimA = if self.trans_lhs {
            lhs_shape[1]
        } else {
            lhs_shape[0]
        };

        let dimB = if self.trans_rhs {
            rhs_shape[0]
        } else {
            rhs_shape[1]
        };

        let group_x = WorkgroupCount::div_ceil(dimB as _, TILE_DIM);
        let div_ceil = WorkgroupCount::div_ceil(dimA, TILE_DIM);
        let group_y = div_ceil;
        let workgroup_count = wgc![group_x as _, group_y as _, self.spec.stacks() as _];

        Ok(Workload {
            workgroup_count,
            workgroup_size: wgs![8, 8, 1],
        })
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = self.spec.select_kernel_element();
        match (self.lhs.dt(), kernel_element) {
            (DType::F32, KernelElement::Scalar) => {
                self.render::<Scalar<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec2) => {
                self.render::<Vec2<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec4) => {
                self.render::<Vec4<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Scalar) => {
                self.render::<Scalar<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec2) => {
                self.render::<Vec2<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec4) => {
                self.render::<Vec4<f16>>(inplace, dst, workgroup_size)
            }
            (DType::Q8_0F(_), _) => self.render::<Scalar<f32>>(inplace, dst, workgroup_size),
            (DType::Q8_0H(_), _) => self.render::<Scalar<f16>>(inplace, dst, workgroup_size),
            (DType::Q4_KF(_), _) => self.render::<Scalar<f32>>(inplace, dst, workgroup_size),
            (DType::Q4_KH(_), _) => self.render::<Scalar<f16>>(inplace, dst, workgroup_size),
            _ => Err(InvariantError::UnsupportedDType(self.lhs.dt()).into()),
        }
    }

    fn kernel_element(&self, _: &Tensor) -> KernelElement {
        self.spec.select_kernel_element()
    }

    fn storage_bind_group_layout(&self, _inplace: bool) -> Result<BGLD, OperationError> {
        let (LHS, RHS, bias) = (&self.lhs, &self.rhs, &self.bias);
        let layout = match (LHS.dt(), RHS.dt(), bias.is_some()) {
            (DType::F32 | DType::F16, DType::F32 | DType::F16, false) => BGLD::binary(),
            (DType::F32 | DType::F16, DType::F32 | DType::F16, true) => BGLD::ternary(),
            (DType::Q8_0F(_) | DType::Q8_0H(_), DType::F32 | DType::F16, false) => BGLD::ternary(),
            (DType::Q8_0F(_) | DType::Q8_0H(_), DType::F32 | DType::F16, true) => BGLD::nthary(4),
            _ => return Err(InvariantError::UnsupportedDType(RHS.dt()).into()),
        };
        Ok(layout)
    }
}

impl GEMM {
    fn write_indexing<P: WgslPrimitive>(&self, builder: &mut WgslKernelBuilder) {
        let accessor = P::render_type();
        let W = P::W;
        builder.write_global(wgsl! {
            fn getAIndexFromCoords3D(coords : vec3<i32>) -> i32 {
                return dot(coords, metadata.lhs_strides);
            }

            fn getBIndexFromCoords3D(coords : vec3<i32>) -> i32 {
                return dot(coords, metadata.rhs_strides);
            }

            fn getOutputIndexFromCoords(coords : vec3<i32>) -> i32 {
                return dot(coords, metadata.dst_strides);
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
        let dt = P::T::DT;
        builder.write_unpack(A.dt());

        let a_getters = match A.dt() {
            DType::F32 | DType::F16 => {
                wgsl! {
                    fn getA(d0 : i32, d1 : i32, d2 : i32) -> 'accessor {
                        return 'accessor(A[getAIndexFromCoords3D(vec3<i32>(d0,d1,d2)) / 'W]);
                    }
                }
            }
            DType::Q8_0F(_) | DType::Q8_0H(_) => {
                wgsl! {
                    fn getA(d0 : i32, d1 : i32, d2 : i32) -> vec4<'dt> {
                        return unpack(A[getAIndexFromCoords3D(vec3<i32>(d0,d1,d2)) / 4]);
                    }

                    fn getAbsMax(d0 : i32, d1 : i32, d2 : i32) -> 'dt {
                        let abs_index = getAIndexFromCoords3D(vec3<i32>(d0,d1,d2)) / 32;
                        return scale[abs_index];
                    }
                }
            }
            _ => return Err(InvariantError::UnsupportedDType(A.dt()).into()),
        };
        builder.write_global(a_getters);

        match A.dt() {
            DType::F32 | DType::F16 => {
                builder.write_global(wgsl! {
                    fn getB(d0 : i32, d1 : i32, d2 : i32) -> 'accessor {
                        return 'accessor(B[getBIndexFromCoords3D(vec3<i32>(d0,d1,d2)) / 'W]);
                    }
                });
            }
            DType::Q8_0F(_) | DType::Q8_0H(_) => {
                builder.write_global(wgsl! {
                    fn getB(d0 : i32, d1 : i32, d2 : i32) -> 'dt {
                        return 'dt(B[getBIndexFromCoords3D(vec3<i32>(d0,d1,d2)) / 'W]);
                    }
                });
            }
            _ => return Err(InvariantError::UnsupportedDType(A.dt()).into()),
        }

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
                if (row < metadata.lhs_shape.z && col < metadata.lhs_shape.y) {
                    'a_inner
                }
            }
        } else {
            wgsl! {
                if (row < metadata.lhs_shape.y && col < metadata.lhs_shape.z) {
                    'a_inner
                }
            }
        };

        let aAccessor = match self.lhs.dt() {
            DType::Q8_0F(_) => Vec4::<f32>::render_type(),
            DType::Q8_0H(_) => Vec4::<f16>::render_type(),
            _ => accessor.clone(),
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
                if (row < metadata.rhs_shape.z && col < metadata.rhs_shape.y) {
                    'b_inner
                }
            }
        } else {
            wgsl! {
                if (row < metadata.rhs_shape.y && col < metadata.rhs_shape.z) {
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
                if (row < metadata.dim_lhs_outer && col < metadata.dim_rhs_outer) {
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

    fn render_scalar<P: WgslPrimitive>(
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
            let batchA = batch % metadata.lhs_shape[0];
            let batchB = batch % metadata.rhs_shape[0];

            let tileRow = i32(local_invocation_id.y) * 'ROW_PER_THREAD;
            let tileCol = i32(local_invocation_id.x) * 4;

            let globalRowStart = i32(workgroup_id.y) * 'T_W;
            let globalRow = i32(global_invocation_id.y) * 'ROW_PER_THREAD;
            let globalCol = i32(global_invocation_id.x) * 'ROW_PER_THREAD;

            let numTiles = (metadata.dim_inner - 1) / 'TILE_DIM + 1;
            var kStart = 0;

            //ALWAYS ACCUM IN FP32
            var acc: array<array<f32, 'ROW_PER_THREAD>, 'ROW_PER_THREAD>;

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
            DType::Q8_0F(_) | DType::Q8_0H(_) => {
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
                acc[innerRow][0] += f32(ACached * BCached0);
                acc[innerRow][1] += f32(ACached * BCached1);
                acc[innerRow][2] += f32(ACached * BCached2);
                acc[innerRow][3] += f32(ACached * BCached3);
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
                    if self.trans_dst {
                        wgsl! { bias[globalRow + 'row] }
                    } else {
                        wgsl! { bias[globalCol + 'col] }
                    }
                } else {
                    wgsl! { 0. }
                };

                let writer = if self.trans_dst {
                    wgsl! { mm_write(batch, globalCol + 'col, globalRow + 'row, val); }
                } else {
                    wgsl! { mm_write(batch, globalRow + 'row, globalCol + 'col, val); }
                };

                kernel_builder.write_main(wgsl! {
                    val = 'dt(acc['row]['col]) + 'bias_val;
                    'writer
                });
            }
        }

        let x = kernel_builder.build()?;
        println!("{}", x);
        Ok(x)
    }

    fn render_vectorized<P: WgslPrimitive>(
        &self,
        mut kernel_builder: WgslKernelBuilder,
    ) -> Result<KernelSource, OperationError> {
        const ROW_PER_THREAD: usize = 4;
        const TILE_DIM: usize = 32;

        let accessor = P::render_type();
        let W = P::W;

        let fp32_accessor = match W {
            1 => Scalar::<f32>::render_type(),
            2 => Vec2::<f32>::render_type(),
            4 => Vec4::<f32>::render_type(),
            _ => panic!("Unsupported W"),
        };

        let T_W = TILE_DIM / W;
        kernel_builder.write_global(wgsl! {
            var<workgroup> mm_Asub: array<array<'accessor, 'T_W>, 'TILE_DIM>;
            var<workgroup> mm_Bsub: array<array<'accessor, 'T_W>, 'TILE_DIM>;
        });

        kernel_builder.write_main(wgsl! {
            let batch = i32(global_invocation_id.z);
            let batchA = batch % metadata.lhs_shape[0];
            let batchB = batch % metadata.rhs_shape[0];

            let localRow = i32(local_invocation_id.y);
            let tileRow = localRow * 'ROW_PER_THREAD;
            let tileCol = i32(local_invocation_id.x);

            let globalRow = i32(global_invocation_id.y) * 'ROW_PER_THREAD;
            let globalCol = i32(global_invocation_id.x) * 'W;

            let numTiles = (metadata.dim_inner - 1) / 'TILE_DIM + 1;
            var kStart = 0;

            var acc: array<'fp32_accessor, 'ROW_PER_THREAD>;

            // Loop over shared dimension.
            let tileRowB = localRow * 'ROW_PER_THREAD;
        });

        let load_a_inner = match self.lhs.dt() {
            DType::F32 | DType::F16 => {
                wgsl! { mm_Asub[inputRow][inputCol] = mm_readA(batchA, globalRow + innerRow, kStart + inputCol * 'W); }
            }
            DType::Q8_0F(_) | DType::Q8_0H(_) => {
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
            let bIdent = format!("BCached{}", c);
            inner_body.write(wgsl! {
                acc[i] += 'fp32_accessor('accessor(ACached['c]) * 'bIdent);
            });
            outer_body.write(wgsl! { let 'bIdent = mm_Bsub[bidx + 'c][tileCol]; });
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
                val = 'accessor(acc['i]) + 'bias_val;
                mm_write(batch, globalRow + 'i, globalCol, val);
            });
        }

        let x = kernel_builder.build()?;
        //println!("{}", x);
        println!("bad");
        Ok(x)
    }
}
