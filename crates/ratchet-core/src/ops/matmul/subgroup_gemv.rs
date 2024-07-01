use encase::ShaderType;
use ratchet_macros::WgslMetadata;

use crate::Kernel;

pub struct SubgroupGEMV {}

#[derive(Debug, Clone, ShaderType, WgslMetadata)]
pub struct SubgroupGEMVMeta {
    OVL: i32, //out_vec_size
    IVL: i32, //in_vec_size
}

impl Kernel for SubgroupGEMV {
    type Metadata;

    fn metadata(
        &self,
        dst: &crate::Tensor,
        kernel_element: &crate::KernelElement,
    ) -> Result<Self::Metadata, crate::OperationError> {
        todo!()
    }

    fn calculate_dispatch(
        &self,
        dst: &crate::Tensor,
    ) -> Result<crate::Workload, crate::OperationError> {
        todo!()
    }

    fn kernel_element(&self, dst: &crate::Tensor) -> crate::KernelElement {
        todo!()
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &crate::Tensor,
        workgroup_size: &crate::WorkgroupSize,
    ) -> Result<crate::KernelSource, crate::OperationError> {
        todo!()
    }
}

impl SubgroupGEMV {
    fn subgroup_gemv<P: WgslPrimitive>(
        &self,
        inplace: bool,
        _: &Tensor,
        workgroup_size: &WorkgroupSize,
        _: MatmulSpec,
    ) -> Result<KernelSource, OperationError> {
        const TM: usize = 4;
        const TN: usize = 4;
        const BM: usize = 8;
        const BN: usize = 32;

        if matches!(self.lhs.dt(), DType::Q8_0F(_) | DType::Q8_0H(_)) {
            assert!(TN == 4);
        }

        let device = self.lhs.device().try_gpu().unwrap();
        let mut kernel_builder = WgslKernelBuilder::new(
            workgroup_size.clone(),
            rvec![
                BuiltIn::LocalInvocationIndex,
                BuiltIn::WorkgroupId,
                BuiltIn::SubgroupSize,
                BuiltIn::SubgroupInvocationId,
            ],
            device.compute_features().clone(),
        );
        kernel_builder.render_metadata::<SubgroupGEMVMeta>();
        kernel_builder.write_unpack(self.lhs.dt());

        self.register_bindings_subgroup::<P>(&mut kernel_builder, inplace)
            .unwrap();

        let dt = P::T::DT;
        let work_size = BN * TN * 2;
        kernel_builder.write_global(wgsl! {
            var<workgroup> tgpMemory: array<'dt, 'work_size>;
        });

        let zero = P::T::zero().render();
        let thread_locals = match self.lhs.dt() {
            DType::F32 | DType::F16 => {
                wgsl! {
                    var result: array<f32, 'TM>;
                    var inter: array<'dt, 'TN>;
                    var vCoeff: array<'dt, 'TN>;
                }
            }
            DType::Q8_0F(_) | DType::Q8_0H(_) => {
                wgsl! {
                    var result: array<f32, 'TM>;
                    var inter = vec4<'dt>('zero);
                    var vCoeff = vec4<'dt>('zero);
                }
            }
            _ => unimplemented!(),
        };

        kernel_builder.write_main(wgsl! {
            let simd_gid = local_invocation_index / subgroup_size;
            let simd_lid = subgroup_invocation_id;


            let matBatchOffset = i32(workgroup_id.z) * metadata.OVL * metadata.IVL;
            let inVecBatchOffset = i32(workgroup_id.z) * metadata.IVL;
            let outVecBatchOffset = i32(workgroup_id.z) * metadata.OVL;


            // Threadgroup in_vec cache
            let inVecBlockOffset = i32(simd_lid * 'TN * 2);

            // Thread local accumulation results
            'thread_locals

            // Block position
            var outRow = i32((workgroup_id.x * 'BM + simd_gid) * 'TM);

            // Exit simdgroup if rows out of bound
            if (outRow >= metadata.OVL) {
                return;
            }

            // Adjust tail simdgroup to ensure in bound reads
            outRow = select(metadata.OVL - 'TM, outRow, outRow + 'TM <= metadata.OVL);

            // Advance matrix
            let matOffset = matBatchOffset + outRow * metadata.IVL;
        });

        let main_tgp_load = (0..TN)
            .map(|tn| {
                wgsl! {
                    tgpMemory[inVecBlockOffset + 'tn] = inVec[inVecBatchOffset + bn + 'tn];
                }
                .into()
            })
            .collect::<WgslFragment>();

        let edge_tgp_load = (0..TN)
            .map(|tn| {
                wgsl! {
                    tgpMemory[inVecBlockOffset + 'tn] = select(inVec[inVecBatchOffset + bn + 'tn], 'dt(0.0), bn + 'tn < metadata.IVL);
                }
                .into()
            })
            .collect::<WgslFragment>();

        let load_rows = (0..TN)
            .map(|tn| {
                wgsl! {
                    vCoeff['tn] = tgpMemory[inVecBlockOffset + 'tn];
                }
                .into()
            })
            .collect::<WgslFragment>();

        let main_inter_load = (0..TN)
            .map(|tn| {
                wgsl! {
                    inter['tn] = mat[matOffset + tm * metadata.IVL + bn + 'tn];
                }
                .into()
            })
            .collect::<WgslFragment>();

        let edge_inter_load = (0..TN)
            .map(|tn| {
                wgsl! {
                    inter['tn] = mat[matOffset + tm * metadata.IVL + select(metadata.IVL - 1, bn + 'tn, bn + 'tn < metadata.IVL)];
                }
                .into()
            })
            .collect::<WgslFragment>();

        let accumulate = (0..TN)
            .map(|tn| {
                wgsl! {
                    result[tm] += f32(inter['tn] * vCoeff['tn]);
                }
                .into()
            })
            .collect::<WgslFragment>();

        let finalizer = (0..TM)
            .map(|tm| {
                if self.bias.is_some() {
                    wgsl! {
                        outVec[outVecBatchOffset + outRow + 'tm] = 'dt(result['tm]) + bias[outRow + 'tm];
                    }
                    .into()
                } else {
                    wgsl! {
                        outVec[outVecBatchOffset + outRow + 'tm] = 'dt(result['tm]);
                    }
                    .into()
                }
            })
            .collect::<WgslFragment>();

        let work_loop_inner = match self.lhs.dt() {
            DType::F32 | DType::F16 => {
                wgsl! {
                    // Load for the row
                    if (bn + 'TN <= metadata.IVL) {
                        'main_inter_load
                    } else { // Edgecase
                        'edge_inter_load
                    }

                    // Accumulate results
                    'accumulate
                }
            }
            DType::Q8_0F(_) | DType::Q8_0H(_) => {
                wgsl! {
                    let matIdx = matOffset + tm * metadata.IVL + bn;
                    inter = unpack(mat[matIdx / 4]) * scale[matIdx / 32];

                    // Accumulate results
                    'accumulate
                }
            }
            _ => unimplemented!(),
        };

        let BNTN = BN * TN;
        kernel_builder.write_main(wgsl! {
            // Loop over in_vec in blocks of SIMD_SIZE * {{TN}}
            for (var bn = i32(simd_lid * 'TN); bn < i32(metadata.IVL); bn += 'BNTN) {
                workgroupBarrier();

                // Prefetch in_vector for threadgroup use
                if (simd_gid == 0u) {
                    // Main load loop
                    if (bn + 'TN <= i32(metadata.IVL)) {
                        'main_tgp_load
                    } else { // Edgecase
                        'edge_tgp_load
                    }
                }

                workgroupBarrier();

                // Load for all rows
                'load_rows

                // Per thread work loop
                for (var tm = 0; tm < 'TM; tm++) {
                    'work_loop_inner
                }
            }

            for (var tm = 0; tm < 'TM; tm++) {
                result[tm] = subgroupAdd(result[tm]);
            }

            // Write outputs
            if (simd_lid == 0u) {
                'finalizer
            }
        });

        let x = kernel_builder.build()?;
        //println!("{}", x);
        Ok(x)
    }
}
