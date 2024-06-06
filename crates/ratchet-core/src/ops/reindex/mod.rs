mod broadcast;
mod permute;
mod slice;

pub use broadcast::Broadcast;
use half::f16;
pub use permute::Permute;
use ratchet_macros::WgslMetadata;
pub use slice::Slice;

use derive_new::new;
use encase::ShaderType;
use inline_wgsl::wgsl;

use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, wgs, Array, BindingMode, BuiltIn, DType, KernelElement, KernelKey, KernelSource,
    MetaOperation, OperationError, RVec, Scalar, Shape, Strides, Tensor, WgslKernelBuilder,
    WgslPrimitive, WorkgroupSize, Workload,
};
use glam::UVec4;

#[derive(new, Debug, Clone)]
pub enum Reindex {
    Permute(Permute),
    Slice(Slice),
    Broadcast(Broadcast),
}

impl Reindex {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        let arr = Array::<P>::default();
        builder.register_storage("X", BindingMode::ReadOnly, arr);
        builder.register_storage("Y", BindingMode::ReadWrite, arr);
        builder.register_uniform();
        Ok(())
    }

    fn build_reindex<P: WgslPrimitive>(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let device = dst.device().try_gpu().unwrap();
        let mut kernel_builder = WgslKernelBuilder::new(
            workgroup_size.clone(),
            rvec![
                BuiltIn::LocalInvocationIndex,
                BuiltIn::WorkgroupId,
                BuiltIn::NumWorkgroups,
            ],
            device.compute_features().clone(),
        );
        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        //In future this metadata could be dynamic
        kernel_builder.write_metadata::<ReindexMeta>();

        let n = P::W;

        //Custom with slice offset
        kernel_builder.write_global(wgsl! {
            //Converts 4D index into 1D offset
            fn ndIndexToOffset(index: vec4<u32>, src_offsets: vec4<u32>, stride: vec4<u32>) -> u32 {
                var offset: u32 = 0u;
                offset = dot(index + src_offsets, stride);
                return offset;
            }
        });
        kernel_builder.write_offset_to_index();

        kernel_builder.write_main(wgsl! {
            //Dispatch 1 thread per output element
            //dst_offset is index into the output buffer (1D)
            let x_offset = workgroup_id.x * 64u;
            var dst_offset = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (dst_offset >= metadata.dst_numel / 'n) {
                return;
            }

            //Convert 1D offset into 4D index
            let dst_index = offsetToNdIndex(dst_offset, metadata.dst_stride);

        });

        let body = match self {
            Reindex::Permute(_) => wgsl! {
                var src_index = vec4<u32>(0u);
                src_index[metadata.perm[0]] = dst_index[0];
                src_index[metadata.perm[1]] = dst_index[1];
                src_index[metadata.perm[2]] = dst_index[2];
                src_index[metadata.perm[3]] = dst_index[3];
            },
            Reindex::Slice(_) => wgsl! { var src_index = dst_index; },
            Reindex::Broadcast(_) => wgsl! {
                // Broadcasting is valid if dims are equal, or if one of the dims is 1
                var src_index = select(dst_index, vec4<u32>(0u), metadata.src_shape == vec4<u32>(1u));
            },
        };
        kernel_builder.write_main(body);

        kernel_builder.write_main(wgsl! {
            //Convert 4D index into 1D offset
            let src_offset = ndIndexToOffset(src_index, metadata.src_offsets, metadata.src_stride);

            //Read from input buffer and write to output buffer
            Y[dst_offset] = X[src_offset];
        });
        Ok(kernel_builder.build()?)
    }
}

#[derive(Debug, ShaderType, WgslMetadata)]
pub struct ReindexMeta {
    src_shape: glam::UVec4,
    dst_shape: glam::UVec4,
    src_stride: glam::UVec4,
    dst_stride: glam::UVec4,
    src_numel: u32,
    dst_numel: u32,
    //"Optional" fields below (if not present, they are set to 0) this is dumb
    perm: glam::UVec4,
    src_offsets: glam::UVec4,
}

impl MetaOperation for Reindex {
    fn kernel_name(&self) -> String {
        match self {
            Reindex::Permute(_) => "permute".to_string(),
            Reindex::Slice(_) => "slice".to_string(),
            Reindex::Broadcast(_) => "broadcast".to_string(),
        }
    }

    fn srcs(&self) -> RVec<&Tensor> {
        match self {
            Reindex::Permute(p) => rvec![&p.src],
            Reindex::Slice(s) => rvec![&s.src],
            Reindex::Broadcast(b) => rvec![&b.src],
        }
    }

    fn kernel_element(&self, _: &Tensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError> {
        let workgroup_size = wgs![8, 8, 1];
        let numel = dst.shape().numel();
        let x_groups = WorkgroupCount::div_ceil(numel as _, workgroup_size.product() as _);
        let (x_groups, y_groups) = if x_groups > WorkgroupCount::MAX_WGS_PER_DIM {
            let y_groups = WorkgroupCount::div_ceil(x_groups, WorkgroupCount::MAX_WGS_PER_DIM);
            (WorkgroupCount::MAX_WGS_PER_DIM, y_groups)
        } else {
            (x_groups, 1)
        };
        Ok(Workload {
            workgroup_size,
            workgroup_count: wgc![x_groups as _, y_groups as _, 1],
        })
    }

    fn storage_bind_group_layout(
        &self,
        _inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        Ok(BindGroupLayoutDescriptor::unary())
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        dst: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        //This is gross
        let srcs = self.srcs();
        let src = srcs.first().unwrap();
        let src_shape = Shape::promote(src.shape().clone(), 4);
        let dst_shape = Shape::promote(dst.shape().clone(), 4);

        let src_numel = src_shape.numel() as u32;
        let dst_numel = dst_shape.numel() as u32;

        let src_strides = Strides::from(&src_shape);
        let dst_strides = Strides::from(&dst_shape);

        let src_stride = UVec4::from(&src_strides);
        let dst_stride = UVec4::from(&dst_strides);

        let src_shape = UVec4::from(&src_shape);
        let dst_shape = UVec4::from(&dst_shape);

        //TODO: move this to the inner ops
        //TODO: this is incredibly bad
        let permute = match &self {
            Reindex::Permute(p) => {
                let dims = p.promote();
                let vdims = dims.iter().map(|&d| d as u32).collect::<Vec<_>>();
                vdims.try_into().unwrap()
            }
            _ => [0, 0, 0, 0],
        };
        let src_offsets = match &self {
            Reindex::Slice(s) => {
                let starts = s.indices().iter().map(|i| i.start).collect::<Vec<_>>();
                let mut offsets = [0; 4];
                let offset = 4 - starts.len();
                for (i, &start) in starts.iter().enumerate() {
                    offsets[i + offset] = start as u32;
                }
                offsets
            }
            _ => [0, 0, 0, 0],
        };
        let perm = glam::UVec4::from(permute);
        let src_offsets = glam::UVec4::from(src_offsets);
        let meta = ReindexMeta {
            src_shape,
            dst_shape,
            src_stride,
            dst_stride,
            src_numel,
            dst_numel,
            perm,
            src_offsets,
        };
        Ok(uniform.write(&meta)?)
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = self.kernel_element(dst);
        match (dst.dt(), &kernel_element) {
            (DType::F32, KernelElement::Scalar) => {
                self.build_reindex::<Scalar<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Scalar) => {
                self.build_reindex::<Scalar<f16>>(inplace, dst, workgroup_size)
            }
            _ => Err(OperationError::CompileError(format!(
                "Unsupported dtype {:?} or kernel element {:?}",
                dst.dt(),
                kernel_element
            ))),
        }
    }
}
