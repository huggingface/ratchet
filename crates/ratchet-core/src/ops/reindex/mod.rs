mod broadcast;
mod permute;
mod slice;

pub use broadcast::Broadcast;
pub use permute::Permute;
pub use slice::Slice;

use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, KernelElement, MetaOperation, OpMetadata, OperationError, RVec, Shape, Strides,
    Tensor,
};
use glam::UVec4;

#[derive(new, Debug, Clone)]
pub enum Reindex {
    Permute(Permute),
    Slice(Slice),
    Broadcast(Broadcast),
}

#[derive(Debug, ShaderType)]
pub struct ReindexMeta {
    src_shape: glam::UVec4,
    dst_shape: glam::UVec4,
    src_stride: glam::UVec4,
    dst_stride: glam::UVec4,
    src_numel: u32,
    dst_numel: u32,
    //"Optional" fields below (if not present, they are set to 0) this is dumb
    permute: glam::UVec4,
    src_offsets: glam::UVec4,
}

impl OpMetadata for ReindexMeta {}

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

    fn kernel_element(&self, dst: &Tensor) -> KernelElement {
        match self {
            Reindex::Broadcast(b) => {
                let src_numel = b.src.shape().numel();
                let src_outer = b.src.shape()[b.src.rank() - 1];
                let dst_outer = dst.shape()[dst.rank() - 1];

                if (src_numel == 1 && dst_outer % 4 == 0)
                    || (src_outer % 4 == 0 && dst_outer % 4 == 0)
                {
                    //Special case for src_numel == 1
                    KernelElement::Vec4
                } else if src_outer % 2 == 0 && dst_outer % 2 == 0 {
                    KernelElement::Vec2
                } else {
                    KernelElement::Scalar
                }
            }
            _ => KernelElement::Scalar,
        }
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<WorkgroupCount, OperationError> {
        let mut numel = dst.shape().numel();
        match self.kernel_element(dst) {
            KernelElement::Vec4 => {
                numel /= 4;
            }
            KernelElement::Vec2 => {
                numel /= 2;
            }
            _ => {}
        }

        let x_groups = WorkgroupCount::div_ceil(numel as _, 64);
        let (x_groups, y_groups) = if x_groups > WorkgroupCount::MAX_WGS_PER_DIM {
            let y_groups = WorkgroupCount::div_ceil(x_groups, WorkgroupCount::MAX_WGS_PER_DIM);
            (WorkgroupCount::MAX_WGS_PER_DIM, y_groups)
        } else {
            (x_groups, 1)
        };
        Ok(wgc![x_groups as _, y_groups as _, 1])
    }

    fn storage_bind_group_layout(
        &self,
        _inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        Ok(BindGroupLayoutDescriptor::unary())
    }

    fn kernel_key(&self, _: bool, dst: &Tensor) -> String {
        let ke = self.kernel_element(dst);
        let op_key = match self {
            Reindex::Permute(_) => "permute",
            Reindex::Slice(_) => "slice",
            Reindex::Broadcast(b) => {
                let src_numel = b.src.shape().numel();
                let dst_outer = dst.shape()[dst.rank() - 1];
                if src_numel == 1 && dst_outer % 4 == 0 {
                    "broadcast_single"
                } else {
                    "broadcast"
                }
            }
        };

        format!("{}_{}", op_key, ke.as_str())
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        dst: &Tensor,
        ke: &KernelElement,
    ) -> Result<u64, OperationError> {
        //This is gross
        let srcs = self.srcs();
        let src = srcs.first().unwrap();
        let mut src_shape = Shape::promote(src.shape().clone(), 4);
        let mut dst_shape = Shape::promote(dst.shape().clone(), 4);

        let src_numel = src_shape.numel() as u32;
        let dst_numel = dst_shape.numel() as u32;

        if matches!(self, Reindex::Broadcast(_)) {
            src_shape[3] /= ke.as_size();
            dst_shape[3] /= ke.as_size();
        }

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
        let permute = glam::UVec4::from(permute);
        let src_offsets = glam::UVec4::from(src_offsets);
        let meta = ReindexMeta {
            src_shape,
            dst_shape,
            src_stride,
            dst_stride,
            src_numel,
            dst_numel,
            permute,
            src_offsets,
        };
        Ok(uniform.write(&meta)?)
    }
}
