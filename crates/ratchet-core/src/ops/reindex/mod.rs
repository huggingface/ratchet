mod permute;
mod slice;
pub use permute::Permute;
pub use slice::Slice;

use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, WorkgroupCount},
    rvec, wgc, KernelElement, MetaOperation, OpMetadata, OperationError, RVec, Tensor,
};

#[cfg(test)]
use test_strategy::Arbitrary;

#[cfg_attr(test, derive(Arbitrary))]
#[derive(Debug, Clone)]
pub enum ReindexOp {
    Permute(Permute),
    Slice(Slice),
}

impl ReindexOp {
    pub fn kernel_name(&self) -> &'static str {
        match self {
            ReindexOp::Permute(_) => "permute",
            ReindexOp::Slice(_) => "slice",
        }
    }
}

#[derive(new, Debug, Clone)]
pub struct Reindex {
    input: Tensor,
    op: ReindexOp,
}

impl Reindex {
    pub fn op(&self) -> &ReindexOp {
        &self.op
    }
}

#[derive(Debug, ShaderType)]
pub struct ReindexMeta {
    src_stride: glam::UVec4,
    dst_stride: glam::UVec4,
    src_numel: u32,
    dst_numel: u32,
    //"Optional" fields below (if not present, they are set to 0)
    permute: glam::UVec4,
    src_offsets: glam::UVec4,
}

impl OpMetadata for ReindexMeta {}

impl MetaOperation for Reindex {
    type Meta = ReindexMeta;

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input]
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        //TODO: add support for Vec4
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<WorkgroupCount, OperationError> {
        let numel = dst.shape().numel();
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

    fn kernel_name(&self) -> &'static str {
        self.op.kernel_name()
    }

    fn metadata(
        &self,
        dst: &Tensor,
        _kernel_element: &KernelElement,
    ) -> Result<Self::Meta, OperationError> {
        //TODO: this is garbage
        let src_stride = glam::UVec4::try_from(self.input.strides()).unwrap();
        let dst_stride = glam::UVec4::try_from(dst.strides()).unwrap();
        let src_numel = self.input.shape().numel() as u32;
        let dst_numel = self.input.shape().numel() as u32;
        let permute = match &self.op {
            ReindexOp::Permute(p) => p.dims.iter().map(|&d| d as u32).collect::<Vec<_>>(),
            _ => vec![0, 0, 0, 0],
        };
        let src_offsets = match &self.op {
            ReindexOp::Slice(s) => s
                .indices()
                .iter()
                .map(|r| r.start as u32)
                .collect::<Vec<_>>(),
            _ => vec![0, 0, 0, 0],
        };
        let permute = glam::UVec4::new(permute[0], permute[1], permute[2], permute[3]);
        let src_offsets = glam::UVec4::new(
            src_offsets[0],
            src_offsets[1],
            src_offsets[2],
            src_offsets[3],
        );
        let meta = ReindexMeta {
            src_stride,
            dst_stride,
            src_numel,
            dst_numel,
            permute,
            src_offsets,
        };
        println!("meta: {:?}", meta);
        Ok(meta)
    }
}
