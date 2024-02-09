mod broadcast;
mod permute;
mod slice;

pub use broadcast::Broadcast;
pub use permute::Permute;
pub use slice::Slice;

use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, WorkgroupCount},
    rvec, wgc, KernelElement, MetaOperation, OpMetadata, OperationError, RVec, Shape, Strides,
    Tensor,
};
use glam::UVec4;

#[derive(Debug, Clone)]
pub enum ReindexOp {
    Permute(Permute),
    Slice(Slice),
    Broadcast(Broadcast),
}

impl ReindexOp {
    pub fn kernel_name(&self) -> &'static str {
        match self {
            ReindexOp::Permute(_) => "permute",
            ReindexOp::Slice(_) => "slice",
            ReindexOp::Broadcast(_) => "broadcast",
        }
    }
}

#[derive(new, Debug, Clone)]
pub struct Reindex {
    input: Tensor,
    op: ReindexOp,
}

impl Reindex {
    pub fn name(&self) -> &'static str {
        self.op.kernel_name()
    }

    pub fn op(&self) -> &ReindexOp {
        &self.op
    }
}

#[derive(Debug, ShaderType)]
pub struct ReindexMeta {
    src_shape: glam::UVec4,
    dst_shape: glam::UVec4,
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
        let padder = |mut shape: Shape| {
            shape.left_pad_to(1, 4);
            let strides = Strides::from(&shape);
            (shape, strides)
        };
        let (input_shape, input_strides) = padder(self.input.shape().clone());
        let (dst_shape, dst_strides) = padder(dst.shape().clone());

        let src_stride = UVec4::try_from(&input_strides).unwrap();
        let dst_stride = UVec4::try_from(&dst_strides).unwrap();
        let src_numel = input_shape.numel() as u32;
        let dst_numel = dst_shape.numel() as u32;

        let src_shape = UVec4::try_from(&input_shape).unwrap();
        let dst_shape = UVec4::try_from(&dst_shape).unwrap();

        //TODO: move this to the inner ops
        //TODO: this is incredibly bad
        let permute = match &self.op {
            ReindexOp::Permute(p) => {
                let dims = p.promote();
                dims.iter().map(|&d| d as u32).collect::<Vec<_>>()
            }
            _ => vec![0, 0, 0, 0],
        };
        let src_offsets = match &self.op {
            ReindexOp::Slice(s) => {
                let starts: [u32; 4] = (0..4)
                    .map(|i| (s.indices().get(i).map(|index| index.start).unwrap_or(0)) as u32)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                starts
            }
            _ => [0, 0, 0, 0],
        };
        let permute = glam::UVec4::new(permute[0], permute[1], permute[2], permute[3]);
        let src_offsets = glam::UVec4::new(
            src_offsets[0],
            src_offsets[1],
            src_offsets[2],
            src_offsets[3],
        );
        Ok(ReindexMeta {
            src_shape,
            dst_shape,
            src_stride,
            dst_stride,
            src_numel,
            dst_numel,
            permute,
            src_offsets,
        })
    }
}
