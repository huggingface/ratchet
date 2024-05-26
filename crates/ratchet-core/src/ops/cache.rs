use derive_new::new;
use encase::ShaderType;
use glam::UVec4;
use wgpu::BindGroupLayoutEntry;

use crate::{
    gpu::{BindGroupLayoutDescriptor, BindGroupLayoutEntryExt, CpuUniform, WorkgroupCount},
    rvec, wgc, KernelElement, KernelKey, MetaOperation, OpGuards, OpMetadata, Operation,
    OperationError, RVec, Shape, StorageView, Strides, Tensor,
};

/// # Cache
///
/// Custom operator used for KV caching. Custom operator to support quantized KV caching.
///
/// Takes in 3 arguments:
/// 1. Cache, large partially filled tensors. E.g [1, 512, 1024], with [1, 5, 1024] filled.
/// 2. Source, new K or V tensor, e.g [1, 1, 1024]
/// 3. offset, where to start the write in the cache tensor, e.g [1, 5, 1024], [1, 1, 1024], offset = 5 -> [1, 6, 1024]
#[derive(new, Debug, Clone)]
pub struct Cache {
    cache: Tensor,
    source: Tensor,
    dim: usize,
    offset: usize,
}

#[derive(Debug, derive_new::new, ShaderType)]
pub struct CacheMeta {
    cache_stride: glam::UVec4,
    source_stride: glam::UVec4,
    dst_stride: glam::UVec4,
    dst_numel: u32,
    cum0: u32,
    cum1: u32,
    dim: u32,
}

impl OpMetadata for CacheMeta {}

impl OpGuards for Cache {
    fn check_shapes(&self) {
        assert!(self.cache.rank() >= 3);
        assert!(self.offset <= self.cache.shape()[self.dim]);
    }

    fn check_dtypes(&self) {
        assert_eq!(self.cache.dt(), self.source.dt());
    }
}

impl Operation for Cache {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let mut result_shape = self.cache.shape().clone();
        result_shape[self.dim] = self.offset + self.source.shape()[self.dim];
        let result_strides = Strides::from(&result_shape);
        Ok(StorageView::new(
            result_shape,
            self.cache.dt(),
            result_strides,
        ))
    }
}

impl MetaOperation for Cache {
    fn kernel_name(&self) -> String {
        "cache".to_string()
    }

    fn supports_inplace(&self) -> bool {
        false
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.cache, &self.source]
    }

    fn kernel_key(&self, _: bool, dst: &Tensor) -> KernelKey {
        KernelKey::new(format!("cache_{}", self.kernel_element(dst).as_str()))
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
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
        let wgc = wgc![x_groups as _, y_groups as _, 1];
        Ok(wgc)
    }

    fn storage_bind_group_layout(
        &self,
        _: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        Ok(BindGroupLayoutDescriptor {
            entries: rvec![
                BindGroupLayoutEntry::compute_storage_buffer(0, false),
                BindGroupLayoutEntry::compute_storage_buffer(1, true),
                BindGroupLayoutEntry::compute_storage_buffer(2, false)
            ],
        })
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        dst: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        let original_rank = self.cache.rank();
        let promotion = 4 - original_rank;
        let promoted_dim = self.dim + promotion;

        let cache_shape = Shape::promote(self.cache.shape().clone(), 4);
        let cache_strides = Strides::from(&cache_shape);

        let source_shape = Shape::promote(self.source.shape().clone(), 4);
        let source_strides = Strides::from(&source_shape);

        let dst_shape = Shape::promote(dst.shape().clone(), 4);
        let dst_strides = Strides::from(&dst_shape);

        let cum0 = self.offset as u32;
        let cum1 = cum0 + source_shape[promoted_dim] as u32;

        let meta = CacheMeta {
            cache_stride: UVec4::from(&cache_strides),
            source_stride: UVec4::from(&source_strides),
            dst_stride: UVec4::from(&dst_strides),
            dst_numel: dst_shape.numel() as u32,
            cum0,
            cum1,
            dim: promoted_dim as u32,
        };

        Ok(uniform.write(&meta)?)
    }
}

#[cfg(test)]
mod tests {
    use crate::{rvec, shape, Device, DeviceRequest, Tensor};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    #[test]
    fn test_cache() -> anyhow::Result<()> {
        let device = GPU_DEVICE.with(|d| d.clone());
        let populated = 2;
        //Create cache with 2 populated entries, and 14 blank entries
        let mut dst0 = Tensor::randn::<f32>(shape![1, 2, populated, 16], Device::CPU);
        println!("PREVIOUS CACHE\n {:?}\n", dst0.to_ndarray_view::<f32>());
        dst0 = dst0.to(&device)?;
        let dst1 = Tensor::zeros::<f32>(&shape![1, 2, 4, 16], &device);
        let cur_cache = Tensor::cat(rvec![dst0.clone(), dst1], 2)?.resolve()?;

        //This is the k or v vector we write
        let mut src = Tensor::randn::<f32>(shape![1, 2, 1, 16], Device::CPU);
        println!("SRC \n {:?}\n", src.to_ndarray_view::<f32>());
        src = src.to(&device)?;

        //The result should be the concatenation of the cache and the source
        let ground_truth = Tensor::cat(rvec![dst0.clone(), src.clone()], 2)?
            .resolve()?
            .to(&Device::CPU)?;

        let dim = 2;
        let b = cur_cache.clone().cache(src, dim, populated)?.resolve()?;

        let cur_cache_cpu = cur_cache.to(&Device::CPU)?;
        println!(
            "CACHE RESULT \n{:?}\n",
            cur_cache_cpu.to_ndarray_view::<f32>()
        );

        let result = b.to(&Device::CPU)?;
        println!("RESULT \n{:?}", result.to_ndarray_view::<f32>());

        result.all_close(&ground_truth, 1e-5, 1e-5).unwrap();
        Ok(())
    }
}
