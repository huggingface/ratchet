use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use ratchet_macros::WgslMetadata;

use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, wgs, Array, BindingMode, BuiltIn, DType, KernelElement, KernelKey, KernelSource,
    MetaOperation, OpGuards, Operation, OperationError, RVec, Scalar, Shape, StorageView, Strides,
    Tensor, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
};

#[derive(new, Debug, Clone)]
pub struct IndexWrite {
    dst: Tensor,
    src: Tensor,
    write_start: RVec<usize>,
}

impl IndexWrite {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError> {
        let arr = Array::<P>::default();
        builder.register_storage("D", BindingMode::ReadWrite, arr);
        builder.register_storage("S", BindingMode::ReadOnly, arr);
        builder.register_uniform();
        Ok(())
    }

    fn build_index_write<P: WgslPrimitive>(
        &self,
        inplace: bool,
        _: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let device = self.src.device().try_gpu().unwrap();
        let mut kernel_builder = WgslKernelBuilder::new(
            workgroup_size.clone(),
            rvec![
                BuiltIn::GlobalInvocationId,
                BuiltIn::LocalInvocationId,
                BuiltIn::WorkgroupId,
            ],
            device.compute_features().clone(),
        );
        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        kernel_builder.write_metadata::<IndexWriteMeta>();
        kernel_builder.write_index_to_offset();

        kernel_builder.write_main(wgsl! {
            let x_offset = group_id.x * 64u;
            let thread_offset = (group_id.y * num_groups.x * 64u) + x_offset + local_index;
            if (thread_offset >= metadata.src_numel) {
                return;
            }
            let offset_index = ndIndexToOffset(metadata.write_start, metadata.dst_strides);
            D[offset_index + thread_offset] = S[thread_offset];
        });

        Ok(kernel_builder.build()?)
    }
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct IndexWriteMeta {
    dst_strides: glam::UVec4,
    src_numel: u32,
    write_start: glam::UVec4,
}

impl OpGuards for IndexWrite {
    fn check_shapes(&self) {}

    fn check_dtypes(&self) {}
}

impl Operation for IndexWrite {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        Ok(self.dst.storage_view().clone())
    }
}

impl MetaOperation for IndexWrite {
    fn kernel_name(&self) -> String {
        "index_write".to_string()
    }

    fn supports_inplace(&self) -> bool {
        true
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.dst, &self.src]
    }

    fn kernel_key(&self, _: bool, dst: &Tensor) -> KernelKey {
        KernelKey::new(format!("index_write_{}", self.kernel_element(dst).as_str()))
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = self.kernel_element(dst);
        match (self.src.dt(), &kernel_element) {
            (DType::F32, KernelElement::Scalar) => {
                self.build_index_write::<Scalar<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec2) => {
                self.build_index_write::<Vec2<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec4) => {
                self.build_index_write::<Vec4<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Scalar) => {
                self.build_index_write::<Scalar<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec2) => {
                self.build_index_write::<Vec2<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec4) => {
                self.build_index_write::<Vec4<f16>>(inplace, dst, workgroup_size)
            }
            _ => Err(OperationError::CompileError(format!(
                "Unsupported dtype {:?} or kernel element {:?}",
                self.src.dt(),
                kernel_element
            ))),
        }
    }

    fn calculate_dispatch(&self, _: &Tensor) -> Result<Workload, OperationError> {
        let workgroup_size = wgs![8, 8, 1];
        let numel = self.src.shape().numel();
        let x_groups = WorkgroupCount::div_ceil(numel as _, workgroup_size.product() as _);
        let (x_groups, y_groups) = if x_groups > WorkgroupCount::MAX_WGS_PER_DIM {
            let y_groups = WorkgroupCount::div_ceil(x_groups, WorkgroupCount::MAX_WGS_PER_DIM);
            (WorkgroupCount::MAX_WGS_PER_DIM, y_groups)
        } else {
            (x_groups, 1)
        };
        Ok(Workload {
            workgroup_count: wgc![x_groups as _, y_groups as _, 1],
            workgroup_size,
        })
    }

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if !inplace {
            panic!("IndexWrite only supports inplace operation");
        }
        Ok(BindGroupLayoutDescriptor::binary_inplace())
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        _: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        let padder = |mut shape: Shape| {
            shape.left_pad_to(1, 4);
            let strides = Strides::from(&shape);
            (shape, strides)
        };
        let (_, dst_strides) = padder(self.dst.shape().clone());
        let (src_shape, _) = padder(self.src.shape().clone());

        let mut start = [0u32; 4];
        let offset = 4 - self.write_start.len();
        for (i, &s) in self.write_start.iter().enumerate() {
            start[i + offset] = s as u32;
        }

        let meta = IndexWriteMeta {
            dst_strides: glam::UVec4::from(&dst_strides),
            src_numel: src_shape.numel() as u32,
            write_start: start.into(),
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
    fn test_index_write() {
        let device = GPU_DEVICE.with(|d| d.clone());
        let dst = Tensor::from_data(vec![1., 2., 3., 4., 5., 6.], shape![3, 2], device.clone());
        let src = Tensor::from_data(vec![7., 8.], shape![1, 2], device.clone());
        let write_start = rvec![2, 0];
        let b = dst
            .index_write(src, write_start)
            .unwrap()
            .resolve()
            .unwrap();

        let result = b.to(&Device::CPU).unwrap();

        let ground_truth =
            Tensor::from_data(vec![1., 2., 3., 4., 7., 8.], shape![3, 2], Device::CPU);
        println!("result: {:?}", result);
        println!("ground_truth: {:?}", ground_truth);
        ground_truth.all_close(&result, 1e-8, 1e-8).unwrap();
    }
}
