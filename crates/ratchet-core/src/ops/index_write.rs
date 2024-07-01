use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use ratchet_macros::WgslMetadata;

use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform},
    rvec, Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel, KernelElement, KernelSource,
    OpGuards, Operation, OperationError, RVec, Scalar, Shape, StorageView, Strides, Tensor, Vec2,
    Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
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
        _: bool,
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
                BuiltIn::LocalInvocationIndex,
                BuiltIn::NumWorkgroups,
                BuiltIn::WorkgroupId,
            ],
            device.compute_features().clone(),
        );
        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        kernel_builder.render_metadata::<IndexWriteMeta>();
        kernel_builder.write_index_to_offset();

        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let thread_offset = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
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

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.dst, &self.src]
    }

    fn supports_inplace(&self) -> bool {
        true
    }
}

impl GPUOperation for IndexWrite {
    type KernelEnum = IndexWriteKernels;

    fn select_kernel(self) -> Self::KernelEnum {
        IndexWriteKernels::Standard(self)
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
}

pub enum IndexWriteKernels {
    Standard(IndexWrite),
}

impl Kernel for IndexWriteKernels {
    type Metadata = IndexWriteMeta;

    fn metadata(
        &self,
        dst: &Tensor,
        kernel_element: &KernelElement,
    ) -> Result<Self::Metadata, OperationError> {
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

        Ok(IndexWriteMeta {
            dst_strides: glam::UVec4::from(&dst_strides),
            src_numel: src_shape.numel() as u32,
            write_start: start.into(),
        })
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError> {
        Ok(Workload::std(
            self.src.shape().numel(),
            KernelElement::Scalar,
        ))
    }

    fn kernel_element(&self, dst: &Tensor) -> KernelElement {
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
