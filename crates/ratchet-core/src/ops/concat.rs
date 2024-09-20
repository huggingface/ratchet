use derive_new::new;
use glam::UVec4;
use half::f16;
use inline_wgsl::wgsl;

use crate::{
    gpu::BindGroupLayoutDescriptor, rvec, Array, BindingMode, BuiltIn, DType, DynKernelMetadata,
    GPUOperation, Kernel, KernelElement, KernelRenderable, KernelSource, OpGuards, Operation,
    OperationError, RVec, Scalar, Shape, StorageView, Strides, Tensor, Vec2, Vec4,
    WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
};

#[derive(new, Debug, Clone)]
pub struct Concat {
    pub inputs: RVec<Tensor>,
    pub dim: usize,
}

impl Concat {
    pub fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl KernelRenderable for ConcatKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError> {
        if inplace {
            return Err(OperationError::InplaceError("bingo".to_string()));
        }
        let arr = Array::<P>::default();
        let ConcatKernels::Standard(inner) = self;

        for i in 0..inner.inputs.len() {
            builder.register_storage(format!("X{}", i).as_str(), BindingMode::ReadOnly, arr);
        }
        builder.register_storage("Y", BindingMode::ReadWrite, arr);
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
                BuiltIn::LocalInvocationIndex,
                BuiltIn::NumWorkgroups,
                BuiltIn::WorkgroupId,
            ],
            device.compute_features().clone(),
        );
        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        kernel_builder.render_metadata(&self.metadata(dst, &self.kernel_element(dst))?);
        kernel_builder.write_offset_to_index();
        kernel_builder.write_index_to_offset();

        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let dst_offset = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (dst_offset >= metadata.dst_numel) {
                return;
            }

            var dst_index = offsetToNdIndex(dst_offset, metadata.dst_stride);
            let dim = metadata.dim;
        });

        kernel_builder.write_main(wgsl! {
            if(dst_index[dim] < metadata.cum0) {
                let src_offset = ndIndexToOffset(dst_index, metadata.x0_stride);
                Y[dst_offset] = X0[src_offset];
                return;
            }
        });

        let ConcatKernels::Standard(inner) = self;

        for i in 1..inner.inputs.len() {
            let prevcum = format!("metadata.cum{}", i - 1);
            let cum = format!("metadata.cum{}", i);
            let stride = format!("metadata.x{}_stride", i);
            let src = format!("X{}", i);

            kernel_builder.write_main(wgsl! {
                if(dst_index[dim] < 'cum) {
                    dst_index[dim] -= 'prevcum;
                    let src_offset = ndIndexToOffset(dst_index, 'stride);
                    Y[dst_offset] = 'src[src_offset];
                    return;
                }
            });
        }

        Ok(kernel_builder.build()?)
    }
}

impl Operation for Concat {
    fn name(&self) -> &'static str {
        "Concat"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let first = &self.inputs[0];
        let stacked_dim = self.inputs.iter().map(|x| x.shape()[self.dim]).sum();
        let mut output_shape = first.shape().clone();
        output_shape[self.dim] = stacked_dim;
        let output_strides = Strides::from(&output_shape);
        Ok(StorageView::new(output_shape, first.dt(), output_strides))
    }

    fn srcs(&self) -> RVec<&Tensor> {
        self.inputs.iter().collect()
    }
}

impl OpGuards for Concat {
    fn check_shapes(&self) {
        assert!(self.inputs.len() > 1);
        assert!(self.inputs.len() <= 8); //We only generate kernels for up to 8 inputs
        let first = &self.inputs[0];
        assert!(self
            .inputs
            .iter()
            .all(|x| x.rank() == first.rank() && x.rank() <= 4));
        assert!(self.inputs.iter().all(|x| self.dim < x.rank()));
        //All tensors must have same shape, sans the concatenation dimension
        for axis in 0..self.dim {
            assert!(self
                .inputs
                .iter()
                .all(|x| x.shape()[axis] == first.shape()[axis]));
        }
        for axis in (self.dim + 1)..first.rank() {
            assert!(self
                .inputs
                .iter()
                .all(|x| x.shape()[axis] == first.shape()[axis]));
        }
    }

    fn check_dtypes(&self) {
        assert!(self.inputs.iter().all(|x| x.dt() == self.inputs[0].dt()));
    }
}

pub enum ConcatKernels {
    Standard(Concat),
}

impl Kernel for ConcatKernels {
    type Metadata = DynKernelMetadata;

    fn storage_bind_group_layout(
        &self,
        _: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        let ConcatKernels::Standard(inner) = self;
        Ok(BindGroupLayoutDescriptor::nthary(inner.inputs.len()))
    }

    fn kernel_name(&self) -> String {
        match self {
            ConcatKernels::Standard(_) => "concat".to_string(),
        }
    }

    fn metadata(&self, dst: &Tensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let ConcatKernels::Standard(inner) = self;

        let original_rank = inner.inputs[0].rank();
        let promotion = 4 - original_rank;
        let input_shapes: Vec<Shape> = inner
            .inputs
            .iter()
            .map(|x| Shape::promote(x.shape().clone(), 4))
            .collect();
        let input_strides: Vec<Strides> = input_shapes.iter().map(Strides::from).collect();
        let promoted_dim = inner.dim + promotion;
        let dst_shape = Shape::promote(dst.shape().clone(), 4);
        let dst_strides = Strides::from(&dst_shape);

        let mut dyn_meta = DynKernelMetadata::new();

        let cumsum = input_shapes
            .iter()
            .map(|s| s[promoted_dim])
            .scan(0_u32, |acc, x| {
                *acc += x as u32;
                Some(*acc)
            })
            .collect::<Vec<u32>>();

        for (si, strides) in input_strides.iter().enumerate() {
            dyn_meta.add_field(format!("x{}_stride", si), UVec4::from(strides));
        }

        dyn_meta.add_field("dst_stride", UVec4::from(&dst_strides));
        dyn_meta.add_field("dst_numel", dst_shape.numel() as u32);

        for (ci, c) in cumsum.iter().enumerate() {
            dyn_meta.add_field(format!("cum{}", ci), *c);
        }

        dyn_meta.add_field("dim", promoted_dim as u32);
        Ok(dyn_meta)
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError> {
        Ok(Workload::std(dst.shape().numel(), self.kernel_element(dst)))
    }

    fn kernel_element(&self, _: &Tensor) -> KernelElement {
        KernelElement::Scalar
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
            _ => Err(OperationError::CompileError(format!(
                "Unsupported dtype {:?} or kernel element {:?}",
                dst.dt(),
                kernel_element
            ))),
        }
    }
}

impl GPUOperation for Concat {
    type KernelEnum = ConcatKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        ConcatKernels::Standard(self.clone())
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use crate::{rvec, shape, test_util::run_py_prg, Device, DeviceRequest, RVec, Tensor};

    #[derive(Debug)]
    struct ConcatProblem {
        tensors: Vec<Tensor>,
        dim: usize,
    }

    fn ground_truth(to_cat: &[&Tensor], args: &str) -> anyhow::Result<Tensor> {
        let prg = format!(
            r#"
import torch
import numpy as np
def permute(*tensors):
    numpy_tensors = []
    for t in tensors:
        numpy_tensors.append(torch.from_numpy(t))
    return np.ascontiguousarray(torch.cat(numpy_tensors, dim={}).numpy())
"#,
            args
        );
        run_py_prg(prg.to_string(), to_cat, &[], to_cat[0].dt())
    }

    fn run_concat_trial(prob: ConcatProblem, device: Device) -> anyhow::Result<()> {
        let ConcatProblem { tensors, dim } = prob;

        let arg_str = format!("{}", dim);
        let ground = ground_truth(
            tensors.iter().collect::<Vec<&Tensor>>().as_slice(),
            arg_str.as_str(),
        )?;

        for t in tensors.iter() {
            t.to(&device)?;
        }
        let t_rvec = RVec::from(tensors);
        let ours = Tensor::cat(t_rvec, dim)?.resolve()?;
        let result = ours.to(&Device::CPU)?;
        println!("Ground: {:?}\n", ground);
        println!("Ours: {:?}", result);
        ground.all_close(&result, 1e-5, 1e-5)?;
        Ok(())
    }

    #[test]
    fn test_concat_gpu() {
        let t0 = Tensor::randn::<f32>(shape![4, 2, 50, 128], Device::CPU);
        let t1 = Tensor::randn::<f32>(shape![4, 2, 13, 128], Device::CPU);
        let t2 = Tensor::randn::<f32>(shape![4, 2, 77, 128], Device::CPU);
        let t3 = Tensor::randn::<f32>(shape![4, 2, 55, 128], Device::CPU);
        let t4 = Tensor::randn::<f32>(shape![4, 2, 11, 128], Device::CPU);

        let dim = 2;
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_concat_trial(
            ConcatProblem {
                tensors: vec![t0, t1, t2, t3, t4],
                dim,
            },
            device,
        )
        .unwrap();
    }

    #[test]
    fn test_concat_cpu() {
        let t0 = Tensor::randn::<f32>(shape![4, 2, 50, 128], Device::CPU);
        let t1 = Tensor::randn::<f32>(shape![4, 2, 13, 128], Device::CPU);
        let t2 = Tensor::randn::<f32>(shape![4, 2, 77, 128], Device::CPU);
        let t3 = Tensor::randn::<f32>(shape![4, 2, 55, 128], Device::CPU);
        let t4 = Tensor::randn::<f32>(shape![4, 2, 11, 128], Device::CPU);

        let dim = 2;
        let device = Device::request_device(DeviceRequest::CPU).unwrap();
        run_concat_trial(
            ConcatProblem {
                tensors: vec![t0, t1, t2, t3, t4],
                dim,
            },
            device,
        )
        .unwrap();
    }
}
