use derive_new::new;
use encase::ShaderType;
use half::f16;
use ratchet_macros::WgslMetadata;

use crate::{
    gpu::{dtype::WgslDType, BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, Array, BuiltIn, DType, DeviceFeatures, KernelElement, MetaOperation, OpGuards,
    OpMetadata, Operation, OperationError, RVec, RenderFragment, Scalar, StorageView, Tensor, Vec2,
    Vec4, WgslFragment, WgslKernel, WgslKernelBuilder, WgslPrimitive, WorkgroupSize,
};

#[derive(new, Debug, Clone)]
pub struct Softmax {
    input: Tensor,
    dim: usize,
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct SoftmaxMeta {
    M: u32,
    N: u32,
    ND2: u32,
    ND4: u32,
}

impl OpMetadata for SoftmaxMeta {}

impl OpGuards for Softmax {
    fn check_shapes(&self) {
        let input = &self.input;
        assert!(input.rank() >= 2);
        assert!(self.dim < input.rank());
    }

    fn check_dtypes(&self) {
        let input = &self.input;
        assert!(input.dt().is_float());
    }
}

impl Softmax {
    pub fn render(&self, inplace: bool, dst: &Tensor, workgroup_size: WorkgroupSize) -> WgslKernel {
        let kernel_element = self.kernel_element(dst);
        match (self.input.dt(), kernel_element) {
            (DType::F32, KernelElement::Scalar) => {
                self.render_softmax::<Scalar<f32>, _, 1>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec2) => {
                self.render_softmax::<Vec2<f32>, _, 2>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec4) => {
                self.render_softmax::<Vec4<f32>, _, 4>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Scalar) => {
                self.render_softmax::<Scalar<f16>, _, 1>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec2) => {
                self.render_softmax::<Vec2<f16>, _, 2>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec4) => {
                self.render_softmax::<Vec4<f16>, _, 4>(inplace, dst, workgroup_size)
            }
            _ => panic!("Unsupported dtype"),
        }
    }

    fn render_softmax<P: WgslPrimitive<T, N>, T: WgslDType + num_traits::Float, const N: usize>(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: WorkgroupSize,
    ) -> WgslKernel {
        let device = self.input.device().try_gpu().unwrap();
        let mut kernel_builder = WgslKernelBuilder::new(device.compute_features().clone());
        let bindings = self.storage_bind_group_layout(inplace).unwrap();
        let bind_vars = self.bindvars::<P, T, N>(inplace, dst);

        kernel_builder.write_bindings(&bindings, bind_vars);
        kernel_builder.write_fragment(SoftmaxMeta::render());
        kernel_builder.shared_memory::<P, T, N>("smem", workgroup_size.x as usize);
        kernel_builder.workgroup_var::<Scalar<T>, _, 1>("maximum");
        kernel_builder.workgroup_var::<Scalar<T>, _, 1>("sum");
        kernel_builder.constant("BLOCK_SIZE", Scalar::<u32>::new(workgroup_size.x));
        kernel_builder.constant("minFloat", Scalar::<T>::new(T::from(-65500).unwrap()));

        //kernel_builder.push_op(Reduce { input: self.input, dim: self.dim, kind: MAX });
        //kernel_builder.push_op(Reduce { input: self.input, dim: self.dim, kind: SUM });

        let reduce_funcs = r#"
fn block_sum(index: u32, stride: u32) {
    if index < stride {
        smem[index] += smem[index + stride];
    }
    workgroupBarrier();
}

fn block_max(index: u32, stride: u32) {
    if index < stride {
        smem[index] = max(smem[index], smem[index + stride]);
    }
    workgroupBarrier();
}
    "#
        .to_string();

        kernel_builder.write_fragment(reduce_funcs.into());

        let accessor = P::render_type();
        let reduce_len = match N {
            1 => "metadata.N",
            2 => "metadata.ND2",
            4 => "metadata.ND4",
            _ => panic!("Invalid dimension"),
        };

        kernel_builder.write_main(
            workgroup_size,
            &[
                BuiltIn::GlobalInvocationId,
                BuiltIn::LocalInvocationId,
                BuiltIn::WorkgroupId,
            ],
        );

        let indexing = format!(
            r#"
    let batch_stride = workgroup_id.y * metadata.M * {reduce_len}; 
    let row_start = batch_stride + workgroup_id.x * {reduce_len}; 
    let index = local_invocation_id.x;
    "#
        );

        kernel_builder.write_fragment(indexing.into());

        let mut reduce_max = format!(
            r#"
smem[index] = {accessor}(minFloat);
for (var i: u32 = index; i < {reduce_len}; i += BLOCK_SIZE) {{
    smem[index] = max(smem[index], X[row_start + i]); 
}}
workgroupBarrier();
"#
        );

        for i in (0..=6).rev().map(|x| 2u32.pow(x)) {
            reduce_max.push_str(&format!(
                r#"
block_max(index, {i}u);"#,
            ));
        }

        reduce_max.push_str(
            r#"
if index == 0u {{
    "#,
        );

        match N {
            1 => reduce_max.push_str("maximum = smem[0];"),
            2 => reduce_max.push_str("maximum = max(smem[0].x, smem[0].y);"),
            4 => reduce_max
                .push_str("maximum = max(smem[0].x, max(smem[0].y, max(smem[0].z, smem[0].w)));"),
            _ => panic!("Invalid dimension"),
        }

        reduce_max.push_str(
            r#"
}}
workgroupBarrier();
"#,
        );

        kernel_builder.write_fragment(reduce_max.into());

        let mut reduce_sum = format!(
            r#"
smem[index] = {accessor}(0.0h);
for (var i: u32 = index; i < {reduce_len}; i += BLOCK_SIZE) {{
    smem[index] += exp(X[row_start + i] - maximum);
}}
workgroupBarrier();
"#
        );

        for i in (0..=6).rev().map(|x| 2u32.pow(x)) {
            reduce_sum.push_str(&format!(
                r#"
block_sum(index, {i}u);"#,
            ));
        }

        reduce_sum.push_str(&format!(
            r#"
    if index == 0u {{
        sum = dot(smem[0], {accessor}(1.0)); 
    }}
    workgroupBarrier();
"#,
        ));

        kernel_builder.indent();
        kernel_builder.write_fragment(reduce_sum.into());
        kernel_builder.dedent();

        let softmax = format!(
            r#"
    for(var i: u32 = index; i < {reduce_len}; i += BLOCK_SIZE) {{
        var val = X[row_start + i];
        X[row_start + i] = exp(val - maximum) / sum;
    }}
}}
"#,
        );

        kernel_builder.write_fragment(softmax.into());
        kernel_builder.render()
    }
}

impl Operation for Softmax {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        Ok(self.input.storage_view().clone())
    }
}

impl MetaOperation for Softmax {
    fn kernel_name(&self) -> String {
        "softmax".to_string()
    }

    fn supports_inplace(&self) -> bool {
        true
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input]
    }

    fn kernel_key(&self, _: bool, dst: &Tensor) -> String {
        format!("softmax_{}", self.kernel_element(dst).as_str())
    }

    fn bindvars<A: WgslPrimitive<T, N>, T: WgslDType, const N: usize>(
        &self,
        inplace: bool,
        _: &Tensor,
    ) -> RVec<WgslFragment> {
        if !inplace {
            panic!("Only inplace softmax is supported");
        }

        let mut fragment = WgslFragment::new(32);
        fragment.write(&format!("X: array<{}>;\n", A::render_type()));
        rvec![fragment]
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        let input = &self.input;
        let N = input.shape()[self.dim] as u32;
        if N % 4 == 0 {
            KernelElement::Vec4
        } else if N % 2 == 0 {
            KernelElement::Vec2
        } else {
            KernelElement::Scalar
        }
    }

    fn calculate_dispatch(&self, _dst: &Tensor) -> Result<WorkgroupCount, OperationError> {
        let input = &self.input;
        let stacks = input.shape().slice(0..self.dim - 1).numel();
        let M = input.shape()[self.dim - 1] as u32;
        Ok(wgc![M as _, stacks as _, 1])
    }

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if !inplace {
            panic!("Only inplace softmax is supported");
        }
        Ok(BindGroupLayoutDescriptor::unary_inplace())
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        _: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        let input = &self.input;
        let M = input.shape()[self.dim - 1] as u32;
        let N = input.shape()[self.dim] as u32;
        let ND2 = N / 2;
        let ND4 = N / 4;
        let meta = SoftmaxMeta { M, N, ND2, ND4 };
        Ok(uniform.write(&meta)?)
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::test_util::run_py_prg;
    use crate::{shape, wgs, Device, DeviceRequest, Softmax, Tensor};
    use half::f16;
    use wgpu::naga::front::wgsl::parse_str;

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    fn ground_truth(a: &Tensor) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
import torch.nn.functional as F
def softmax(a):
    return F.softmax(torch.from_numpy(a), dim=-1).numpy()
"#;
        run_py_prg(prg.to_string(), &[a], &[])
    }

    fn run_softmax_trial(problem: SoftmaxProblem) {
        let device = GPU_DEVICE.with(|d| d.clone());
        let SoftmaxProblem { B, M, N } = problem;
        let a = Tensor::randn::<f32>(shape![B, M, N], Device::CPU);
        let ground = ground_truth(&a).unwrap();

        let a_gpu = a.to(&device).unwrap();
        let b = a_gpu.softmax(2).unwrap().resolve().unwrap();

        let ours = b.to(&Device::CPU).unwrap();
        println!("ours = {:?}", ours);
        println!("ground = {:?}", ground);
        ground.all_close(&ours, 1e-6, 1e-6).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct SoftmaxProblem {
        #[strategy(1..=3usize)]
        B: usize,
        #[strategy(1..=256usize)]
        M: usize,
        #[strategy(1..=256usize)]
        N: usize,
    }

    #[proptest(cases = 8)]
    fn test_softmax(prob: SoftmaxProblem) {
        let SoftmaxProblem { B, M, N } = prob;
        println!("B = {}, M = {}, N = {}", B, M, N);
        run_softmax_trial(prob);
    }

    #[test]
    fn test_render_softmax() {
        let device = GPU_DEVICE.with(|d| d.clone());
        let a = Tensor::randn::<f16>(shape![1, 2, 128], device.clone());
        let dst = Tensor::zeros::<f16>(&shape![1, 2, 128], &device);
        let op = Softmax::new(a, 2);
        let wgs = wgs![128, 1, 1];
        let kernel = op.render(true, &dst, wgs);
        println!("{}", kernel);
        parse_str(&kernel.to_string()).unwrap();
    }
}
