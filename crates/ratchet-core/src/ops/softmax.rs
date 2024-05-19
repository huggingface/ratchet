use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{dtype::WgslDType, BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, wgs, Accessor, BuiltIn, KernelElement, MetaOperation, OpGuards, OpMetadata,
    Operation, OperationError, RVec, StorageView, Tensor, WgslKernel, WgslKernelBuilder,
};

#[derive(new, Debug, Clone)]
pub struct Softmax {
    input: Tensor,
    dim: usize,
}

#[derive(Debug, derive_new::new, ShaderType)]
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
        assert!(input.dt() == crate::DType::F32);
    }
}

impl Softmax {
    fn write_bindings(&self, inplace: bool, dst: &Tensor) -> WgslKernel {
        let mut builder = WgslKernelBuilder::new();
        let bindings = self.storage_bind_group_layout(inplace).unwrap();
        builder.bind_tensors(&[&self.input, dst], &bindings);
        let result = builder.render();
        println!("{}", result);
        result
    }

    pub fn render_softmax<A: Accessor<T, N>, T: WgslDType, const N: usize>() -> WgslKernel {
        let mut kernel_builder = WgslKernelBuilder::new();
        let accessor = A::render();
        let reduce_len = match N {
            1 => "metadata.N",
            2 => "metadata.ND2",
            4 => "metadata.ND4",
            _ => panic!("Invalid dimension"),
        };

        kernel_builder.write_main(
            wgs![128, 1, 1],
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
    smem[index] = max(smem[index], X0[row_start + i]); 
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
smem[index] = {accessor}(0.0);
for (var i: u32 = index; i < {reduce_len}; i += BLOCK_SIZE) {{
    smem[index] += exp(X[row_start + i] - maximum);
}}
workgroupBarrier();
"#
        );

        //Need to iterate from 64, 32, 16,
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
    use crate::{shape, Device, DeviceRequest, Softmax, Tensor};

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
    fn test_render() {
        let a = Tensor::randn::<f32>(shape![1, 2, 3], Device::CPU);
        let op = Softmax::new(a, 2);
        let _ = op.write_bindings(true, &op.input);
    }
}
