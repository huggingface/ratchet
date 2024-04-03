use derive_new::new;
use encase::ShaderType;

use crate::{
    gpu::{BindGroupLayoutDescriptor, CpuUniform, WorkgroupCount},
    rvec, wgc, KernelElement, MetaOperation, OpGuards, OpMetadata, Operation, OperationError, RVec,
    StorageView, Strides, Tensor,
};

#[derive(new, Debug, Clone)]
pub struct RoPE {
    input: Tensor,
    dim: usize,
    base: f32,
}

#[derive(Debug, derive_new::new, ShaderType)]
pub struct RoPEMeta {
    in_strides: glam::UVec3,
    out_strides: glam::UVec3,
    offset: u32,
    base: f32,
    scale: f32,
}

impl OpMetadata for RoPEMeta {}

impl OpGuards for RoPE {
    fn check_shapes(&self) {
        let input = &self.input;
        assert!(input.rank() == 4);
    }

    fn check_dtypes(&self) {
        let input = &self.input;
        assert!(input.dt() == crate::DType::F32);
    }
}

impl Operation for RoPE {
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        Ok(self.input.storage_view().clone())
    }
}

impl MetaOperation for RoPE {
    fn supports_inplace(&self) -> bool {
        false
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input]
    }

    fn kernel_key(&self, dst: &Tensor) -> String {
        format!("rope_{}", self.kernel_element(dst).as_str())
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, _dst: &Tensor) -> Result<WorkgroupCount, OperationError> {
        const WGSX: usize = 16;
        const WGSY: usize = 8;
        const WGSZ: usize = 8;

        let input = &self.input;
        let [_, _, SL, HD]: [usize; 4] = input.shape().try_into()?;

        let total_x = self.dim / 2;
        let total_y = SL;
        let total_z = input.shape().numel() / (SL * HD);

        let wgcx = (total_x / WGSX) as _;
        let wgcy = (total_y / WGSY) as _;
        let wgcz = (total_z / WGSZ) as _;

        Ok(wgc![wgcx, wgcy, wgcz])
    }

    fn storage_bind_group_layout(
        &self,
        _: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        Ok(BindGroupLayoutDescriptor::unary())
    }

    fn write_metadata(
        &self,
        uniform: &mut CpuUniform,
        dst: &Tensor,
        _: &KernelElement,
    ) -> Result<u64, OperationError> {
        let mut input_shape = self.input.shape().clone();
        let mut out_shape = dst.shape().clone();
        input_shape.remove(0);
        out_shape.remove(0);
        let in_strides = Strides::from(&input_shape);
        let out_strides = Strides::from(&out_shape);
        let meta = RoPEMeta::new(
            (&in_strides).into(),
            (&out_strides).into(),
            0,
            self.base,
            1.0,
        );
        Ok(uniform.write(&meta)?)
    }
}

#[cfg(test)]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::test_util::run_py_prg;
    use crate::{shape, Device, DeviceRequest, Tensor};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    fn ground_truth(a: &Tensor, dim: usize) -> anyhow::Result<Tensor> {
        let prg = r#"
import mlx.core as mx
import mlx.nn as nn
import numpy as np

def mlx_rope(input, dim):
    rope = nn.RoPE(dim)
    mx_input = mx.array(input)
    y = rope(mx_input)
    mx.eval(y)
    return np.array(y)
"#;
        run_py_prg(prg.to_string(), &[a], &[&dim])
    }

    fn run_softmax_trial(problem: RoPEProblem) {
        let rope_dim = 128;
        let device = GPU_DEVICE.with(|d| d.clone());
        let RoPEProblem { BS, NH, SL, HD } = problem;
        let a = Tensor::randn::<f32>(shape![BS, NH, SL, HD], Device::CPU);
        let ground = ground_truth(&a, rope_dim).unwrap();

        let a_gpu = a.to(&device).unwrap();
        let b = a_gpu.rope(10000.0, rope_dim).unwrap().resolve().unwrap();

        let ours = b.to(&Device::CPU).unwrap();
        println!("ours = {:?}", ours);
        println!("ground = {:?}", ground);
        ground.all_close(&ours, 1e-6, 1e-6).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct RoPEProblem {
        #[strategy(1..=3usize)]
        BS: usize,
        #[strategy(1..=64usize)]
        NH: usize,
        #[strategy(1..=512usize)]
        SL: usize,
        #[strategy(1..=128usize)]
        HD: usize,
    }

    #[proptest(cases = 8)]
    fn test_softmax(prob: RoPEProblem) {
        let RoPEProblem { BS, NH, SL, HD } = prob;
        println!("BS = {}, NH = {}, SL = {}, HD = {}", BS, NH, SL, HD);
        run_softmax_trial(prob);
    }
}
