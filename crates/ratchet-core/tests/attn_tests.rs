#[cfg(test)]
mod tests {
    use ratchet::{shape, test_util::run_py_prg, Device, DeviceRequest, Tensor};

    #[derive(Debug, derive_new::new)]
    struct AttentionTest {
        input: Tensor,
        qw: Tensor,
        kw: Tensor,
        vw: Tensor,
        n_heads: Option<usize>,
    }

    impl AttentionTest {
        fn to_gpu(&self, device: Device) -> Self {
            Self {
                input: self.input.to(&device).unwrap(),
                qw: self.qw.to(&device).unwrap(),
                kw: self.kw.to(&device).unwrap(),
                vw: self.vw.to(&device).unwrap(),
                n_heads: self.n_heads,
            }
        }
    }

    fn sdpa_ground(case: &AttentionTest) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
import math
def scaled_dot_product_attention(input, qw, kw, vw) -> torch.Tensor:
    input = torch.from_numpy(input)
    query = input @ torch.from_numpy(qw) 
    key = input @ torch.from_numpy(kw)
    value = input @ torch.from_numpy(vw)
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) 
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return (attn_weight @ value).numpy()
"#;
        run_py_prg(
            prg.to_string(),
            &[&case.input, &case.qw, &case.kw, &case.vw],
            &[],
        )
    }

    fn sdpa_cfg(case: &AttentionTest, device: Device) -> anyhow::Result<Tensor> {
        let (input, qw, kw, vw) = (
            case.input.clone(),
            case.qw.clone(),
            case.kw.clone(),
            case.vw.clone(),
        );
        let q_proj = input.clone().matmul(qw, false, false)?;
        let k_proj = input.clone().matmul(kw, false, false)?;
        let v_proj = input.matmul(vw, false, false)?;

        let scale_factor = 1f64 / (q_proj.shape()[2] as f64).sqrt();
        let scale_factor = Tensor::from_data([scale_factor as f32], shape![1], device);
        let kt = k_proj.permute(&[0, 2, 1])?;

        let logits = q_proj.matmul(kt, false, false)?.mul(scale_factor)?;
        logits.softmax(2)?.matmul(v_proj, false, false)
    }

    #[test]
    pub fn test_sdpa() -> anyhow::Result<()> {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = Tensor::randn::<f32>(shape![1, 128, 256], Device::CPU);
        let qw = Tensor::randn::<f32>(shape![256, 256], Device::CPU);
        let kw = Tensor::randn::<f32>(shape![256, 256], Device::CPU);
        let vw = Tensor::randn::<f32>(shape![256, 256], Device::CPU);
        let cpu_test_case = AttentionTest::new(input, qw, kw, vw, None);
        let ground = sdpa_ground(&cpu_test_case)?;

        let device = Device::request_device(DeviceRequest::GPU)?;
        let gpu_test_case = cpu_test_case.to_gpu(device.clone());
        let out = sdpa_cfg(&gpu_test_case, device.clone())?.resolve()?;
        let out_cpu = out.to(&Device::CPU)?;
        println!("OURS: {:?}\n", out_cpu);
        println!("GROUND: {:?}", ground);
        println!("Output shape: {:?}", out_cpu.shape());
        ground.all_close(&out_cpu, 8e-3, 8e-3)?;

        Ok(())
    }

    fn mha_ground(case: &AttentionTest) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
import torch.nn.functional as F
import numpy as np

def qkv_attention(input, qw, kw, vw, n_heads):
    input = torch.from_numpy(input)
    q = input @ torch.from_numpy(qw)
    k = input @ torch.from_numpy(kw)
    v = input @ torch.from_numpy(vw)

    n_batch, n_ctx, n_state = q.shape
    scale = (n_state // n_heads) ** -0.25
    q = q.view(*q.shape[:2], n_heads, -1).permute(0, 2, 1, 3) * scale
    k = k.view(*k.shape[:2], n_heads, -1).permute(0, 2, 3, 1) * scale
    v = v.view(*v.shape[:2], n_heads, -1).permute(0, 2, 1, 3)
    qk = q @ k
    qk = qk.float()

    w = F.softmax(qk, dim=-1).to(q.dtype)
    out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
    return np.ascontiguousarray(out.numpy())
"#;
        run_py_prg(
            prg.to_string(),
            &[&case.input, &case.qw, &case.kw, &case.vw],
            &[&case.n_heads.unwrap()],
        )
    }

    fn mha_cfg(case: &AttentionTest, device: Device) -> anyhow::Result<Tensor> {
        let (input, qw, kw, vw) = (
            case.input.clone(),
            case.qw.clone(),
            case.kw.clone(),
            case.vw.clone(),
        );
        let q_proj = input.clone().matmul(qw, false, false)?;
        let k_proj = input.clone().matmul(kw, false, false)?;
        let v_proj = input.matmul(vw, false, false)?;

        let n_heads = case.n_heads.unwrap();
        let qdim = q_proj.shape()[2];
        let scale = ((qdim / n_heads) as f32).powf(-0.25);
        let scale = Tensor::from_data([scale], shape![1], device);

        let hdim = qdim / n_heads;
        let q = q_proj
            .view(shape![1, hdim, n_heads, hdim])?
            .permute(&[0, 2, 1, 3])?
            .mul(scale.clone())?;
        let k = k_proj
            .view(shape![1, hdim, n_heads, hdim])?
            .permute(&[0, 2, 3, 1])?
            .mul(scale.clone())?;
        let v = v_proj
            .view(shape![1, hdim, n_heads, hdim])?
            .permute(&[0, 2, 1, 3])?;

        let qk = q.matmul(k, false, false)?;
        let attn = qk.softmax(3)?;
        attn.matmul(v, false, false)?
            .permute(&[0, 2, 1, 3])?
            .view(shape![1, hdim, qdim])
    }

    #[test]
    pub fn test_mha() -> anyhow::Result<()> {
        let _ = env_logger::builder().is_test(true).try_init();
        let input = Tensor::randn::<f32>(shape![1, 64, 384], Device::CPU);
        let qw = Tensor::randn::<f32>(shape![1, 384, 384], Device::CPU);
        let kw = Tensor::randn::<f32>(shape![1, 384, 384], Device::CPU);
        let vw = Tensor::randn::<f32>(shape![1, 384, 384], Device::CPU);
        let cpu_test_case = AttentionTest::new(input, qw, kw, vw, Some(6));
        let ground = mha_ground(&cpu_test_case)?;

        let device = Device::request_device(DeviceRequest::GPU)?;
        let gpu_test_case = cpu_test_case.to_gpu(device.clone());
        let out = mha_cfg(&gpu_test_case, device.clone())?.resolve()?;
        let out_cpu = out.to(&Device::CPU)?;
        println!("OURS: {:?}\n", out_cpu);
        println!("GROUND: {:?}", ground);
        println!("Output shape: {:?}", out_cpu.shape());
        ground.all_close(&out_cpu, 1e-2, 1e-2)?;

        Ok(())
    }
}
