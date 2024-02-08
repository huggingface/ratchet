use ratchet::{shape, Tensor};
use ratchet_nn::{Linear, Module};

#[derive(Debug, derive_new::new)]
pub struct MultiHeadAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads: usize,
}

#[derive(Debug, derive_new::new)]
pub struct MHAInputs {
    x: Tensor,
    xa: Option<Tensor>,
    mask: Option<Tensor>,
    is_causal: bool,
}

impl Module for MultiHeadAttention {
    type Input = MHAInputs;

    fn forward(&self, input: &Self::Input) -> anyhow::Result<Tensor> {
        let MHAInputs {
            x,
            xa,
            mask,
            is_causal,
        } = input;

        let q = self.q.forward(x)?;

        let to_project = xa.as_ref().unwrap_or(x);
        let k = self.k.forward(to_project)?;
        let v = self.v.forward(to_project)?;

        //TODO: cache
        self.qkv_attention(q, k, v, mask, xa.is_some(), *is_causal)
    }
}

impl MultiHeadAttention {
    fn qkv_attention(
        &self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: &Option<Tensor>,
        x_attn: bool,
        is_causal: bool,
    ) -> anyhow::Result<Tensor> {
        let [bs, n_ctx, n_state]: [usize; 3] = q.shape().try_into()?;
        let [k0, k1, _]: [usize; 3] = k.shape().try_into()?;
        let [v0, v1, _]: [usize; 3] = v.shape().try_into()?;

        let hdim = n_state / self.n_heads;
        let dk = Tensor::from_data([(hdim as f32).powf(-0.25)], shape![1], q.device().clone());

        let qs = shape![bs, n_ctx, self.n_heads, hdim];
        let ks = shape![k0, k1, self.n_heads, hdim];
        let vs = shape![v0, v1, self.n_heads, hdim];

        let q = q.view(qs)?.permute(&[0, 2, 1, 3])?.mul(&dk)?;
        let k = k.view(ks)?.permute(&[0, 2, 3, 1])?.mul(&dk)?;
        let v = v.view(vs)?.permute(&[0, 2, 1, 3])?;

        if x_attn {
            //TODO: static caching
        }

        let mut qk = q.matmul(&k)?;

        if let Some(ref m) = mask {
            let prepared_mask = if is_causal {
                m.slice(&[0..n_ctx, 0..n_ctx])?
            } else {
                m.clone()
            };
            qk = qk.add(&prepared_mask)?;
        }

        let w = qk.softmax(3)?;
        let wv = w
            .matmul(&v)?
            .permute(&[0, 2, 1, 3])?
            .view(shape![bs, n_ctx, n_state])?;

        self.o.forward(&wv)
    }
}

#[cfg(test)]
mod tests {
    use crate::{ResidualAttentionBlock, ResidualAttentionBlockInputs, Whisper};
    use hf_hub::api::sync::Api;
    use ratchet::{shape, Device, DeviceRequest, Tensor};
    use ratchet_loader::GGMLCompatible;
    use ratchet_nn::Module;

    #[test]
    fn whisper_mha() -> anyhow::Result<()> {
        let api = Api::new().unwrap();
        let model = api.model("ggerganov/whisper.cpp".to_string());
        let path = model.get("ggml-tiny.bin").unwrap();

        let mut reader = std::io::BufReader::new(std::fs::File::open(path).unwrap());
        let gg_disk = Whisper::load_ggml(&mut reader).unwrap();

        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let block0 = ResidualAttentionBlock::load(&gg_disk, 0, 6, false, &mut reader, &device)?;

        let x = Tensor::randn::<f32>(shape![1, 1500, 384], device.clone());
        let inputs = ResidualAttentionBlockInputs::new(x, None, None);
        let result = block0.forward(&inputs)?;
        result.resolve()?;

        let ours = result.to(&Device::CPU)?;
        println!("{:?}", ours);

        Ok(())
    }
}
