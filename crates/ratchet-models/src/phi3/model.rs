use std::io::{BufRead, Seek};

use ratchet::{shape, Device, DeviceRequest, Tensor};
use ratchet_loader::gguf::gguf::Header;
use ratchet_nn::{Embedding, KVCache, KVEntry, LayerNorm, Linear, Module, RMSNorm};

use super::{
    attn::{PhiAttnInput, PhiSelfAttention},
    mlp::MLP,
};

#[cfg(target_arch = "wasm32")]
use {
    crate::ratchet_from_gguf_web, crate::TensorMap, js_sys::Uint8Array,
    ratchet_loader::gguf::gguf::ratchet_from_gguf, std::collections::HashMap,
    wasm_bindgen::prelude::*,
};

#[derive(Debug)]
pub struct DecoderLayer {
    input_norm: RMSNorm,
    self_attn: PhiSelfAttention,
    ffn_norm: RMSNorm,
    mlp: MLP,
}

impl DecoderLayer {
    pub fn load<R: BufRead + Seek>(
        header: &Header,
        reader: &mut R,
        layer_index: usize,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let self_attn = PhiSelfAttention::load(header, reader, layer_index, device)?;
        let mut lt = |name: &str| {
            let key = format!("blk.{}.{}", layer_index, name);
            header.tensor(reader, &key, device)
        };

        let norm_eps = header
            .metadata
            .get("phi3.attention.layer_norm_rms_epsilon")?
            .to_f32()?;

        let input_norm = RMSNorm::new(lt("attn_norm.weight")?, norm_eps);
        let ffn_norm = RMSNorm::new(lt("ffn_norm.weight")?, norm_eps);

        let mlp = MLP::new(
            Linear::new(lt("ffn_up.weight")?, None),
            Linear::new(lt("ffn_down.weight")?, None),
        );
        Ok(Self {
            input_norm,
            self_attn,
            ffn_norm,
            mlp,
        })
    }

    #[cfg(target_arch = "wasm32")]
    pub fn from_web(
        header: &Header,
        tensors: &mut TensorMap,
        layer_index: usize,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let self_attn = PhiSelfAttention::from_web(header, tensors, layer_index, device)?;
        let mut lt = |name: &str| {
            let key = format!("blk.{}.{}", layer_index, name);
            let tensor = tensors
                .remove(&key)
                .ok_or_else(|| anyhow::anyhow!("missing tensor"))?;
            ratchet_from_gguf_web(tensor, device)
        };

        let norm_eps = header
            .metadata
            .get("phi3.attention.layer_norm_rms_epsilon")?
            .to_f32()?;

        let input_norm = RMSNorm::new(lt("attn_norm.weight")?, norm_eps);
        let ffn_norm = RMSNorm::new(lt("ffn_norm.weight")?, norm_eps);

        let mlp = MLP::new(
            Linear::new(lt("ffn_up.weight")?, None),
            Linear::new(lt("ffn_down.weight")?, None),
        );
        Ok(Self {
            input_norm,
            self_attn,
            ffn_norm,
            mlp,
        })
    }
}

pub struct DecoderLayerInput {
    pub x: Tensor,
    pub mask: Option<Tensor>,
    pub cache: Option<KVEntry>,
}

impl Module for DecoderLayer {
    type Input = DecoderLayerInput;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let DecoderLayerInput { x, mask, cache } = input;
        let residual = x.clone();
        let xs = self.input_norm.schedule(x)?;
        let attn_output = self.self_attn.schedule(PhiAttnInput {
            input: xs.clone(),
            mask,
            cache,
        })?;
        let xs = residual.add(attn_output)?;
        let residual = xs.clone();
        let xs = self.ffn_norm.schedule(xs)?;
        let xs = self.mlp.schedule(xs)?;
        let xs = residual.add(xs)?;
        Ok(xs)
    }
}

#[derive(Debug)]
pub struct Phi3 {
    pub embedding: Embedding,
    pub layers: Vec<DecoderLayer>,
    pub ln_post: RMSNorm,
    pub lm_head: Linear,
    pub kv_cache: KVCache,
    pub device: Device,
}

impl Module for Phi3 {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let mut x = self.embedding.schedule(input)?;

        let [_, seq_len, n_state]: [usize; 3] = x.shape().try_into()?;
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(Self::generate_mask(seq_len, x.device())?)
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let input = DecoderLayerInput {
                x,
                mask: mask.clone(),
                cache: Some(self.kv_cache[layer_idx].clone()),
            };
            x = layer.schedule(input)?;
        }
        x = self.ln_post.schedule(x)?;
        x = x.slice(&[0..1, seq_len - 1..seq_len, 0..n_state])?;
        let logits = self.lm_head.schedule(x)?;
        Ok(logits)
    }
}

impl Phi3 {
    const MAX_CACHE: usize = 4096; //TODO: configurable

    pub fn load<R: BufRead + Seek>(
        header: Header,
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let embedding = Embedding::new(header.tensor(reader, "token_embd.weight", device)?);

        let n_layers = header.metadata.get("phi3.block_count")?.to_u32()? as i32;

        let layers = (0..n_layers)
            .fold(Vec::with_capacity(n_layers as _), |mut blocks, i| {
                blocks.push(DecoderLayer::load(&header, reader, i as _, device));
                blocks
            })
            .into_iter()
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut lt = |name: &str| {
            let key = format!("output{}", name);
            header.tensor(reader, &key, device)
        };

        let metadata = &header.metadata;

        let norm_eps = metadata
            .get("phi3.attention.layer_norm_rms_epsilon")?
            .to_f32()?;
        let ln_post = RMSNorm::new(lt("_norm.weight")?, norm_eps);
        let lm_head = Linear::new(lt(".weight")?, None);

        let n_layers = metadata.get("phi3.block_count")?.to_u32()?;
        let d_model = metadata.get("phi3.embedding_length")?.to_u32()?;
        let n_heads = metadata.get("phi3.attention.head_count")?.to_u32()?;
        let hdim = d_model as f32 / n_heads as f32;

        let cache_shape = shape![1, n_layers as _, Self::MAX_CACHE, hdim as _];
        Ok(Self {
            embedding,
            layers,
            ln_post,
            lm_head,
            kv_cache: KVCache::new(n_layers as _, cache_shape, device),
            device: device.clone(),
        })
    }

    //TODO: dedup
    #[cfg(target_arch = "wasm32")]
    pub async fn from_web(header: Header, mut tensors: TensorMap) -> anyhow::Result<Self> {
        let device = Device::request_device(DeviceRequest::GPU).await?;
        let embedding = Embedding::new(ratchet_from_gguf_web(
            tensors
                .remove("token_embd.weight")
                .ok_or_else(|| anyhow::anyhow!("missing tensor"))?,
            &device,
        )?);

        let n_layers = header.metadata.get("phi3.block_count")?.to_u32()? as i32;

        let layers = (0..n_layers)
            .fold(Vec::with_capacity(n_layers as _), |mut blocks, i| {
                blocks.push(DecoderLayer::from_web(
                    &header,
                    &mut tensors,
                    i as _,
                    &device,
                ));
                blocks
            })
            .into_iter()
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut lt = |name: &str| {
            let key = format!("output{}", name);
            let tensor = tensors
                .remove(&key)
                .ok_or_else(|| anyhow::anyhow!("missing tensor"))?;
            ratchet_from_gguf_web(tensor, &device)
        };

        let metadata = &header.metadata;

        let norm_eps = metadata
            .get("phi3.attention.layer_norm_rms_epsilon")?
            .to_f32()?;
        let ln_post = RMSNorm::new(lt("_norm.weight")?, norm_eps);
        let lm_head = Linear::new(lt(".weight")?, None);

        let n_layers = metadata.get("phi3.block_count")?.to_u32()?;
        let d_model = metadata.get("phi3.embedding_length")?.to_u32()?;
        let n_heads = metadata.get("phi3.attention.head_count")?.to_u32()?;
        let hdim = d_model as f32 / n_heads as f32;

        let cache_shape = shape![1, n_layers as _, Self::MAX_CACHE, hdim as _];
        Ok(Self {
            embedding,
            layers,
            ln_post,
            lm_head,
            kv_cache: KVCache::new(n_layers as _, cache_shape, &device),
            device: device.clone(),
        })
    }

    pub fn generate_mask(seq_len: usize, device: &Device) -> anyhow::Result<Tensor> {
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();

        Ok(Tensor::from_data(
            mask,
            shape![seq_len, seq_len],
            device.clone(),
        ))
    }

    pub fn reset(&mut self) {
        self.kv_cache.reset();
    }

    pub fn cache_mut(&mut self) -> &mut KVCache {
        &mut self.kv_cache
    }
}

#[cfg(all(test, not(target_arch = "wasm32"), feature = "pyo3"))]
mod tests {
    use hf_hub::api::sync::Api;
    use ndarray::Axis;
    use ndarray_stats::QuantileExt;
    use numpy::PyArrayDyn;
    use pyo3::{types::PyModule, Python};
    use ratchet::{prelude::shape, Device, DeviceRequest, Tensor};
    use ratchet_loader::gguf;
    use ratchet_nn::Module;
    use tokenizers::Tokenizer;

    use super::Phi3;

    #[test]
    #[cfg_attr(feature = "ci", ignore)]
    fn load_phi3() -> anyhow::Result<()> {
        let _ = env_logger::builder().is_test(true).try_init();
        let api = Api::new().unwrap();
        let model_repo = api.model("FL33TW00D-HF/phi3".to_string());
        let model_path = model_repo.get("phi3-mini-4k-f16.gguf").unwrap();
        println!("MODEL PATH: {}", model_path.display());

        let mut reader = std::io::BufReader::new(std::fs::File::open(model_path)?);
        let device = Device::request_device(DeviceRequest::GPU)?;
        let content = gguf::gguf::Header::read(&mut reader)?;
        let mut model = Phi3::load(content, &mut reader, &device)?;

        let tokenizer_repo = api.model("microsoft/Phi-3-mini-4k-instruct".to_string());
        let tokenizer_path = tokenizer_repo.get("tokenizer.json").unwrap();
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

        let prompt = "<|system|>
You are a helpful AI assistant.<|end|>
<|user|>
Plan me a 2 day trip to SF<|end|>
<|assistant|>";
        let encoding = tokenizer.encode(prompt, true).unwrap();
        let mut tokens = encoding
            .get_ids()
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        println!("PROMPT TOKENS: {:?}", tokens);
        let mut all_logits = vec![];
        let mut all_tokens = tokens.clone();
        let mut loop_cnt = 0;
        let start = std::time::Instant::now();
        while tokens[tokens.len() - 1] != 32000 && loop_cnt < 512 {
            let input = Tensor::from_data(tokens.clone(), shape![1, tokens.len()], device.clone());
            let result = model.schedule(input)?.resolve()?;
            let logits = result.to(&Device::CPU)?;
            all_logits.push(logits.clone());
            model.cache_mut().update(tokens.len());

            tokens = logits
                .to_ndarray_view::<f32>()
                .map_axis(Axis(2), |row| row.argmax_skipnan().unwrap())
                .iter()
                .map(|&x| x as i32)
                .collect::<Vec<_>>();
            let u32_toks = tokens.iter().map(|&x| x as u32).collect::<Vec<_>>();
            all_tokens.extend(tokens.clone());
            loop_cnt += 1;
        }
        let elapsed = start.elapsed();
        let u32_toks = all_tokens.iter().map(|&x| x as u32).collect::<Vec<_>>();
        println!("{}", tokenizer.decode(&u32_toks, true).unwrap());

        println!(
            "Tok/sec: {}",
            all_tokens.len() as f64 / elapsed.as_secs_f64()
        );

        Ok(())
    }
}
