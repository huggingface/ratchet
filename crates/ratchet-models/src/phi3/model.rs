use std::io::{BufRead, Seek};

use half::f16;
use ratchet::{shape, DType, Device, Tensor};
use ratchet_loader::gguf::gguf::Header;
use ratchet_nn::{Embedding, KVCache, KVEntry, Linear, Module, RMSNorm};

use super::{
    attn::{PhiAttnInput, PhiSelfAttention},
    mlp::MLP,
};

#[cfg(target_arch = "wasm32")]
use {crate::ratchet_from_gguf_web, crate::TensorMap};

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
        let kv_cache = match device.compute_precision() {
            DType::F16 => KVCache::new::<f16>(n_layers as _, cache_shape, device),
            DType::F32 => KVCache::new::<f32>(n_layers as _, cache_shape, device),
            _ => unimplemented!(),
        };

        Ok(Self {
            embedding,
            layers,
            ln_post,
            lm_head,
            kv_cache,
            device: device.clone(),
        })
    }

    //TODO: dedup
    #[cfg(target_arch = "wasm32")]
    pub async fn from_web(header: Header, mut tensors: TensorMap) -> anyhow::Result<Self> {
        let device = Device::request_device(ratchet::DeviceRequest::GPU).await?;
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
            kv_cache: KVCache::new::<f32>(n_layers as _, cache_shape, &device),
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

    fn ground_truth(prompt: &str, max_tokens: usize) -> anyhow::Result<Vec<Tensor>> {
        let prg = format!(
            r#"
import torch
from transformers import Phi3ForCausalLM, AutoTokenizer

def ground():
    model = Phi3ForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
    inputs = tokenizer("""{}""", return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs, max_length={}, return_dict_in_generate=True, output_logits=True)
    generated_logits = outputs.logits
    print("Generated: ", tokenizer.decode(outputs[0][0], skip_special_tokens=True))

    result = [torch.unsqueeze(l, 0).numpy() for l in generated_logits]
    return result
"#,
            prompt, max_tokens
        );

        println!("GROUND TRUTH PROGRAM: {}", prg);
        Python::with_gil(|py| {
            let prg = PyModule::from_code(py, &prg, "x.py", "x")?;
            let py_result: Vec<&PyArrayDyn<f32>> = prg.getattr("ground")?.call0()?.extract()?;
            Ok(py_result.into_iter().map(Tensor::from).collect::<_>())
        })
    }

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

        let MAX_TOKENS = 100;
        let prompt = r#"<|user|>
How to explain Internet for a medieval knight?<|end|>
<|assistant|>"#;
        let encoding = tokenizer.encode(prompt, true).unwrap();
        let mut tokens = encoding
            .get_ids()
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        tokens.insert(0, 1); //TODO: what is going on here with tokenizers?
        let mut all_logits = vec![];
        let mut all_tokens = tokens.clone();
        let mut generated_cnt = tokens.len();
        let start = std::time::Instant::now();

        while tokens[tokens.len() - 1] != 32007 && generated_cnt < MAX_TOKENS {
            let input = Tensor::from_data(tokens.clone(), shape![1, tokens.len()], device.clone());
            let result = model.schedule(input)?.full()?.resolve()?;
            let logits = result.to(&Device::CPU)?;
            all_logits.push(logits.clone());
            model.cache_mut().update(tokens.len());

            tokens = logits
                .to_ndarray_view::<f32>()
                .map_axis(Axis(2), |row| row.argmax_skipnan().unwrap())
                .iter()
                .map(|&x| x as i32)
                .collect::<Vec<_>>();
            all_tokens.extend(tokens.clone());
            generated_cnt += 1;
        }
        let elapsed = start.elapsed();
        let u32_toks = all_tokens.iter().map(|&x| x as u32).collect::<Vec<_>>();

        let ground_logits = ground_truth(prompt, MAX_TOKENS)?;
        assert_eq!(all_logits.len(), ground_logits.len());
        let all_equal =
            ground_logits
                .iter()
                .zip(all_logits.iter())
                .enumerate()
                .all(|(i, (their, our))| {
                    print!("Checking: {}", i);
                    our.all_close(their, 5e-4, 5e-4).is_ok()
                });

        println!("All logits equal: {}", all_equal);
        assert!(all_equal);
        Ok(())
    }
}
