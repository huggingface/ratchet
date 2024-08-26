use super::{
    attn::{PhiAttnInput, PhiSelfAttention},
    mlp::MLP,
};
use half::f16;
use ratchet::{shape, DType, Device, Tensor};
use ratchet_loader::gguf::gguf::Header;
use ratchet_nn::{Embedding, KVCache, KVEntry, LayerNorm, Linear, Module};
use std::io::{BufRead, Seek};

#[cfg(target_arch = "wasm32")]
use {crate::ratchet_from_gguf_web, crate::TensorMap};

#[derive(Debug)]
pub struct DecoderLayer {
    ln: LayerNorm,
    self_attn: PhiSelfAttention,
    mlp: MLP,
}

impl DecoderLayer {
    pub fn load<R: BufRead + Seek>(
        disk_model: &Header,
        reader: &mut R,
        layer_index: usize,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let self_attn = PhiSelfAttention::load(disk_model, reader, layer_index, device)?;
        let mut lt = |name: &str| {
            let key = format!("blk.{}.{}", layer_index, name);
            disk_model.tensor(reader, &key, device)
        };

        let ln = LayerNorm::new(lt("attn_norm.weight")?, Some(lt("attn_norm.bias")?), 1e-5);

        let mlp = MLP::new(
            Linear::new(lt("ffn_up.weight")?, Some(lt("ffn_up.bias")?)),
            Linear::new(lt("ffn_down.weight")?, Some(lt("ffn_down.bias")?)),
        );
        Ok(Self { ln, self_attn, mlp })
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

        let ln = LayerNorm::new(lt("attn_norm.weight")?, Some(lt("attn_norm.bias")?), 1e-5);

        let mlp = MLP::new(
            Linear::new(lt("ffn_up.weight")?, Some(lt("ffn_up.bias")?)),
            Linear::new(lt("ffn_down.weight")?, Some(lt("ffn_down.bias")?)),
        );
        Ok(Self { ln, self_attn, mlp })
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
        let xs = self.ln.schedule(x)?;
        let attn_output = self.self_attn.schedule(PhiAttnInput {
            input: xs.clone(),
            mask,
            cache,
        })?;
        let ff_hs = self.mlp.schedule(xs)?;
        attn_output.add(ff_hs)?.add(residual)
    }
}

#[derive(Debug)]
pub struct Phi2 {
    pub embedding: Embedding,
    pub layers: Vec<DecoderLayer>,
    pub ln_post: LayerNorm,
    pub lm_head: Linear,
    pub kv_cache: KVCache,
    pub device: Device,
}

impl Module for Phi2 {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let mut x = self.embedding.schedule(input)?;
        println!("Phi2 x storage: {}", x.resolved());

        let [_, seq_len, n_state]: [usize; 3] = x.shape().try_into()?;
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(Self::generate_mask(seq_len, x.device())?)
        };
        match mask.as_ref() {
            Some(m) => {
                println!("Phi2 mask storage: {}", m.resolved());
            }
            None => {
                println!("Phi2 mask: None");
            }
        }

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

        println!("Phi2 logits storage: {}", logits.resolved());

        Ok(logits)
    }
}

impl Phi2 {
    const MAX_CACHE: usize = 1024; //TODO: configurable

    pub fn load<R: BufRead + Seek>(
        header: Header,
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let token_embedding = header.tensor(reader, "token_embd.weight", device)?;
        let embedding = Embedding::new(token_embedding);

        let n_layers = header.metadata.get("phi2.block_count").unwrap().to_u32()? as i32;

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

        let ln_post = LayerNorm::new(lt("_norm.weight")?, Some(lt("_norm.bias")?), 1e-5);
        let lm_head = Linear::new(lt(".weight")?, Some(lt(".bias")?));

        let cache_shape = shape![1, 32, Self::MAX_CACHE, 80];
        let kv_cache = match device.compute_precision() {
            DType::F16 => KVCache::new::<f16>(n_layers, cache_shape, device),
            DType::F32 => KVCache::new::<f32>(n_layers, cache_shape, device),
            _ => unimplemented!(),
        };

        println!(
            "Phi2::load embedding storage: {}",
            embedding.weight.resolved()
        );
        /*
        println!("Phi2 ln_post storage: {}", ln_post.weight().resolved());
        println!("Phi2 lm_head storage: {}", lm_head.w.resolved());
        for (idx, entry) in kv_cache.0.iter().enumerate() {
            println!(
                "Phi2 kv_cache #{idx}| k: {}, v: {}",
                entry.k_cache.resolved(),
                entry.v_cache.resolved()
            );
        }
         */

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

        let n_layers = header.metadata.get("phi2.block_count").unwrap().to_u32()? as i32;

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

        let ln_post = LayerNorm::new(lt("_norm.weight")?, Some(lt("_norm.bias")?), 1e-5);
        let lm_head = Linear::new(lt(".weight")?, Some(lt(".bias")?));

        let cache_shape = shape![1, 32, Self::MAX_CACHE, 80];
        let kv_cache = match device.compute_precision() {
            DType::F16 => KVCache::new::<f16>(n_layers, cache_shape, &device),
            DType::F32 => KVCache::new::<f32>(n_layers, cache_shape, &device),
            _ => unimplemented!(),
        };

        Ok(Self {
            embedding,
            layers,
            ln_post,
            lm_head,
            kv_cache,
            device,
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

    use super::Phi2;

    fn ground_truth() -> anyhow::Result<Vec<Tensor>> {
        let prg = r#"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import collections

def ground():
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    inputs = tokenizer("def print_prime(n):", return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs, max_length=20, return_dict_in_generate=True, output_logits=True)
    generated_logits = outputs.logits

    result = [torch.unsqueeze(l, 0).numpy() for l in generated_logits]
    return result
"#;
        Python::with_gil(|py| {
            let prg = PyModule::from_code(py, prg, "x.py", "x")?;
            let py_result: Vec<&PyArrayDyn<f32>> = prg.getattr("ground")?.call0()?.extract()?;
            Ok(py_result.into_iter().map(Tensor::from).collect::<_>())
        })
    }

    #[test]
    #[cfg_attr(feature = "ci", ignore)]
    fn load_phi2() -> anyhow::Result<()> {
        let _ = env_logger::builder().is_test(true).try_init();
        let api = Api::new().unwrap();
        let model_repo = api.model("FL33TW00D-HF/phi2".to_string());
        let model_path = model_repo.get("phi2-f16.gguf").unwrap();
        println!("MODEL PATH: {}", model_path.display());

        let mut reader = std::io::BufReader::new(std::fs::File::open(model_path)?);
        let device = Device::request_device(DeviceRequest::GPU)?;
        let content = gguf::gguf::Header::read(&mut reader)?;
        let mut model = Phi2::load(content, &mut reader, &device)?;

        let tokenizer_repo = api.model("microsoft/phi-2".to_string());
        let tokenizer_path = tokenizer_repo.get("tokenizer.json").unwrap();
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

        let prompt = "def print_prime(n):";
        print!("{}", prompt);
        let encoding = tokenizer.encode(prompt, true).unwrap();
        let mut tokens = encoding
            .get_ids()
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        let mut all_logits = vec![];
        let mut all_tokens = tokens.clone();
        let mut loop_cnt = 0;
        while tokens[tokens.len() - 1] != 50256 && loop_cnt < 13 {
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
            print!("{}", tokenizer.decode(&u32_toks, true).unwrap());
            all_tokens.extend(tokens.clone());
            loop_cnt += 1;
        }

        let ground_logits = ground_truth()?;
        let all_equal = all_logits
            .iter()
            .zip(ground_logits.iter())
            .all(|(our, their)| their.all_close(our, 1e-3, 1e-3).is_ok());
        println!("All logits equal: {}", all_equal);
        assert!(all_equal);
        Ok(())
    }
}
