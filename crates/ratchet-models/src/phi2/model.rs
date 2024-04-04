use std::io::{BufRead, Seek};

use ratchet::{shape, Device, Tensor};
use ratchet_loader::gguf::gguf::Content;
use ratchet_nn::{Embedding, KVCache, KVEntry, LayerNorm, Linear, Module};

use super::{
    attn::{PhiAttnInput, PhiSelfAttention},
    mlp::MLP,
};

#[derive(Debug)]
pub struct DecoderLayer {
    ln: LayerNorm,
    self_attn: PhiSelfAttention,
    mlp: MLP,
}

impl DecoderLayer {
    pub fn load<R: BufRead + Seek>(
        disk_model: &Content,
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
            Linear::new(lt("ffn_up.weight")?, Some(lt("ffn_up.bias")?), true),
            Linear::new(lt("ffn_down.weight")?, Some(lt("ffn_down.bias")?), true),
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

    fn forward(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let DecoderLayerInput { x, mask, cache } = input;
        let residual = x.clone();
        let xs = self.ln.forward(x)?;
        let attn_output = self.self_attn.forward(PhiAttnInput {
            input: xs.clone(),
            mask,
            cache,
        })?;
        let ff_hs = self.mlp.forward(xs)?;
        Ok(attn_output.add(ff_hs)?.add(residual)?)
    }
}

#[derive(Debug)]
pub struct Phi2 {
    pub embedding: Embedding,
    pub layers: Vec<DecoderLayer>,
    pub ln_post: LayerNorm,
    pub lm_head: Linear,
    pub kv_cache: KVCache,
}

impl Module for Phi2 {
    type Input = Tensor;

    fn forward(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let mut x = self.embedding.forward(input)?;
        let [_, seq_len, n_state]: [usize; 3] = x.shape().try_into()?;
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(Self::generate_mask(seq_len, x.device())?)
        };

        for index in 0..32 {
            println!(
                "K ARC COUNT: {}",
                self.kv_cache[index].k_cache.strong_count()
            );
            println!(
                "V ARC COUNT: {}",
                self.kv_cache[index].v_cache.strong_count()
            );
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let input = DecoderLayerInput {
                x,
                mask: mask.clone(),
                cache: Some(self.kv_cache[layer_idx].clone()),
            };
            x = layer.forward(input)?;
        }
        x = self.ln_post.forward(x)?;
        x = x.slice(&[0..1, seq_len - 1..seq_len, 0..n_state])?;
        let logits = self.lm_head.forward(x)?;
        Ok(logits)
    }
}

impl Phi2 {
    const MAX_CACHE: usize = 1024;

    pub fn load<R: BufRead + Seek>(
        disk_model: Content,
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let embedding = Embedding::new(
            disk_model.tensor(reader, "token_embd.weight", device)?,
            false,
        );

        let n_layers = disk_model
            .metadata
            .get("phi2.block_count")
            .unwrap()
            .to_u32()? as i32;

        let layers = (0..n_layers)
            .fold(Vec::with_capacity(n_layers as _), |mut blocks, i| {
                blocks.push(DecoderLayer::load(&disk_model, reader, i as _, device));
                blocks
            })
            .into_iter()
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut lt = |name: &str| {
            let key = format!("output{}", name);
            disk_model.tensor(reader, &key, device)
        };

        let ln_post = LayerNorm::new(lt("_norm.weight")?, Some(lt("_norm.bias")?), 1e-5);
        let lm_head = Linear::new(lt(".weight")?, Some(lt(".bias")?), true);

        Ok(Self {
            embedding,
            layers,
            ln_post,
            lm_head,
            kv_cache: KVCache::new(n_layers, shape![1, 32, Self::MAX_CACHE, 80], device),
        })
    }

    pub fn generate_mask(seq_len: usize, device: &Device) -> anyhow::Result<Tensor> {
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();

        Ok(Tensor::from_data(
            &mask,
            shape![seq_len, seq_len],
            device.clone(),
        ))
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
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
    extracted = collections.defaultdict(list)
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    inputs = tokenizer("def print_prime(n):", return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs, max_length=8, return_dict_in_generate=True, output_logits=True)
    generated_logits = outputs.logits[0]
    generated_logits = torch.unsqueeze(torch.unsqueeze(generated_logits, 0), 0)
    return generated_logits.numpy()
"#;
        Python::with_gil(|py| {
            let prg = PyModule::from_code(py, &prg, "x.py", "x")?;
            let py_result: Vec<&PyArrayDyn<f32>> = prg.getattr("ground")?.call0()?.extract()?;
            Ok(py_result.into_iter().map(Tensor::from).collect::<_>())
        })
    }

    #[test]
    fn load_phi2() -> anyhow::Result<()> {
        let _ = env_logger::builder().is_test(true).try_init();
        let model_path = concat!(
            env!("CARGO_RUSTC_CURRENT_DIR"),
            "/models/microsoft/phi-2/phi2-f16.gguf"
        );
        let mut reader = std::io::BufReader::new(std::fs::File::open(model_path)?);
        let device = Device::request_device(DeviceRequest::GPU)?;
        let content = gguf::gguf::Content::read(&mut reader)?;
        let model = Phi2::load(content, &mut reader, &device)?;

        let tokenizer = Tokenizer::from_file(concat!(
            env!("CARGO_RUSTC_CURRENT_DIR"),
            "/models/microsoft/phi-2/tokenizer.json"
        ))
        .unwrap();

        let tokens = tokenizer.encode("def print_prime(n):", true).unwrap();
        let i32_tokens = tokens
            .get_ids()
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        let num_tokens = i32_tokens.len();
        let input = Tensor::from_data(i32_tokens, shape![1, num_tokens], device.clone());
        let result = model.forward(input)?.resolve()?;
        let cpu_result = result.to(&Device::CPU)?;
        let ground_truth = ground_truth()?;

        println!("OURS: {:?}\n", cpu_result.to_ndarray_view::<f32>());
        println!("THEIRS: {:?}", ground_truth);

        cpu_result.all_close(&ground_truth[0], 1e-3, 1e-3)?;
        Ok(())
    }
}
