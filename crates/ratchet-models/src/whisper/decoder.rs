use super::config::Config;
use crate::whisper::residual_block::*;
use ratchet::prelude::*;
use ratchet_loader::gguf::gguf::Header;
use ratchet_nn::{Embedding, KVCache, LayerNorm, Module};
use std::io::{BufRead, Seek};

#[cfg(target_arch = "wasm32")]
use {crate::ratchet_from_gguf_web, crate::TensorMap};

#[derive(Debug)]
pub(crate) struct DecoderStem {
    pub token_embed: Embedding,
    pub pos_embed: Tensor,
}

impl DecoderStem {
    pub fn load<R: BufRead + Seek>(
        header: &Header,
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let lt = |name: &str| {
            let key = format!("model.decoder.{}", name);
            header.tensor(reader, &key, device)
        };
        Self::load_inner(lt)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn from_web(
        header: &Header,
        tensor_map: &mut TensorMap,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let lt = |name: &str| {
            let key = format!("model.decoder.{}", name);
            let wt = tensor_map.remove(&key).unwrap();
            ratchet_from_gguf_web(wt, device)
        };
        Self::load_inner(lt)
    }

    fn load_inner<F>(mut lt: F) -> anyhow::Result<Self>
    where
        F: FnMut(&str) -> anyhow::Result<Tensor>,
    {
        Ok(Self {
            token_embed: Embedding::new(lt("embed_tokens.weight")?),
            pos_embed: lt("embed_positions.weight")?,
        })
    }
}

#[derive(Debug)]
pub struct StemInput {
    pub tokens: Tensor,
    pub offset: usize,
}

impl Module for DecoderStem {
    type Input = StemInput;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let StemInput { tokens, offset } = input;
        let num_tokens = tokens.shape()[tokens.rank() - 1];
        let start = offset;
        let end = offset + num_tokens;
        let sliced = self
            .pos_embed
            .clone()
            .slice(&[start..end, 0..self.pos_embed.shape()[1]])?;
        self.token_embed.schedule(tokens)?.add(sliced)
    }
}

#[derive(Debug)]
pub struct WhisperDecoder {
    stem: DecoderStem,
    blocks: Vec<ResidualAttentionBlock>,
    mask: Tensor,
    ln_post: LayerNorm,
    cache: KVCache,
    device: Device,
}

impl Module for WhisperDecoder {
    type Input = [Tensor; 2];

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let [audio_ctx, tokens] = input;
        let mut x = self.stem.schedule(StemInput {
            tokens,
            offset: self.cache.entries(0),
        })?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            let block_input = ResidualAttentionBlockInputs {
                x,
                xa: Some(audio_ctx.clone()),
                mask: Some(self.mask.clone()),
                cache: Some(self.cache[block_idx].clone()),
            };
            x = block.schedule(block_input)?;
        }
        x = self.ln_post.schedule(x)?;
        let logits = self
            .stem
            .token_embed
            .weight
            .clone()
            .gemm(x, None, false, true, true)?;
        Ok(logits)
    }
}

impl WhisperDecoder {
    pub const MAX_CACHE: usize = 512;

    pub fn cache_mut(&mut self) -> &mut KVCache {
        &mut self.cache
    }

    pub fn reset(&mut self) {
        self.cache.reset();
    }

    fn load_mask(n_ctx: usize, device: &Device) -> Tensor {
        let mask: Vec<_> = (0..n_ctx)
            .flat_map(|i| (0..n_ctx).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        Tensor::from_data(mask, shape![n_ctx, n_ctx], device.clone())
    }

    #[cfg(target_arch = "wasm32")]
    pub fn from_web(
        header: &Header,
        config: &Config,
        tensor_map: &mut TensorMap,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let (n_layers, n_heads) = (config.n_text_layer, config.n_text_head);
        let stem = DecoderStem::from_web(header, tensor_map, device)?;

        let blocks = (0..n_layers)
            .fold(Vec::with_capacity(n_layers as _), |mut blocks, i| {
                blocks.push(ResidualAttentionBlock::from_web(
                    header,
                    tensor_map,
                    i as _,
                    n_heads as _,
                    "decoder",
                    device,
                ));
                blocks
            })
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;

        let mut lt = |name: &str| {
            let key = format!("model.decoder.layer_norm.{}", name);
            let wt = tensor_map.remove(&key).unwrap();
            ratchet_from_gguf_web(wt, device)
        };

        let n_state = config.n_audio_state as _;
        Ok(Self {
            stem,
            blocks,
            mask: Self::load_mask(config.n_text_ctx as _, device),
            ln_post: LayerNorm::new(lt("weight")?, Some(lt("bias")?), 1e-5),
            cache: KVCache::new(n_layers as _, shape![1, Self::MAX_CACHE, n_state], device),
            device: device.clone(),
        })
    }

    pub fn load<R: BufRead + Seek>(
        header: &Header,
        config: &Config,
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let stem = DecoderStem::load(header, reader, device)?;
        let (n_layers, n_heads) = (config.n_text_layer, config.n_text_head);

        let blocks = (0..n_layers)
            .fold(Vec::with_capacity(n_layers as _), |mut blocks, i| {
                blocks.push(ResidualAttentionBlock::load(
                    header,
                    reader,
                    i as _,
                    n_heads as _,
                    "decoder",
                    device,
                ));
                blocks
            })
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;

        let mut lt = |name: &str| {
            let key = format!("model.decoder.layer_norm.{}", name);
            header.tensor(reader, &key, device)
        };

        let n_state = config.n_audio_state as _;
        Ok(Self {
            stem,
            blocks,
            mask: Self::load_mask(config.n_text_ctx as _, device),
            ln_post: LayerNorm::new(lt("weight")?, Some(lt("bias")?), 1e-5),
            cache: KVCache::new(n_layers as _, shape![1, Self::MAX_CACHE, n_state], device),
            device: device.clone(),
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use hf_hub::api::sync::Api;
    use ndarray::{s, Axis};
    use ndarray_stats::QuantileExt;
    use numpy::PyArrayDyn;
    use pyo3::{
        prelude::*,
        types::{IntoPyDict, PyTuple},
    };
    use ratchet::{shape, Device, DeviceRequest, Tensor};
    use ratchet_loader::gguf::gguf;
    use ratchet_nn::Module;
    use tokenizers::Tokenizer;

    use crate::whisper::decoder::Config;
    use crate::whisper::{
        decoder::WhisperDecoder,
        options::{DecodingOptions, DecodingOptionsBuilder},
    };

    fn log_init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    fn ground_truth(audio_path: &str, options: DecodingOptions) -> anyhow::Result<Vec<Tensor>> {
        let prg = format!(
            r#"
import warnings
warnings.simplefilter("ignore")
import whisper
import numpy as np
def ground(options):
    model = whisper.load_model("tiny")
    result = model.transcribe(audio="{}", **options)
    output_logits = [l.numpy()[np.newaxis] for logits in result["all_logits"] for l in logits]
    return output_logits
"#,
            audio_path
        );
        Python::with_gil(|py| {
            let prg = PyModule::from_code(py, &prg, "x.py", "x")?;
            let py_args = PyTuple::new(py, [options.into_py_dict(py)]);
            let py_result: Vec<&PyArrayDyn<f32>> =
                prg.getattr("ground")?.call1(py_args)?.extract()?;
            Ok(py_result.into_iter().map(Tensor::from).collect::<_>())
        })
    }

    #[test]
    fn decoder_matches() -> anyhow::Result<()> {
        log_init();
        let api = Api::new().unwrap();
        let model = api.model("FL33TW00D-HF/whisper-tiny".to_string());
        let path = model.get("tiny_f32.gguf").unwrap();
        let config_path = model.get("config.json").unwrap();
        let config: Config = serde_json::from_slice(&std::fs::read(config_path).unwrap()).unwrap();
        println!("MODEL LOADED FROM: {}", path.display());

        let dataset = api.dataset("FL33TW00D-HF/ratchet-util".to_string());
        let options = DecodingOptionsBuilder::new().build();
        let hs_npy = dataset.get("jfk_tiny_encoder_hs.npy").unwrap();
        let audio_path = dataset.get("jfk.wav").unwrap();

        let tokenizer_repo = api.model("openai/whisper-tiny".to_string());
        let tokenizer_path = tokenizer_repo.get("tokenizer.json").unwrap();
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

        let mut reader = std::io::BufReader::new(std::fs::File::open(path).unwrap());
        let header = gguf::Header::read(&mut reader).unwrap();

        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let audio_ctx = Tensor::from_npy_path::<f32, _>(hs_npy, &device)?;
        let mut decoder = WhisperDecoder::load(&header, &config, &mut reader, &device)?;

        let mut tokens = vec![50258, 50259, 50359];
        let mut all_tokens = tokens.clone();
        let mut all_logits = vec![];
        let start = std::time::Instant::now();
        while tokens[tokens.len() - 1] != 50257 {
            let token_t =
                Tensor::from_data(tokens.clone(), shape![1, tokens.len()], device.clone());
            let result = decoder.schedule([audio_ctx.clone(), token_t])?.resolve()?;

            let our_logits = result.to(&Device::CPU)?;
            let nd_logits = our_logits.to_ndarray_view::<f32>();
            all_logits.push(Tensor::from(
                nd_logits
                    .slice(s![.., .., ..tokenizer.get_vocab_size(true)])
                    .to_owned()
                    .into_dyn(),
            ));

            let sliced = nd_logits
                .slice(s![.., -1.., ..tokenizer.get_vocab_size(true)])
                .remove_axis(Axis(1));
            decoder.cache_mut().update(tokens.len());

            tokens = sliced
                .map_axis(Axis(1), |row| row.argmax_skipnan().unwrap())
                .iter()
                .map(|&x| x as i32)
                .collect::<Vec<_>>();
            println!("Token: {:?}", tokens);
            all_tokens.extend(tokens.clone());
        }
        println!("Took: {:?}", start.elapsed());

        let u32_tokens: Vec<_> = all_tokens.iter().map(|&x| x as u32).collect();
        let decoded = tokenizer.decode(&u32_tokens, true).unwrap();
        println!("Decoded: {}", decoded);

        let ground_logits = ground_truth(&audio_path.to_string_lossy(), options)?;
        let all_equal = all_logits
            .iter()
            .zip(ground_logits.iter())
            .all(|(our, their)| their.all_close(our, 1e-4, 1e-4).is_ok());

        assert!(all_equal);

        Ok(())
    }
}
