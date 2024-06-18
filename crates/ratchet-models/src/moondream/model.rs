use std::io::{BufRead, Seek};

use anyhow::Ok;
use ratchet::{shape, Device, Tensor};
use ratchet_loader::gguf::gguf::Header;
use ratchet_nn::{Embedding, KVCache, LayerNorm, Linear, RotaryEmbedding};

#[cfg(target_arch = "wasm32")]
use {crate::ratchet_from_gguf_web, crate::TensorMap};

use super::{
    mlp::MLP,
    text_model::{DecoderLayer, SelfAttention, TextModel},
    vision_encoder::{
        Attention, LinearPatchEmbedding, VisionEncoder, VisionProjection, VisionTransformer,
        VitBlock,
    },
};

#[derive(Debug)]
pub struct Moondream {
    pub vision_encoder: VisionEncoder,
    pub text_model: TextModel,
}

impl Moondream {
    pub fn load<R: BufRead + Seek>(
        header: Header,
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let lt = |name: &str| header.tensor(reader, name, device).unwrap();
        Self::load_inner(&header, lt, device)
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn from_web(header: Header, mut tensors: TensorMap) -> anyhow::Result<Self> {
        let device = Device::request_device(ratchet::DeviceRequest::GPU).await?;
        let mut lt = |name: &str| {
            let tensor = tensors
                .remove(name)
                .ok_or_else(|| anyhow::anyhow!("missing tensor"));
            ratchet_from_gguf_web(tensor.unwrap(), &device).unwrap()
        };
        Self::load_inner(&header, lt, &device)
    }

    fn load_inner<F>(header: &Header, mut lt: F, device: &Device) -> anyhow::Result<Self>
    where
        F: FnMut(&str) -> Tensor,
    {
        let n_layers = 24_i32;
        let dim = 2048_f32;
        let n_heads = 32_u32;
        let n_kv_heads = 32_u32;
        let rope_base = 10000.0f32;
        let rope_dim = 32_u32;
        let ln_eps = 1e-05;
        let hdim = dim / n_heads as f32;
        let softmax_scale = Tensor::from_data([1.0 / hdim.sqrt()], shape![1], device.clone());
        let cache_shape = shape![1, 32, 4096, 64];

        let text_model = TextModel::new(
            Embedding::new(lt("text_model.transformer.embd.wte.weight")),
            (0..n_layers)
                .map(|i| {
                    DecoderLayer::new(
                        LayerNorm::new(
                            lt(&format!("text_model.transformer.h.{}.ln.weight", i)),
                            Some(lt(&format!("text_model.transformer.h.{}.ln.bias", i))),
                            ln_eps,
                        ),
                        SelfAttention::new(
                            Linear::new(
                                lt(&format!("text_model.transformer.h.{}.mixer.Wqkv.weight", i)),
                                Some(lt(&format!(
                                    "text_model.transformer.h.{}.mixer.Wqkv.bias",
                                    i
                                ))),
                            ),
                            Linear::new(
                                lt(&format!(
                                    "text_model.transformer.h.{}.mixer.out_proj.weight",
                                    i
                                )),
                                Some(lt(&format!(
                                    "text_model.transformer.h.{}.mixer.out_proj.bias",
                                    i
                                ))),
                            ),
                            RotaryEmbedding::new(rope_dim as usize, false, rope_base, 1.0),
                            n_heads,
                            softmax_scale.clone(),
                            n_kv_heads,
                        ),
                        MLP::new(
                            Linear::new(
                                lt(&format!("text_model.transformer.h.{}.mlp.fc1.weight", i)),
                                Some(lt(&format!("text_model.transformer.h.{}.mlp.fc1.bias", i))),
                            ),
                            Linear::new(
                                lt(&format!("text_model.transformer.h.{}.mlp.fc2.weight", i)),
                                Some(lt(&format!("text_model.transformer.h.{}.mlp.fc2.bias", i))),
                            ),
                        ),
                    )
                })
                .collect(),
            LayerNorm::new(
                lt("text_model.lm_head.ln.weight"),
                Some(lt("text_model.lm_head.ln.bias")),
                ln_eps,
            ),
            Linear::new(
                lt("text_model.lm_head.linear.weight"),
                Some(lt("text_model.lm_head.linear.bias")),
            ),
            KVCache::new::<f32>(n_layers as _, cache_shape, device),
            device.clone(),
        );

        let projection = VisionProjection::new(MLP::new(
            Linear::new(
                lt("vision_encoder.projection.mlp.fc1.weight"),
                Some(lt("vision_encoder.projection.mlp.fc1.bias")),
            ),
            Linear::new(
                lt("vision_encoder.projection.mlp.fc2.weight"),
                Some(lt("vision_encoder.projection.mlp.fc2.bias")),
            ),
        ));

        let transformer = VisionTransformer::new(
            LinearPatchEmbedding::new(
                Linear::new(lt("vision_encoder.encoder.model.visual.patch_embed.linear.weight"), Some(lt("vision_encoder.encoder.model.visual.patch_embed.linear.bias"))),
            ),
            lt("vision_encoder.encoder.model.visual.pos_embed"),
            (0..27)
                .map(|layer| {
                    let qkvw = lt(&format!("vision_encoder.encoder.model.visual.blocks.{}.attn.qkv.weight", layer));
                    let qkvb = lt(&format!("vision_encoder.encoder.model.visual.blocks.{}.attn.qkv.bias", layer));

                    let n_heads = 16;
                    let dim = 1152;
                    let h_dim = dim / n_heads;
                    let scale_factor =
                        Tensor::from_data([1.0 / (h_dim as f32).sqrt()], shape![1], device.clone());

                    VitBlock::new(
                        1152,
                        Attention::new(
                            n_heads,
                            dim,
                            Linear::new(qkvw, Some(qkvb)),
                            Linear::new(
                                lt(&format!("vision_encoder.encoder.model.visual.blocks.{}.attn.proj.weight", layer)),
                                Some(lt(&format!("vision_encoder.encoder.model.visual.blocks.{}.attn.proj.bias", layer))),
                            ),
                            scale_factor,
                        ),
                        MLP::new(
                            Linear::new(
                                lt(&format!("vision_encoder.encoder.model.visual.blocks.{}.mlp.fc1.weight", layer)),
                                Some(lt(&format!("vision_encoder.encoder.model.visual.blocks.{}.mlp.fc1.bias", layer))),
                            ),
                            Linear::new(
                                lt(&format!("vision_encoder.encoder.model.visual.blocks.{}.mlp.fc2.weight", layer)),
                                Some(lt(&format!("vision_encoder.encoder.model.visual.blocks.{}.mlp.fc2.bias", layer))),
                            ),
                        ),
                        LayerNorm::new(
                            lt(&format!("vision_encoder.encoder.model.visual.blocks.{}.norm1.weight", layer)),
                            Some(lt(&format!("vision_encoder.encoder.model.visual.blocks.{}.norm1.bias", layer))),
                            ln_eps,
                        ),
                        LayerNorm::new(
                            lt(&format!("vision_encoder.encoder.model.visual.blocks.{}.norm2.weight", layer)),
                            Some(lt(&format!("vision_encoder.encoder.model.visual.blocks.{}.norm2.bias", layer))),
                            ln_eps,
                        ),
                    )
                }).collect::<Vec<_>>(),
            LayerNorm::new(lt("vision_encoder.encoder.model.visual.norm.weight"), Some(lt("vision_encoder.encoder.model.visual.norm.bias")), ln_eps),
        );

        let vision_encoder = VisionEncoder::new(projection, transformer);
        Ok(Self {
            vision_encoder,
            text_model,
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32"), feature = "pyo3"))]
mod tests {
    use std::fs;

    use anyhow::Ok;
    use hf_hub::api::sync::Api;
    use ratchet::{shape, test_util::run_py_prg, Device, DeviceRequest, Tensor};
    use ratchet_loader::gguf;
    use ratchet_nn::Module;
    use tokenizers::Tokenizer;

    use crate::moondream::{
        generate::generate, text_model::TextModel, vision_encoder::VisionEncoder,
    };

    use super::Moondream;

    fn vision_ground_truth(tensor: Tensor) -> anyhow::Result<Tensor> {
        let prg = r#"
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def ground(*args):
    tensor = torch.from_numpy(args[0])
    model_id = "vikhyatk/moondream2"
    revision = "2024-05-20"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    return model.encode_image(tensor).numpy()
"#;

        run_py_prg(prg.to_string(), &[&tensor], &[], ratchet::DType::F32)
    }

    #[test]
    #[cfg_attr(feature = "ci", ignore)]
    fn vision_encoder() {
        thread_local! {
            static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
        }

        let api = Api::new().unwrap();
        let model_repo = api.model("ratchet-community/ratchet-moondream-2".to_string());
        let model_path = model_repo.get("moondream_f32.gguf").unwrap();
        let mut reader = std::io::BufReader::new(std::fs::File::open(model_path).unwrap());
        let content = gguf::gguf::Header::read(&mut reader).unwrap();
        let device = GPU_DEVICE.with(|d| d.clone());
        let model = Moondream::load(content, &mut reader, &device).unwrap();

        let input = Tensor::randn::<f32>(shape![1, 3, 378, 378], device);
        let ours = model
            .vision_encoder
            .schedule(input.clone())
            .unwrap()
            .resolve()
            .unwrap()
            .to(&Device::CPU)
            .unwrap();
        let theirs = vision_ground_truth(input.to(&Device::CPU).unwrap()).unwrap();
        ours.all_close(&theirs, 1e-1, 1e-1).unwrap();
    }

    fn end_to_end() {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();

        let api = Api::new().unwrap();
        let model_repo = api.model("ratchet-community/ratchet-moondream-2".to_string());
        let model_path = model_repo.get("moondream_q8_0.gguf").unwrap();
        let mut reader = std::io::BufReader::new(std::fs::File::open(model_path).unwrap());
        let content = gguf::gguf::Header::read(&mut reader).unwrap();
        let mut model = Moondream::load(content, &mut reader, &device).unwrap();

        let tokenizer_path = model_repo.get("tokenizer.json").unwrap();
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

        let img_path = model_repo.get("demo.jpg").unwrap();
        let img = fs::read(img_path).unwrap();

        generate(
            &mut model,
            &img,
            "What is happening here?".to_owned(),
            tokenizer,
            |token| print!("{}", token),
        )
        .unwrap();
    }
}
