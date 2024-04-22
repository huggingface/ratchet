use std::io::{BufRead, Seek};

use ratchet::{Device, Tensor};
use ratchet_loader::gguf::gguf::Header;
use ratchet_nn::{LayerNorm, Module};

use super::{
    config::Config,
    residual_block::{ResidualAttentionBlock, ResidualAttentionBlockInputs},
};

#[cfg(target_arch = "wasm32")]
use {crate::ratchet_from_gguf_web, crate::TensorMap};

#[derive(Debug, derive_new::new)]
struct ConvBlock {
    w: Tensor,
    b: Tensor,
    stride: usize,
    padding: usize,
}

impl Module for ConvBlock {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        input
            .conv1d(
                self.w.clone(),
                Some(self.b.clone()),
                self.stride,
                self.padding,
            )?
            .gelu()
    }
}

#[derive(Debug)]
pub(crate) struct EncoderStem {
    conv1: ConvBlock,
    conv2: ConvBlock,
    pos_embed: Tensor,
}

impl Module for EncoderStem {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let convolved = self.conv2.schedule(self.conv1.schedule(input)?)?;
        convolved.permute(&[0, 2, 1])?.add(self.pos_embed.clone())
    }
}

impl EncoderStem {
    pub fn load<R: BufRead + Seek>(
        header: &Header,
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let lt = |name: &str| {
            let key = format!("model.encoder.{}", name);
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
        let mut lt = |name: &str| {
            let key = format!("model.encoder.{}", name);
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
            conv1: ConvBlock::new(lt("conv1.weight")?, lt("conv1.bias")?, 1, 1),
            conv2: ConvBlock::new(lt("conv2.weight")?, lt("conv2.bias")?, 2, 1),
            pos_embed: lt("embed_positions.weight")?,
        })
    }
}

#[derive(Debug)]
pub struct WhisperEncoder {
    stem: EncoderStem,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
}

impl Module for WhisperEncoder {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let mut x = self.stem.schedule(input)?;
        for block in &self.blocks {
            let input = ResidualAttentionBlockInputs {
                x: x.clone(),
                xa: None,
                mask: None,
                cache: None,
            };
            x = block.schedule(input)?;
        }
        self.ln_post.schedule(x)
    }
}

impl WhisperEncoder {
    #[cfg(target_arch = "wasm32")]
    pub fn from_web(
        header: &Header,
        config: &Config,
        tensor_map: &mut TensorMap,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let stem = EncoderStem::from_web(header, tensor_map, device)?;
        let (n_layers, n_heads) = (config.n_audio_layer, config.n_audio_head);

        let blocks = (0..n_layers)
            .fold(Vec::with_capacity(n_layers as _), |mut blocks, i| {
                blocks.push(ResidualAttentionBlock::from_web(
                    header,
                    tensor_map,
                    i as _,
                    n_heads as _,
                    "encoder",
                    device,
                ));
                blocks
            })
            .into_iter()
            .collect::<Result<_, _>>()?;

        let mut lt = |name: &str| {
            let key = format!("model.encoder.layer_norm.{}", name);
            let wt = tensor_map.remove(&key).unwrap();
            ratchet_from_gguf_web(wt, device)
        };

        Ok(Self {
            stem,
            blocks,
            ln_post: LayerNorm::new(lt("weight")?, Some(lt("bias")?), 1e-5),
        })
    }

    pub fn load<R: BufRead + Seek>(
        header: &Header,
        config: &Config,
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let stem = EncoderStem::load(header, reader, device)?;
        let (n_layers, n_heads) = (config.n_audio_layer, config.n_audio_head);

        let blocks = (0..n_layers)
            .fold(Vec::with_capacity(n_layers as _), |mut blocks, i| {
                blocks.push(ResidualAttentionBlock::load(
                    header,
                    reader,
                    i as _,
                    n_heads as _,
                    "encoder",
                    device,
                ));
                blocks
            })
            .into_iter()
            .collect::<Result<_, _>>()?;

        let mut lt = |name: &str| {
            let key = format!("model.encoder.layer_norm.{}", name);
            header.tensor(reader, &key, device)
        };

        Ok(Self {
            stem,
            blocks,
            ln_post: LayerNorm::new(lt("weight")?, Some(lt("bias")?), 1e-5),
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use hf_hub::api::sync::Api;
    use ratchet::{Device, DeviceRequest, Tensor};
    use ratchet_loader::gguf::gguf;
    use ratchet_nn::Module;

    use crate::whisper::{config::Config, encoder::WhisperEncoder};

    fn log_init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn encoder_matches() -> anyhow::Result<()> {
        log_init();
        let api = Api::new().unwrap();
        let model = api.model("FL33TW00D-HF/whisper-tiny".to_string());
        let model_path = model.get("tiny_f32.gguf").unwrap();
        println!("Path: {}", model_path.display());
        let config_path = model.get("config.json").unwrap();

        let dataset = api.dataset("FL33TW00D-HF/ratchet-util".to_string());
        let input_npy = dataset.get("jfk_tiny_encoder_input.npy").unwrap();
        let ground_npy = dataset.get("jfk_tiny_encoder_hs.npy").unwrap();

        let mut reader = std::io::BufReader::new(std::fs::File::open(model_path).unwrap());
        let header = gguf::Header::read(&mut reader).unwrap();
        let config: Config = serde_json::from_slice(&std::fs::read(config_path).unwrap()).unwrap();
        let device = Device::request_device(DeviceRequest::GPU).unwrap();

        let encoder = WhisperEncoder::load(&header, &config, &mut reader, &device)?;
        let input = Tensor::from_npy_path::<f32, _>(input_npy, &device)?;

        let result = encoder.schedule(input)?.resolve()?;
        let ours = result.to(&Device::CPU)?;
        let ground = Tensor::from_npy_path::<f32, _>(ground_npy, &Device::CPU)?;
        println!("OURS: {:#?}", ours);
        println!("Ground: {:#?}", ground);
        ground.all_close(&ours, 1e-3, 1e-3)?;

        Ok(())
    }
}
