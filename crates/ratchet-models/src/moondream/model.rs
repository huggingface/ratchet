use image::{DynamicImage, ImageFormat};
use ndarray::Axis;
use ndarray_stats::QuantileExt;
use ratchet::{shape, Device, Tensor};
use ratchet_nn::Module;
use tokenizers::Tokenizer;

use super::{text_model::Phi2, vision_encoder::VisionEncoder};

struct Moondream {
    vision_encoder: VisionEncoder,
    text_model: Phi2,
    tokenizer: Tokenizer,
}

impl Moondream {
    fn generate(
        &self,
        image_bytes: &[u8],
        image_format: ImageFormat,
        prompt: String,
        max_tokens: usize,
        device: Device,
    ) -> anyhow::Result<String> {
        let img = image::load_from_memory_with_format(image_bytes, image_format)
            .unwrap()
            .resize_to_fill(378, 378, image::imageops::FilterType::Triangle); // Adjusted to 378x378

        let pixels: Vec<_> = img
            .to_rgb8()
            .to_vec()
            .iter()
            .map(|&x| (x as f32 / 255.0))
            .collect();

        let img_tensor = Tensor::from_data(&pixels, shape![378, 378, 3], device.clone())
            .permute(&[2, 0, 1])?
            .view(shape![1, 3, 378, 378])?;

        let img_embed = self.vision_encoder.schedule(img_tensor)?.resolve()?;

        let bos_token = self
            .text_model
            .embedding
            .schedule(Tensor::from_data([50256], shape![1], device.clone()))?
            .view(shape![1, 1, 2048])?;

        let mut tokens = self
            .tokenizer
            .encode(prompt, false)
            .unwrap()
            .get_ids()
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();

        let mut generated_tokens = vec![];

        while tokens.len() < max_tokens && *tokens.last().unwrap() != 50256 {
            let input = Tensor::from_data(tokens.clone(), shape![1, tokens.len()], device.clone());
            let mut embeds = self.text_model.embedding.schedule(input).unwrap();
            embeds = Tensor::cat(
                vec![bos_token.clone(), img_embed.clone(), embeds.clone()].into(),
                1,
            )
            .unwrap();
            let result = self
                .text_model
                .schedule(embeds.clone())
                .unwrap()
                .resolve()
                .unwrap();
            let logits = result.to(&Device::CPU).unwrap();
            let next_tokens = logits
                .to_ndarray_view::<f32>()
                .map_axis(Axis(2), |row| row.argmax_skipnan().unwrap())
                .iter()
                .map(|&x| x as i32)
                .collect::<Vec<_>>();

            tokens.extend(next_tokens.clone());
            generated_tokens.extend(next_tokens);
        }

        let u32_toks = generated_tokens
            .iter()
            .map(|&x| x as u32)
            .collect::<Vec<_>>();
        Ok(self.tokenizer.decode(&u32_toks, true).unwrap())
    }
}

mod example {
    use std::fs;

    use hf_hub::api::sync::Api;
    use ratchet::{Device, DeviceRequest};
    use ratchet_loader::gguf;
    use tokenizers::Tokenizer;

    use crate::moondream::{text_model::Phi2, vision_encoder::VisionEncoder};

    use super::Moondream;

    fn end_to_end() {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let img = fs::read("<Insert Local File Here>").unwrap();

        let api = Api::new().unwrap();
        let model_repo = api.model("tgestson/ratchet-moondream2".to_string());

        let model_path = model_repo.get("moondream2-mmproj-f16.gguf").unwrap();
        let mut reader = std::io::BufReader::new(std::fs::File::open(model_path).unwrap());
        let content = gguf::gguf::Header::read(&mut reader).unwrap();
        let vision_encoder = VisionEncoder::load(&content, &mut reader, &device).unwrap();

        let model_path = model_repo.get("moondream2-text-model-f16.gguf").unwrap();
        let mut reader = std::io::BufReader::new(std::fs::File::open(model_path).unwrap());
        let content = gguf::gguf::Header::read(&mut reader).unwrap();
        let text_model = Phi2::load(content, &mut reader, &device).unwrap();

        let tokenizer_repo = api.model("vikhyatk/moondream2".to_string());
        let tokenizer_path = tokenizer_repo.get("tokenizer.json").unwrap();
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

        let prompt = "\n\nQuestion: What is happening here?\n\nAnswer:";

        let model = Moondream {
            vision_encoder: vision_encoder,
            text_model: text_model,
            tokenizer: tokenizer,
        };

        let result = model
            .generate(
                &img,
                image::ImageFormat::Jpeg,
                prompt.to_owned(),
                50,
                device,
            )
            .unwrap();
        print!("{}", result);
    }
}
