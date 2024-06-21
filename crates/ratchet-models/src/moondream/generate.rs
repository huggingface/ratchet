use super::model::Moondream;
use crate::TokenOutputStream;
use ndarray::Axis;
use ndarray_stats::QuantileExt;
use ratchet::shape;
use ratchet::Device;
use ratchet::Tensor;
use ratchet_nn::Module;
use tokenizers::Tokenizer;

#[cfg(not(target_arch = "wasm32"))]
pub fn generate(
    model: &mut Moondream,
    image_bytes: &[u8],
    question: String,
    tokenizer: Tokenizer,
    callback: impl Fn(String),
) -> anyhow::Result<()> {
    use ratchet::rvec;
    use web_time::Instant;
    let device = model.text_model.device.clone();

    let prompt = format!("\n\nQuestion: {}\n\nAnswer:", question);
    log::warn!("Prompt: {}", prompt);

    let mut tos = TokenOutputStream::new(tokenizer);

    let img = image::io::Reader::new(std::io::Cursor::new(image_bytes))
        .with_guessed_format()?
        .decode()
        .unwrap()
        .resize_to_fill(378, 378, image::imageops::FilterType::Triangle); // Adjusted to 378x378

    let pixels: Vec<_> = img
        .to_rgb8()
        .to_vec()
        .iter()
        .map(|&x| (x as f32 / 255.0))
        .collect();

    let img_tensor = Tensor::from_data(pixels, shape![378, 378, 3], device.clone())
        .permute(&[2, 0, 1])?
        .view(shape![1, 3, 378, 378])?
        .cast(device.compute_precision())?;

    let img_embed = model.vision_encoder.schedule(img_tensor)?.resolve()?;

    let bos_token = model
        .text_model
        .embedding
        .schedule(Tensor::from_data([50256], shape![1], device.clone()))?
        .view(shape![1, 1, 2048])?;

    let encoding = tos.tokenizer().encode(prompt, false).unwrap();

    let mut tokens = encoding
        .get_ids()
        .iter()
        .map(|&x| x as i32)
        .collect::<Vec<_>>();

    let mut all_tokens = tokens.clone();

    let start = Instant::now();
    let mut generated_tokens = vec![];
    while *tokens.last().unwrap() != 50256 {
        let input = Tensor::from_data(tokens.clone(), shape![1, tokens.len()], device.clone());
        let mut embeds: Tensor;
        if generated_tokens.is_empty() {
            embeds = model.text_model.embedding.schedule(input)?;
            embeds = Tensor::cat(
                rvec![bos_token.clone(), img_embed.clone(), embeds.clone()],
                1,
            )?;
        } else {
            embeds = model.text_model.embedding.schedule(input).unwrap();
        }

        let result = model
            .text_model
            .schedule(embeds.clone())?
            .full()?
            .resolve()?;

        model.text_model.cache_mut().update(embeds.shape()[1]);

        let logits = result.to(&Device::CPU).unwrap();
        let next_tokens = logits
            .to_ndarray_view::<f32>()
            .map_axis(Axis(2), |row| row.argmax_skipnan().unwrap())
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        tokens = next_tokens.clone();
        generated_tokens.extend(next_tokens.clone());
        all_tokens.extend(next_tokens.clone());

        if let Some(t) = tos.next_token(next_tokens[0] as u32)? {
            callback(t);
        }
    }

    let elapsed = start.elapsed();
    log::warn!("Elapsed: {:?}", elapsed);
    log::warn!("Tok/s {}", all_tokens.len() as f64 / elapsed.as_secs_f64());
    model.text_model.reset();
    Ok(())
}

#[cfg(target_arch = "wasm32")]
pub async fn generate(
    model: &mut Moondream,
    image_bytes: Vec<u8>,
    question: String,
    tokenizer: Tokenizer,
    callback: impl Fn(String),
) -> anyhow::Result<()> {
    use web_time::Instant;
    let device = model.text_model.device.clone();

    let img = image::io::Reader::new(std::io::Cursor::new(image_bytes))
        .with_guessed_format()?
        .decode()
        .unwrap()
        .resize_to_fill(378, 378, image::imageops::FilterType::Triangle); // Adjusted to 378x378

    let prompt = format!("\n\nQuestion: {}\n\nAnswer:", question);
    log::warn!("Prompt: {}", prompt);

    let mut tos = TokenOutputStream::new(tokenizer);

    let pixels: Vec<_> = img
        .to_rgb8()
        .to_vec()
        .iter()
        .map(|&x| (x as f32 / 255.0))
        .collect();

    let img_tensor = Tensor::from_data(&pixels, shape![378, 378, 3], device.clone())
        .permute(&[2, 0, 1])?
        .view(shape![1, 3, 378, 378])?;

    let img_embed = model.vision_encoder.schedule(img_tensor)?.resolve()?;

    let bos_token = model
        .text_model
        .embedding
        .schedule(Tensor::from_data([50256], shape![1], device.clone()))?
        .view(shape![1, 1, 2048])?;

    let encoding = tos.tokenizer().encode(prompt, false).unwrap();

    let mut tokens = encoding
        .get_ids()
        .iter()
        .map(|&x| x as i32)
        .collect::<Vec<_>>();

    let mut all_tokens = tokens.clone();

    let start = Instant::now();
    let mut generated_tokens = vec![];
    while *tokens.last().unwrap() != 50256 {
        let input = Tensor::from_data(tokens.clone(), shape![1, tokens.len()], device.clone());
        let mut embeds: Tensor;
        if generated_tokens.len() == 0 {
            embeds = model.text_model.embedding.schedule(input).unwrap();
            embeds = Tensor::cat(
                vec![bos_token.clone(), img_embed.clone(), embeds.clone()].into(),
                1,
            )
            .unwrap();
        } else {
            embeds = model.text_model.embedding.schedule(input).unwrap();
        }

        let result = model
            .text_model
            .schedule(embeds.clone())?
            .full()?
            .resolve()?;

        model.text_model.cache_mut().update(embeds.shape()[1]);

        let logits = result.to(&Device::CPU).await.unwrap();
        let next_tokens = logits
            .to_ndarray_view::<f32>()
            .map_axis(Axis(2), |row| row.argmax_skipnan().unwrap())
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        tokens = next_tokens.clone();
        generated_tokens.extend(next_tokens.clone());
        all_tokens.extend(next_tokens.clone());

        if let Some(t) = tos.next_token(next_tokens[0] as u32)? {
            callback(t);
        }
    }

    let elapsed = start.elapsed();
    log::warn!("Elapsed: {:?}", elapsed);
    log::warn!("Tok/s {}", all_tokens.len() as f64 / elapsed.as_secs_f64());
    println!("Tok/s {}", all_tokens.len() as f64 / elapsed.as_secs_f64());
    model.text_model.reset();
    Ok(())
}
