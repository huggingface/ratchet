use crate::phi3::Phi3;
use crate::TokenOutputStream;
use ndarray::Axis;
use ndarray_stats::QuantileExt;
use ratchet::{shape, Device, Tensor};
use ratchet_nn::Module;
use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
pub async fn generate(
    model: &mut Phi3,
    tokenizer: Tokenizer,
    prompt: String,
    callback: impl Fn(String),
) -> anyhow::Result<()> {
    use web_time::Instant;
    log::warn!("Prompt: {}", prompt);

    let prompt = format!(
        r#"<|user|>
{}<|end|>
<|assistant|>"#,
        prompt
    );

    let mut tos = TokenOutputStream::new(tokenizer);

    let encoding = tos.tokenizer().encode(prompt, true).unwrap();
    let mut tokens = encoding
        .get_ids()
        .iter()
        .map(|&x| x as i32)
        .collect::<Vec<_>>();
    tokens.insert(0, 1);
    let mut all_tokens = tokens.clone();
    let start = Instant::now();
    while tokens[tokens.len() - 1] != 32007 && all_tokens.len() < 2048 {
        let input = Tensor::from_data(
            tokens.clone(),
            shape![1, tokens.len()],
            model.device.clone(),
        );
        let result = model.schedule(input)?.resolve()?;
        let logits = result.to(&Device::CPU).await?;
        model.cache_mut().update(tokens.len());

        tokens = logits
            .to_ndarray_view::<f32>()
            .map_axis(Axis(2), |row| row.argmax_skipnan().unwrap())
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        all_tokens.extend(tokens.clone());
        if let Some(t) = tos.next_token(tokens[0] as u32)? {
            callback(t);
        }
    }
    let elapsed = start.elapsed();
    log::warn!("Elapsed: {:?}", elapsed);
    log::warn!("Tok/s {}", all_tokens.len() as f64 / elapsed.as_secs_f64());
    model.reset();
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn generate(
    model: &mut Phi3,
    tokenizer: Tokenizer,
    prompt: String,
    callback: impl Fn(String),
) -> anyhow::Result<()> {
    use web_time::Instant;
    log::warn!("Prompt: {}", prompt);

    let prompt = format!(
        r#"<|user|>
{}<|end|>
<|assistant|>"#,
        prompt
    );

    let mut tos = TokenOutputStream::new(tokenizer);

    let encoding = tos.tokenizer().encode(prompt, true).unwrap();
    let mut tokens = encoding
        .get_ids()
        .iter()
        .map(|&x| x as i32)
        .collect::<Vec<_>>();
    tokens.insert(0, 1);
    let mut all_tokens = tokens.clone();
    let start = Instant::now();
    while tokens[tokens.len() - 1] != 32007 && all_tokens.len() < 2048 {
        let input = Tensor::from_data(
            tokens.clone(),
            shape![1, tokens.len()],
            model.device.clone(),
        );
        let result = model.schedule(input)?.resolve()?;
        let logits = result.to(&Device::CPU)?;
        model.cache_mut().update(tokens.len());

        tokens = logits
            .to_ndarray_view::<f32>()
            .map_axis(Axis(2), |row| row.argmax_skipnan().unwrap())
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        all_tokens.extend(tokens.clone());
        if let Some(t) = tos.next_token(tokens[0] as u32)? {
            callback(t);
        }
    }
    let elapsed = start.elapsed();
    log::warn!("Elapsed: {:?}", elapsed);
    log::warn!("Tok/s {}", all_tokens.len() as f64 / elapsed.as_secs_f64());
    model.reset();
    Ok(())
}
