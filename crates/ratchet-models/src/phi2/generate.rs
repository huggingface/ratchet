#![cfg(target_arch = "wasm32")]
use crate::phi2::Phi2;
use ndarray::Axis;
use ndarray_stats::QuantileExt;
use ratchet::{shape, Device, Tensor};
use ratchet_nn::Module;
use tokenizers::Tokenizer;

pub async fn generate(
    model: &mut Phi2,
    tokenizer: Tokenizer,
    prompt: String,
) -> anyhow::Result<()> {
    use web_time::Instant;
    log::warn!("Prompt: {}", prompt);
    let encoding = tokenizer.encode(prompt, true).unwrap();
    let mut tokens = encoding
        .get_ids()
        .iter()
        .map(|&x| x as i32)
        .collect::<Vec<_>>();
    let mut all_logits = vec![];
    let mut all_tokens = tokens.clone();
    let mut loop_cnt = 0;
    let start = Instant::now();
    while tokens[tokens.len() - 1] != 50256 && loop_cnt < 256 {
        let input = Tensor::from_data(
            tokens.clone(),
            shape![1, tokens.len()],
            model.device.clone(),
        );
        let result = model.schedule(input)?.resolve()?;
        let logits = result.to(&Device::CPU).await?;
        all_logits.push(logits.clone());
        model.cache_mut().update(tokens.len());

        tokens = logits
            .to_ndarray_view::<f32>()
            .map_axis(Axis(2), |row| row.argmax_skipnan().unwrap())
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        let u32_toks = tokens.iter().map(|&x| x as u32).collect::<Vec<_>>();
        log::warn!("{}", tokenizer.decode(&u32_toks, true).unwrap());
        all_tokens.extend(tokens.clone());
        loop_cnt += 1;
    }
    let elapsed = start.elapsed();
    log::warn!("Elapsed: {:?}", elapsed);
    log::warn!("Tok/s {}", all_tokens.len() as f64 / elapsed.as_secs_f64());
    Ok(())
}
