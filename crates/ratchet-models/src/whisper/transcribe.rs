use std::cmp::min;

use ratchet_nn::Module;

use crate::{
    DecodingOptions, DecodingTask, Language, Prompt, Whisper, HOP_LENGTH, N_AUDIO_CTX, N_FRAMES,
    SAMPLE_RATE,
};

pub async fn transcribe(
    model: &Whisper,
    audio: Vec<f32>,
    mut decode_options: DecodingOptions,
) -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    let mel = model.specgen.generate(audio)?.to(&model.device)?;
    #[cfg(target_arch = "wasm32")]
    let mel = model.specgen.generate(audio)?.to(&model.device).await?;

    let content_frames = mel.shape()[mel.rank() - 1] - N_FRAMES;

    if decode_options.language.is_none() {
        if !model.is_multilingual() {
            log::error!("No language specified, using English");
            decode_options.language = Some(Language::String("en".to_string()));
        } else {
            log::error!("No language specified, using language detection");
            let mel = mel.slice(&[0..80, 0..3000])?;
            decode_options.language = Some(model.detect_language(mel)?);
        }
    }

    let _language = decode_options.language.as_ref().unwrap();
    let _task = decode_options.task;

    let seek = 0;
    let all_tokens = Vec::with_capacity(512);
    let _input_stride = N_FRAMES / N_AUDIO_CTX;
    let prompt_since_reset = 0;

    while seek < content_frames {
        let mut decode_options = decode_options.clone();
        let time_offset = (seek * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
        decode_options.time_offset = Some(time_offset);
        let mel_segment = mel.slice(&[0..80, 0..3000])?;
        log::info!(
            "processing segment - from: {}, to: {}",
            seek,
            seek + N_FRAMES
        );

        let segment_size = min(N_FRAMES, content_frames - seek);
        let _segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE;

        if !all_tokens.is_empty() {
            decode_options.prompt = Some(Prompt::Tokens(all_tokens[prompt_since_reset..].to_vec()));
        }

        let hs = model.encoder.forward(&mel_segment)?;
        hs.resolve();

        let task = DecodingTask::new(decode_options, &model.tokenizer);
        let decoded = task.run(&model.decoder, &hs, &model.tokenizer)?;
        println!("{}: {:?}", time_offset, decoded);
    }

    Ok(())
}
