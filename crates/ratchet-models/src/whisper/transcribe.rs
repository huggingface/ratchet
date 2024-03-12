use crate::StreamedSegment;
use crate::{
    DecodingOptions, DecodingTask, Language, Prompt, TranscriptionResult, Whisper,
    WhisperTokenizer, HOP_LENGTH, N_AUDIO_CTX, N_FRAMES, SAMPLE_RATE,
};
use ratchet_nn::Module;
use std::cmp::min;
use web_time::Instant;

#[cfg(not(target_arch = "wasm32"))]
pub fn transcribe(
    model: &mut Whisper,
    audio: Vec<f32>,
    mut decode_options: DecodingOptions,
) -> anyhow::Result<TranscriptionResult> {
    let n_mels = model.hparams.n_mels as usize;
    let runtime = Instant::now();
    let mel = model.specgen.generate(audio)?.to(&model.device)?;
    let content_frames = mel.shape()[mel.rank() - 1] - N_FRAMES;

    if decode_options.language.is_none() {
        if !model.is_multilingual() {
            log::warn!("No language specified, using English");
            decode_options.language = Some(Language::String("en".to_string()));
        } else {
            log::warn!("No language specified, using language detection");
            let mel = mel.slice(&[0..1, 0..n_mels, 0..3000])?;
            decode_options.language = Some(model.detect_language(mel)?);
        }
    }

    let language = decode_options.language.as_ref().unwrap();
    let task = decode_options.task;
    let tokenizer = WhisperTokenizer::load(None, n_mels == 128, language.clone(), task);

    let mut seek = 0;
    let input_stride = N_FRAMES / N_AUDIO_CTX;
    let mut all_tokens = Vec::with_capacity(512);
    let mut all_segments = Vec::with_capacity(512);
    let prompt_since_reset = 0;

    let mut pass_idx = 0;
    while seek < content_frames {
        model.device.try_gpu()?.begin_pass(pass_idx);
        println!("seek: {}", seek);
        let mut decode_options = decode_options.clone();
        let time_offset = (seek * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
        decode_options.time_offset = Some(time_offset);
        let mel_segment = mel.slice(&[0..1, 0..n_mels, seek..(seek + N_FRAMES)])?;
        log::info!(
            "processing segment - from: {}, to: {}",
            seek,
            seek + N_FRAMES
        );

        let segment_size = min(N_FRAMES, content_frames - seek);
        let segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE;

        if !all_tokens.is_empty() {
            decode_options.prompt = Some(Prompt::Tokens(all_tokens[prompt_since_reset..].to_vec()));
        }

        let hs = model.encoder.forward(&mel_segment)?.resolve()?;

        let task = DecodingTask::new(decode_options, tokenizer.clone());
        let decoded = task.run(&mut model.decoder, hs)?;
        let (segments, advance) = DecodingTask::build_segments(
            &tokenizer,
            decoded,
            time_offset,
            segment_size,
            segment_duration,
            input_stride,
        );
        let all_segment_tokens = segments
            .iter()
            .flat_map(|s| s.tokens.iter().copied())
            .map(|x| x as i32)
            .collect::<Vec<_>>();
        all_tokens.extend(all_segment_tokens);
        all_segments.extend(segments);
        model.decoder.reset();
        seek += advance;
        pass_idx += 1;
    }

    let mut t = TranscriptionResult::new(runtime.elapsed(), all_segments, None);
    t.generate_formatted(&tokenizer);
    Ok(t)
}

#[cfg(target_arch = "wasm32")]
pub async fn transcribe(
    model: &mut Whisper,
    audio: Vec<f32>,
    mut decode_options: DecodingOptions,
    callback: Option<impl Fn(StreamedSegment)>,
) -> anyhow::Result<TranscriptionResult> {
    let runtime = Instant::now();
    let n_mels = model.hparams.n_mels as usize;
    let mel = model.specgen.generate(audio)?.to(&model.device).await?;
    let content_frames = mel.shape()[mel.rank() - 1] - N_FRAMES;

    if decode_options.language.is_none() {
        if !model.is_multilingual() {
            log::warn!("No language specified, using English");
            decode_options.language = Some(Language::String("en".to_string()));
        } else {
            log::warn!("No language specified, using language detection");
            let mel = mel.slice(&[0..1, 0..n_mels, 0..3000])?;
            decode_options.language = Some(model.detect_language(mel).await?);
        }
    }

    let language = decode_options.language.as_ref().unwrap();
    let task = decode_options.task;
    let tokenizer =
        WhisperTokenizer::load(None, n_mels == 128, language.clone(), task.clone()).await;

    let mut seek = 0;
    let input_stride = N_FRAMES / N_AUDIO_CTX;
    let mut all_tokens = Vec::with_capacity(512);
    let mut all_segments = Vec::with_capacity(512);
    let prompt_since_reset = 0;

    let mut pass_idx = 0;
    while seek < content_frames {
        model.device.try_gpu()?.begin_pass(pass_idx);
        println!("seek: {}", seek);
        let mut decode_options = decode_options.clone();
        let time_offset = (seek * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
        decode_options.time_offset = Some(time_offset);
        let mel_segment = mel.slice(&[0..1, 0..n_mels, seek..(seek + N_FRAMES)])?;
        log::info!(
            "processing segment - from: {}, to: {}",
            seek,
            seek + N_FRAMES
        );

        let segment_size = min(N_FRAMES, content_frames - seek);
        let segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE;

        if !all_tokens.is_empty() {
            decode_options.prompt = Some(Prompt::Tokens(all_tokens[prompt_since_reset..].to_vec()));
        }

        let hs = model.encoder.forward(&mel_segment)?.resolve()?;

        let task = DecodingTask::new(decode_options, tokenizer.clone());
        let decoded = task.run(&mut model.decoder, hs, &callback).await?;

        let (segments, advance) = DecodingTask::build_segments(
            &tokenizer,
            decoded,
            time_offset,
            segment_size,
            segment_duration,
            input_stride,
        );
        let all_segment_tokens = segments
            .iter()
            .flat_map(|s| s.tokens.iter().copied())
            .map(|x| x as i32)
            .collect::<Vec<_>>();
        all_tokens.extend(all_segment_tokens);
        all_segments.extend(segments);
        model.decoder.reset();
        seek += advance;
        pass_idx += 1;
    }

    if let Some(cb) = callback {
        cb(StreamedSegment::from_tokens(
            &tokenizer,
            &[WhisperTokenizer::EOT],
            content_frames as f64 / 100.,
            true,
        ));
    }

    let mut t = TranscriptionResult::new(runtime.elapsed(), all_segments, None);
    t.generate_formatted(&tokenizer);
    Ok(t)
}
