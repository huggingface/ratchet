use crate::{DecodingOptions, Language, Whisper, N_FRAMES};

pub async fn transcribe(
    model: &Whisper,
    audio: Vec<f32>,
    mut decode_options: DecodingOptions,
) -> anyhow::Result<()> {
    let mel = model.specgen.generate(audio)?.to(&model.device)?;
    let content_frames = mel.shape()[mel.rank() - 1] - N_FRAMES;

    if decode_options.language.is_none() {
        if !model.is_multilingual() {
            log::error!("No language specified, using English");
            decode_options.language = Some(Language::String("en".to_string()));
        } else {
            log::error!("No language specified, using language detection");
            let mel = mel.slice(&[])?;
            decode_options.language = Some(model.detect_language(mel)?);
        }
    }

    Ok(())
}
