mod whisper;

pub use whisper::*;

/// Required as `[wasm_bindgen]` does not support generics
#[derive(Debug)]
pub enum WebModel {
    Whisper(whisper::Whisper),
}

impl WebModel {
    #[cfg(target_arch = "wasm32")]
    pub async fn from_bytes(bytes: &[u8]) -> Result<WebModel, String> {
        let whisper = whisper::Whisper::from_bytes(bytes)
            .await
            .map_err(|e| e.to_string())?;
        Ok(WebModel::Whisper(whisper))
    }
}
