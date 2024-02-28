#![cfg(target_arch = "wasm32")]
use db::StoredModel;
use ratchet_models::{transcribe, DecodingOptions, Whisper};
use wasm_bindgen::JsValue;

mod db;
mod model;

#[derive(Debug)]
pub enum WebModel {
    Whisper(Whisper),
    //ImageModel
}

impl WebModel {
    pub async fn run(&self, input: JsValue) -> Result<JsValue, anyhow::Error> {
        match self {
            WebModel::Whisper(model) => {
                let input: WhisperInputs = serde_wasm_bindgen::from_value(input)?;
                model.run(input).await
            }
        }
    }

    pub async fn from_stored(stored: StoredModel) -> Result<WebModel, anyhow::Error> {
        match stored.repo_id.as_str() {
            "ggerganov/whisper.cpp" => {
                let whisper = Whisper::from_bytes(&stored.bytes.to_vec()).await?;
                Ok(WebModel::Whisper(whisper))
            }
            _ => Err(anyhow::anyhow!("Unknown model type")),
        }
    }
}

#[async_trait::async_trait]
pub trait WebCompatible {
    type Input: serde::de::DeserializeOwned;

    async fn run(&mut self, input: Self::Input) -> Result<JsValue, JsValue>;
}

#[derive(serde::Deserialize)]
pub struct WhisperInputs {
    pub audio: Vec<f32>,
    pub decode_options: DecodingOptions,
}

impl WebCompatible for Whisper {
    type Input = WhisperInputs;

    async fn run(&mut self, input: Self::Input) -> Result<JsValue, JsValue> {
        let WhisperInputs {
            audio,
            decode_options,
        } = input;
        let result = transcribe(self, audio, decode_options).await.unwrap();
        serde_wasm_bindgen::to_value(&result).map_err(|e| e.into())
    }
}
