use crate::db::ModelKey;
use wasm_bindgen::prelude::*;

#[derive(Debug)]
#[wasm_bindgen]
pub enum AvailableModels {
    WHISPER_TINY,
    WHISPER_BASE,
    WHISPER_SMALL,
    WHISPER_MEDIUM,
    WHISPER_LARGE_V2,
    WHISPER_LARGE_V3,
}

impl AvailableModels {
    fn repo_id(&self) -> String {
        match self {
            _ => "FL33TW00D-HF/ratchet-whisper".to_string(),
        }
    }

    fn model_id(&self, quantization: Quantization) -> String {
        let model_stem = match self {
            AvailableModels::WHISPER_TINY => "tiny",
            AvailableModels::WHISPER_BASE => "base",
            AvailableModels::WHISPER_SMALL => "small",
            AvailableModels::WHISPER_MEDIUM => "medium",
            AvailableModels::WHISPER_LARGE_V2 => "large-v2",
            AvailableModels::WHISPER_LARGE_V3 => "large-v3",
        };
        match quantization {
            Quantization::Q8 => format!("{}_q8.bin", model_stem),
            Quantization::F32 => format!("{}_f32.bin", model_stem),
        }
    }

    pub fn as_key(&self, quantization: Quantization) -> ModelKey {
        ModelKey::new(self.repo_id(), self.model_id(quantization))
    }
}

#[derive(Debug)]
#[wasm_bindgen]
pub enum Quantization {
    Q8,
    F32,
}
