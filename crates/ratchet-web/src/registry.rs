use crate::db::ModelKey;
use wasm_bindgen::prelude::*;

#[derive(Debug)]
#[wasm_bindgen]
pub enum AvailableModels {
    WhisperTiny,
    WhisperBase,
    WhisperSmall,
    WhisperMedium,
    WhisperLargeV2,
    WhisperLargeV3,
    DistilWhisperLargeV3,
}

impl AvailableModels {
    fn repo_id(&self) -> String {
        let id = match self {
            AvailableModels::WhisperTiny => "FL33TW00D-HF/whisper-tiny",
            AvailableModels::WhisperBase => "FL33TW00D-HF/whisper-base",
            AvailableModels::WhisperSmall => "FL33TW00D-HF/whisper-small",
            AvailableModels::WhisperMedium => "FL33TW00D-HF/whisper-medium",
            AvailableModels::WhisperLargeV2 => "FL33TW00D-HF/whisper-large-v2",
            AvailableModels::WhisperLargeV3 => "FL33TW00D-HF/whisper-large-v3",
            AvailableModels::DistilWhisperLargeV3 => "FL33TW00D-HF/distil-whisper-large-v3",
        };
        id.to_string()
    }

    fn model_id(&self, quantization: Quantization) -> String {
        let model_stem = match self {
            AvailableModels::WhisperTiny => "tiny",
            AvailableModels::WhisperBase => "base",
            AvailableModels::WhisperSmall => "small",
            AvailableModels::WhisperMedium => "medium",
            AvailableModels::WhisperLargeV2 => "large-v2",
            AvailableModels::WhisperLargeV3 => "large-v3",
            AvailableModels::DistilWhisperLargeV3 => "distil-large-v3",
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
