use crate::db::ModelKey;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Debug, tsify::Tsify, Serialize, Deserialize)]
#[tsify(from_wasm_abi)]
#[serde(rename_all = "snake_case")]
pub enum Whisper {
    Tiny,
    Base,
    Small,
    Medium,
    LargeV2,
    LargeV3,
    DistilLargeV3,
}

/// Not actually supported, placeholder.
#[derive(Debug, tsify::Tsify, Serialize, Deserialize)]
#[tsify(from_wasm_abi)]
pub enum Llama {
    #[serde(rename = "7B")]
    _7B,
    #[serde(rename = "13B")]
    _13B,
}

#[derive(Debug, tsify::Tsify, Serialize, Deserialize)]
#[tsify(from_wasm_abi)]
pub enum AvailableModels {
    Whisper(Whisper),
    Llama(Llama),
}

impl AvailableModels {
    fn repo_id(&self) -> String {
        let id = match self {
            AvailableModels::Whisper(w) => match w {
                Whisper::Tiny => "FL33TW00D-HF/whisper-tiny",
                Whisper::Base => "FL33TW00D-HF/whisper-base",
                Whisper::Small => "FL33TW00D-HF/whisper-small",
                Whisper::Medium => "FL33TW00D-HF/whisper-medium",
                Whisper::LargeV2 => "FL33TW00D-HF/whisper-large-v2",
                Whisper::LargeV3 => "FL33TW00D-HF/whisper-large-v3",
                Whisper::DistilLargeV3 => "FL33TW00D-HF/distil-whisper-large-v3",
            },
            _ => unimplemented!(),
        };
        id.to_string()
    }

    fn model_id(&self, quantization: Quantization) -> String {
        let model_stem = match self {
            AvailableModels::Whisper(w) => match w {
                Whisper::Tiny => "tiny",
                Whisper::Base => "base",
                Whisper::Small => "small",
                Whisper::Medium => "medium",
                Whisper::LargeV2 => "large-v2",
                Whisper::LargeV3 => "large-v3",
                Whisper::DistilLargeV3 => "distil-large-v3",
            },
            _ => unimplemented!(),
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
