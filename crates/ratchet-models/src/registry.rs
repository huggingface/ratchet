#![allow(non_local_definitions)]
//! # Registry
//!
//! The registry is responsible for surfacing available models to the user in both the CLI & WASM interfaces.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;

#[derive(Debug, Clone)]
#[cfg_attr(
    target_arch = "wasm32",
    derive(tsify::Tsify, serde::Serialize, serde::Deserialize),
    tsify(from_wasm_abi),
    serde(rename_all = "snake_case")
)]
#[cfg_attr(not(target_arch = "wasm32"), derive(clap::ValueEnum))]
pub enum WhisperVariants {
    Tiny,
    Base,
    Small,
    Medium,
    LargeV2,
    LargeV3,
    DistilLargeV3,
}

impl WhisperVariants {
    pub fn repo_id(&self) -> &str {
        match self {
            WhisperVariants::Tiny => "FL33TW00D-HF/whisper-tiny",
            WhisperVariants::Base => "FL33TW00D-HF/whisper-base",
            WhisperVariants::Small => "FL33TW00D-HF/whisper-small",
            WhisperVariants::Medium => "FL33TW00D-HF/whisper-medium",
            WhisperVariants::LargeV2 => "FL33TW00D-HF/whisper-large-v2",
            WhisperVariants::LargeV3 => "FL33TW00D-HF/whisper-large-v3",
            WhisperVariants::DistilLargeV3 => "FL33TW00D-HF/distil-whisper-large-v3",
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(
    target_arch = "wasm32",
    derive(tsify::Tsify, serde::Serialize, serde::Deserialize),
    tsify(from_wasm_abi),
    serde(rename_all = "snake_case")
)]
#[cfg_attr(not(target_arch = "wasm32"), derive(clap::ValueEnum))]
pub enum PhiVariants {
    Phi2,
    Phi3,
}

/// # Available Models
///
/// This is a type safe way to surface models to users,
/// providing autocomplete **within** model families.
#[derive(Debug, Clone)]
#[non_exhaustive]
#[cfg_attr(
    target_arch = "wasm32",
    derive(tsify::Tsify, serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(target_arch = "wasm32", tsify(from_wasm_abi))]
pub enum AvailableModels {
    Whisper(WhisperVariants),
    Phi(PhiVariants),
}

impl AvailableModels {
    pub fn repo_id(&self) -> String {
        let id = match self {
            AvailableModels::Whisper(w) => w.repo_id(),
            AvailableModels::Phi(p) => match p {
                PhiVariants::Phi2 => "FL33TW00D-HF/phi2",
                PhiVariants::Phi3 => "FL33TW00D-HF/phi3",
            },

            _ => unimplemented!(),
        };
        id.to_string()
    }

    pub fn model_id(&self, quantization: Quantization) -> String {
        let model_stem = match self {
            AvailableModels::Whisper(w) => match w {
                WhisperVariants::Tiny => "tiny",
                WhisperVariants::Base => "base",
                WhisperVariants::Small => "small",
                WhisperVariants::Medium => "medium",
                WhisperVariants::LargeV2 => "large-v2",
                WhisperVariants::LargeV3 => "large-v3",
                WhisperVariants::DistilLargeV3 => "distil-large-v3",
            },
            AvailableModels::Phi(p) => match p {
                PhiVariants::Phi2 => "phi2",
                PhiVariants::Phi3 => "phi3-mini-4k",
            },
            _ => unimplemented!(),
        };
        match quantization {
            Quantization::Q8_0 => format!("{}_q8_0.gguf", model_stem),
            Quantization::F32 => format!("{}_f32.gguf", model_stem),
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
#[cfg_attr(not(target_arch = "wasm32"), derive(clap::ValueEnum))]
pub enum Quantization {
    Q8_0,
    F32,
}
