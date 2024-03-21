mod whisper;

pub use whisper::*;

use std::collections::HashMap;

pub enum WhisperVariants {
    Tiny,
    Small,
    Large,
}

pub enum LlamaVariants {
    _7B,
    _13B,
}

pub enum AvailableModels {
    Whisper(WhisperVariants),
    Llama(LlamaVariants),
}

/// # ModelID
///
/// Unique identifier for a model on the Hub, e.g `FL33TW00D-HF/whisper-tiny`
pub struct ModelID {
    repo: String,
    model_name: String,
}

pub struct RegisteredModel {
    pub variant: String,
    pub key: ModelID,
}

// # Model Registry
//
// The model registry maps from user-facing AvailableModels to a ModelCollection.
// A ModelCollection contains all of the variants of the specified model e.g (TINY, SMALL, LARGE)
// or (7B, 13B).
pub struct ModelRegistry(HashMap<AvailableModels, Vec<RegisteredModel>>);
