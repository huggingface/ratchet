#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    #[serde(alias = "num_mel_bins")]
    pub n_mels: usize,
    #[serde(alias = "max_source_positions")]
    pub n_audio_ctx: usize,
    #[serde(alias = "d_model")]
    pub n_audio_state: usize,
    #[serde(alias = "encoder_attention_heads")]
    pub n_audio_head: usize,
    #[serde(alias = "encoder_layers")]
    pub n_audio_layer: usize,
    #[serde(alias = "vocab_size")]
    pub n_vocab: usize,
    #[serde(alias = "max_target_positions")]
    pub n_text_ctx: usize,
    #[serde(alias = "decoder_attention_heads")]
    pub n_text_head: usize,
    #[serde(alias = "decoder_layers")]
    pub n_text_layer: usize,
    #[serde(default)]
    pub suppress_tokens: Vec<u32>,
}
