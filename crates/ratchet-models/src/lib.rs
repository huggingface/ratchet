pub mod phi2;
pub mod registry;
pub mod whisper;

#[cfg(target_arch = "wasm32")]
#[derive(Debug, derive_new::new)]
pub struct WebTensor {
    ggml_dtype: ratchet_loader::GgmlDType,
    data: js_sys::Uint8Array,
    shape: ratchet::Shape,
}

#[cfg(target_arch = "wasm32")]
pub type TensorMap = std::collections::HashMap<String, WebTensor>;

#[cfg(target_arch = "wasm32")]
pub fn ratchet_from_gguf_web(
    wt: WebTensor,
    device: &ratchet::Device,
) -> anyhow::Result<ratchet::Tensor> {
    use ratchet_loader::gguf::gguf::ratchet_from_gguf;
    let shape = wt.shape.clone();
    let data = wt.data.to_vec();
    ratchet_from_gguf(wt.ggml_dtype, &data, shape, device)
}
