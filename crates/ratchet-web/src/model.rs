#![cfg(target_arch = "wasm32")]
use ratchet_models::WebModel;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Model {
    inner: WebModel,
}

impl Model {
    pub async fn from_bytes(bytes: &[u8]) -> Result<Model, anyhow::Error> {
        let inner = WebModel::from_bytes(bytes).await.unwrap();
        Ok(Model { inner })
    }
}

#[wasm_bindgen]
#[derive(Default)]
pub struct ModelBuilder {
    id: Option<String>,
    model_file: Option<Vec<u8>>,
    tokenizer: Option<Vec<u8>>,
}

#[wasm_bindgen]
impl ModelBuilder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    #[wasm_bindgen(js_name = "setId")]
    pub fn id(mut self, id: String) -> Self {
        self.id = Some(id);
        self
    }

    #[wasm_bindgen(js_name = "setTokenizer")]
    pub fn tokenizer_bytes(mut self, tokenizer_bytes: Vec<u8>) -> Self {
        self.tokenizer = Some(tokenizer_bytes);
        self
    }

    #[wasm_bindgen(js_name = "setModel")]
    pub fn model_file(mut self, model_file: Vec<u8>) -> Self {
        self.model_file = Some(model_file);
        self
    }

    pub async fn build(self) -> Result<Model, JsError> {
        let model_file = self
            .model_file
            .ok_or(JsError::new("model file is required"))?;
        let m = Model::from_bytes(&model_file)
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(m)
    }
}
