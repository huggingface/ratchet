#![cfg(target_arch = "wasm32")]
use ratchet_models::WebModel;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Model {
    inner: WebModel,
}

impl Model {
    /// The main JS entrypoint into the library.
    ///
    /// Loads a model with the provided ID.
    /// This id should be an enum!
    pub async fn load(id: String) -> Model {
        todo!()
    }

    async fn from_bytes(bytes: &[u8]) -> Result<Model, anyhow::Error> {
        let inner = WebModel::from_bytes(bytes).await.unwrap();
        Ok(Model { inner })
    }
}
