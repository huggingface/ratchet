#![cfg(target_arch = "wasm32")]
use crate::db::*;
use crate::WebModel;
use ratchet_hub::{ApiBuilder, RepoType};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Debug)]
pub struct Model {
    inner: WebModel,
}

impl Model {
    /// The main JS entrypoint into the library.
    ///
    /// Loads a model with the provided ID.
    /// This id should be an enum of supported models.
    pub async fn load(key: ModelKey) -> Result<Model, JsValue> {
        let model_repo = ApiBuilder::from_hf(&key.repo_id, RepoType::Model).build();
        let db = RatchetDB::open().await.map_err(|e| {
            let e: JsError = e.into();
            Into::<JsValue>::into(e)
        })?;
        log::warn!("Loading model: {:?}", key);
        if let None = db.get_model(&key).await.map_err(|e| {
            let e: JsError = e.into();
            Into::<JsValue>::into(e)
        })? {
            log::warn!("Model not found in db, fetching from remote");
            let model_data = model_repo.get(&key.model_id).await?;
            let bytes = model_data.to_uint8().await?;
            let model = StoredModel::new(&key, bytes);
            db.put_model(&key, model).await.unwrap();
        }
        let model = db.get_model(&key).await.unwrap().unwrap();
        Ok(Model {
            inner: WebModel::from_stored(model).await.unwrap(),
        })
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    use super::*;
    use ratchet_hub::{ApiBuilder, RepoType};
    use wasm_bindgen_test::*;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    fn log_init() {
        console_error_panic_hook::set_once();
        log::set_max_level(log::LevelFilter::Off);
        console_log::init_with_level(log::Level::Warn).unwrap();
    }

    #[wasm_bindgen_test]
    async fn can_we_load_from_db() -> Result<(), JsValue> {
        log_init();
        let key = ModelKey::new("ggerganov/whisper.cpp", "ggml-tiny.bin");
        let model = Model::load(key).await.unwrap();
        log::warn!("Model: {:?}", model);
        Ok(())
    }
}
