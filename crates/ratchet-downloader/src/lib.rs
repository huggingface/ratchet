#[cfg(test)]
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

use gloo::console::error as log_error;
use wasm_bindgen::{prelude::*, JsValue};

mod fetch;
pub mod huggingface;

#[cfg(test)]
wasm_bindgen_test_configure!(run_in_browser);

pub struct Model {
    url: String,
}

impl Model {
    fn from_hf(repo_id: String) -> Self {
        Self {
            url: format!("https://huggingface.co/{}/resolve/main", repo_id),
        }
    }

    fn from_hf_with_revision(repo_id: String, revision: String) -> Self {
        Self {
            url: format!("https://huggingface.co/{repo_id}/resolve/{revision}"),
        }
    }

    fn from_custom(url: String) -> Self {
        Self { url }
    }

    async fn get(&self, file_name: String) -> Result<(), JsError> {
        let file_url = format!("{}/{}", self.url, file_name);
        // let response = fetch::fetch(file_url.as_str()).await?;

        let res = reqwest::Client::new().get(file_url).send().await?;
        Ok(())
    }
}

#[cfg(test)]
#[wasm_bindgen_test]
async fn pass() -> Result<(), JsValue> {
    use js_sys::JsString;

    let model = Model::from_hf("jantxu/ratchet-test".to_string());
    let file = model
        .get("model.safetensors".to_string())
        .await
        .map_err(|err| {
            log_error!(err);
            JsString::from("Failed to download file")
        });
    Ok(())
}
