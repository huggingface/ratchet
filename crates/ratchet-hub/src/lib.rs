use gloo_net::http::Request;
use js_sys::{Object, Reflect, Uint8Array};
use util::{js_error, js_to_js_error};
use wasm_bindgen::{prelude::*, JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::RequestMode;

mod util;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let logger = fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "{}[{}][{}] {}",
                chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"),
                record.target(),
                record.level(),
                message
            ))
        })
        .level_for("tokenizers", log::LevelFilter::Off)
        .level(log::LevelFilter::Warn)
        .chain(fern::Output::call(console_log::log))
        .apply();
    match logger {
        Ok(_) => log::info!("Logging initialized."),
        Err(error) => eprintln!("Error initializing logging: {:?}", error),
    }
}

pub type ProgressBar = dyn Fn(u32);

#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum RepoType {
    /// This is a model, usually it consists of weight files and some configuration
    Model,
    /// This is a dataset, usually contains data within parquet files
    Dataset,
    /// This is a space, usually a demo showcashing a given model or dataset
    Space,
}

#[wasm_bindgen]
pub struct ApiBuilder {
    endpoint: String,
}

#[wasm_bindgen]
impl ApiBuilder {
    /// Build an Api from a HF hub repository.
    #[wasm_bindgen]
    pub fn from_hf(repo_id: &str, ty: RepoType) -> Self {
        Self {
            endpoint: Self::endpoint(repo_id, ty),
        }
    }

    pub fn endpoint(repo_id: &str, ty: RepoType) -> String {
        match ty {
            RepoType::Model => {
                format!("https://huggingface.co/{repo_id}/resolve/main")
            }
            RepoType::Dataset => {
                format!("https://huggingface.co/datasets/{repo_id}/resolve/main")
            }
            RepoType::Space => {
                format!("https://huggingface.co/spaces/{repo_id}/resolve/main")
            }
        }
    }

    /// Build an Api from a HF hub repository at a specific revision.
    #[wasm_bindgen]
    pub fn from_hf_with_revision(repo_id: String, revision: String) -> Self {
        Self {
            endpoint: format!("https://huggingface.co/{repo_id}/resolve/{revision}"),
        }
    }

    /// Build an Api from a custom URL.
    #[wasm_bindgen]
    pub fn from_custom(endpoint: String) -> Self {
        Self { endpoint }
    }

    /// Build the Api.
    #[wasm_bindgen]
    pub fn build(&self) -> Api {
        Api {
            endpoint: self.endpoint.clone(),
        }
    }
}

#[wasm_bindgen]
pub struct Api {
    endpoint: String,
}

#[wasm_bindgen]
impl Api {
    /// Get a file from the repository
    #[wasm_bindgen]
    pub async fn get(&self, file_name: &str) -> Result<Uint8Array, JsError> {
        self.get_internal(file_name).await.map_err(js_to_js_error)
    }

    async fn get_internal(&self, file_name: &str) -> Result<Uint8Array, JsValue> {
        let file_url = format!("{}/{}", self.endpoint, file_name);
        log::debug!("Fetching file: {}", file_url);

        let response = Request::get(&file_url)
            .mode(RequestMode::Cors)
            .send()
            .await
            .unwrap();
        if !response.ok() {
            return Err(
                js_error(format!("Failed to fetch file: {}", response.status()).as_str()).into(),
            );
        }

        let content_len = response
            .headers()
            .get("Content-Length")
            .ok_or(js_error("No content length"))?
            .parse::<u32>()
            .map_err(|p| js_error(format!("Failed to parse content length: {}", p).as_str()))?;

        let reader = response
            .body()
            .ok_or(js_error("No body"))?
            .get_reader()
            .dyn_into::<web_sys::ReadableStreamDefaultReader>()?;

        let mut recv_len = 0;
        let buf = Uint8Array::new_with_length(content_len);
        while let Some(result) = JsFuture::from(reader.read())
            .await?
            .dyn_into::<Object>()
            .ok()
        {
            let done = Reflect::get(&result, &"done".into())?
                .as_bool()
                .unwrap_or(true);
            if done {
                break;
            }

            if let Ok(chunk) = Reflect::get(&result, &"value".into()) {
                let chunk_array: Uint8Array = chunk.dyn_into()?;
                buf.set(&chunk_array, recv_len);
                recv_len += chunk_array.length();
                log::warn!(
                    "{}% downloaded",
                    (recv_len as f64 / content_len as f64) * 100.0
                );
            }
        }
        Ok(buf)
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    fn log_init() {
        console_error_panic_hook::set_once();
        let logger = fern::Dispatch::new()
            .format(|out, message, record| {
                out.finish(format_args!(
                    "{}[{}][{}] {}",
                    chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"),
                    record.target(),
                    record.level(),
                    message
                ))
            })
            .level_for("tokenizers", log::LevelFilter::Off)
            .level(log::LevelFilter::Info)
            .chain(fern::Output::call(console_log::log))
            .apply();
        match logger {
            Ok(_) => log::info!("Logging initialized."),
            Err(error) => eprintln!("Error initializing logging: {:?}", error),
        }
    }

    #[wasm_bindgen_test]
    async fn pull_from_hf() -> Result<(), JsValue> {
        log_init();
        let model_repo = ApiBuilder::from_hf("jantxu/ratchet-test", RepoType::Model).build();
        let model_bytes = model_repo.get("model.safetensors").await?;
        let length = model_bytes.length();
        assert!(length == 8388776, "Length was {length}");
        Ok(())
    }
}
