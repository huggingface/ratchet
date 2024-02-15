use std::io::BufReader;

use futures_util::StreamExt;
use gloo::console::debug;
use js_sys::Uint8Array;
use util::{js_error, js_to_js_error, to_future};
use wasm_bindgen::{prelude::*, JsCast, JsValue};
use wasm_streams::ReadableStream;
use web_sys::{Cache, Request, RequestInit, RequestMode, Response};
pub mod error;
mod gguf;
mod util;

#[cfg(test)]
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

#[cfg(test)]
wasm_bindgen_test_configure!(run_in_browser);

pub type ProgressBar = dyn Fn(u32);

#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum RepoType {
    /// This is a model, usually it consists of weight files and some configuration
    /// files
    Model,
    /// This is a dataset, usually contains data within parquet files
    Dataset,
    /// This is a space, usually a demo showcashing a given model or dataset
    Space,
}

#[wasm_bindgen]
pub struct ApiBuilder {
    endpoint: String,
    cached: bool,
}

#[wasm_bindgen]
impl ApiBuilder {
    /// Build an Api from a HF hub repository.
    #[wasm_bindgen]
    pub fn from_hf(repo_id: &str, ty: RepoType) -> Self {
        Self {
            cached: true,
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
            cached: true,
            endpoint: format!("https://huggingface.co/{repo_id}/resolve/{revision}"),
        }
    }

    /// Build an Api from a custom URL.
    #[wasm_bindgen]
    pub fn from_custom(endpoint: &str) -> Self {
        Self {
            cached: true,
            endpoint: endpoint.to_string(),
        }
    }

    /// Disable caching
    #[wasm_bindgen]
    pub fn uncached(mut self) -> Self {
        self.cached = false;
        self
    }

    /// Build the Api.
    #[wasm_bindgen]
    pub fn build(&self) -> Api {
        Api {
            endpoint: self.endpoint.clone(),
            cached: self.cached,
        }
    }
}

#[wasm_bindgen]
pub struct Api {
    endpoint: String,
    cached: bool,
}

#[wasm_bindgen]
impl Api {
    /// Get a file from the repository
    #[wasm_bindgen]
    pub async fn get(&self, file_name: &str) -> Result<ApiResponse, JsError> {
        self.get_internal(file_name).await.map_err(js_to_js_error)
    }

    async fn get_internal(&self, file_name: &str) -> Result<ApiResponse, JsValue> {
        let file_url = format!("{}/{}", self.endpoint, file_name);

        let caches = web_sys::window()
            .ok_or(js_error("Couldn't get window handle"))?
            .caches()?;
        let cache: Cache = to_future(caches.open("ratchet-cache")).await?;

        let mut opts = RequestInit::new();
        opts.method("GET");
        opts.mode(RequestMode::Cors);

        let request = Request::new_with_str_and_init(&file_url, &opts)?;

        let promise = cache.match_with_request(&request);
        let cache_hit: JsValue = to_future(promise).await?;

        let (raw, cached) = if cache_hit.is_undefined() || !self.cached {
            let raw_response = util::fetch(file_url.as_str()).await?;

            let status = raw_response.status();

            // [TODO] Need to improve the whole caching here
            if (status >= 200 && status < 400) {
                let _ = to_future::<JsValue>(
                    cache.put_with_str(file_url.as_str(), &raw_response.clone()?),
                )
                .await;
            }
            (raw_response, false)
        } else {
            let raw_response: Response = cache_hit.dyn_into()?;
            (raw_response, true)
        };

        Ok(ApiResponse { raw, cached })
    }
}

#[wasm_bindgen]
pub struct ApiResponse {
    raw: Response,
    cached: bool,
}

#[wasm_bindgen]
impl ApiResponse {
    /// Get the response as bytes
    #[wasm_bindgen]
    pub async fn to_uint8(&self) -> Result<Uint8Array, JsError> {
        let promise = self.raw.array_buffer().map_err(js_to_js_error)?;

        let buf_js = util::to_future::<wasm_bindgen::JsValue>(promise)
            .await
            .map_err(js_to_js_error)?;

        let buffer = Uint8Array::new(&buf_js);
        Ok(buffer)
    }

    #[wasm_bindgen]
    pub async fn gguf(&self) -> Result<(), JsError> {
        let data = self.to_uint8().await?;
        let mut reader = std::io::BufReader::new(std::io::Cursor::new(data.to_vec()));

        let result = gguf::gguf::Content::read(&mut reader)?;

        debug!("result", format!("{:?}", result.metadata));
        return Ok(());
    }

    #[wasm_bindgen]
    pub fn is_cached(&self) -> bool {
        self.cached
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[wasm_bindgen_test]
    async fn test_loading_from_hub() -> Result<(), JsValue> {
        let model_repo = ApiBuilder::from_hf("jantxu/ratchet-test", RepoType::Model).build();
        let model = model_repo.get("model.safetensors").await?;
        let bytes = model.to_uint8().await?;
        let length = bytes.length();
        assert!(length == 8388776, "Length was {length}");
        Ok(())
    }

    #[wasm_bindgen_test]
    async fn test_gguf() -> Result<(), JsValue> {
        const DUMMY_GGUF: &[u8] =
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/test-data/dummy.gguf"));

        debug!(format!("{:?}", DUMMY_GGUF));
        let model_repo = ApiBuilder::from_hf("jantxu/ratchet-test", RepoType::Model)
            .uncached()
            .build();
        let file = model_repo.get("dummy.gguf").await?;
        let model_data = file.gguf().await?;

        Ok(())
    }
}
