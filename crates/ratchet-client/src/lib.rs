use js_sys::Uint8Array;
use util::{js_error, js_to_js_error, to_future};
#[cfg(test)]
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

use futures_util::{AsyncRead, AsyncReadExt, StreamExt, TryFutureExt};
use gloo::console::{debug, error as log_error};
use js_sys::JsString;
use wasm_bindgen::{prelude::*, JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use wasm_streams::readable::{IntoAsyncRead, ReadableStreamBYOBReader};
use wasm_streams::ReadableStream;
use web_sys::{
    console, Cache, CacheStorage, ReadableStreamGetReaderOptions, ReadableStreamReaderMode,
    Request, RequestInit, RequestMode, Response, Window,
};
pub mod error;
mod gguf;
mod util;
use bytes::Bytes;
use tokio_util::compat::FuturesAsyncReadCompatExt;

#[cfg(test)]
wasm_bindgen_test_configure!(run_in_browser);

pub type ProgressBar = dyn Fn(u32) -> ();

#[wasm_bindgen]
pub struct ApiBuilder {
    endpoint: String,
    cached: bool,
}

#[wasm_bindgen]
impl ApiBuilder {
    /// Build an Api from a HF hub repository.
    #[wasm_bindgen]
    pub fn from_hf(repo_id: &str) -> Self {
        Self {
            cached: true,
            endpoint: format!("https://huggingface.co/{repo_id}/resolve/main"),
        }
    }

    /// Build an Api from a HF hub repository at a specific revision.
    #[wasm_bindgen]
    pub fn from_hf_with_revision(repo_id: &str, revision: &str) -> Self {
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
            let _ =
                to_future::<JsValue>(cache.put_with_str(file_url.as_str(), &raw_response.clone()?))
                    .await;
            (raw_response, false)
        } else {
            let raw_response: Response = cache_hit.dyn_into()?;
            (raw_response, true)
        };

        return Ok(ApiResponse { raw, cached });
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
    pub fn is_cached(&self) -> bool {
        self.cached
    }

    #[wasm_bindgen]
    pub async fn gguf(&self) -> Result<(), JsError> {
        let raw_body = self.raw.body().ok_or(js_error("Failed to open body"))?;

        let mut body: ReadableStream = ReadableStream::from_raw(raw_body);
        let reader: ReadableStreamBYOBReader<'_> = body.get_byob_reader();

        let mut async_read = reader.into_async_read().compat();

        let magic = gguf::VersionedMagic::read(&mut async_read).await?;
        debug!("Got gguf: ", format!("{:?}", magic));

        return Ok(());
    }
}

trait ToBytes {
    fn to_bytes(self) -> Bytes;
}

impl ToBytes for Uint8Array {
    fn to_bytes(self) -> Bytes {
        let mut bytes = vec![0; self.length() as usize];
        self.copy_to(&mut bytes);
        bytes.into()
    }
}
#[cfg(test)]
#[wasm_bindgen_test]
async fn pass() -> Result<(), JsValue> {
    use js_sys::JsString;

    let model_repo = ApiBuilder::from_custom("http://localhost:8888/test-data/").build();

    let model = model_repo.get("dummy.gguf").await?;

    let gguf = model.gguf().await?;
    Ok(())
}
