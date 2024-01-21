use js_sys::Uint8Array;
use util::{js_error, to_error, to_future};
#[cfg(test)]
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

use futures_util::{AsyncRead, AsyncReadExt, StreamExt, TryFutureExt};
use gloo::console::{debug, error as log_error};
use js_sys::JsString;
use wasm_bindgen::{prelude::*, JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use wasm_streams::readable::ReadableStreamBYOBReader;
use wasm_streams::ReadableStream;
use web_sys::{console, ReadableStreamGetReaderOptions, ReadableStreamReaderMode, Response};
mod util;
use bytes::Bytes;

#[cfg(test)]
wasm_bindgen_test_configure!(run_in_browser);

pub type ProgressBar = dyn Fn(u32) -> ();

pub struct ApiBuilder {
    endpoint: String,
    cached: bool,
}

impl ApiBuilder {
    fn from_hf(repo_id: &str) -> Self {
        Self {
            cached: true,
            endpoint: format!("https://huggingface.co/{repo_id}/resolve/main"),
        }
    }

    fn from_hf_with_revision(repo_id: String, revision: String) -> Self {
        Self {
            cached: true,
            endpoint: format!("https://huggingface.co/{repo_id}/resolve/{revision}"),
        }
    }

    fn from_custom(endpoint: String) -> Self {
        Self {
            cached: true,
            endpoint,
        }
    }

    fn uncached(mut self) -> Self {
        self.cached = false;
        self
    }

    fn build(&self) -> Api {
        Api {
            endpoint: self.endpoint.clone(),
            cached: self.cached,
        }
    }
}

pub struct Api {
    endpoint: String,
    cached: bool,
}

impl Api {
    pub async fn get(&self, file_name: &str) -> Result<ApiResponse, JsError> {
        let file_url = format!("{}/{}", self.endpoint, file_name);
        let raw = util::fetch(file_url.as_str()).await?;

        // let raw_body = response
        //     .body()
        //     .ok_or(js_error(format!("Failed to load {}", file_name)))?;

        // let mut body: ReadableStream = ReadableStream::from_raw(raw_body);
        // let reader: ReadableStreamBYOBReader<'_> = body.get_byob_reader();
        // let mut async_read = reader.into_async_read();

        return Ok(ApiResponse { raw });
    }
}

pub struct ApiResponse {
    raw: Response,
}

impl ApiResponse {
    /// Get the response as bytes
    pub async fn bytes(self) -> Result<Bytes, JsError> {
        let promise = self.raw.array_buffer().map_err(to_error)?;

        let buf_js = util::to_future::<wasm_bindgen::JsValue>(promise).await?;

        let buffer = Uint8Array::new(&buf_js);
        let mut bytes = vec![0; buffer.length() as usize];
        buffer.copy_to(&mut bytes);
        Ok(bytes.into())
    }

    // fn to_async_read(self) -> Result<impl AsyncRead, JsError> {
    //     let raw_body = self
    //         .raw
    //         .body()
    //         .ok_or(js_error("Failed to load response body"))?;
    //     let mut body: ReadableStream = ReadableStream::from_raw(raw_body);
    //     let reader: ReadableStreamBYOBReader<'_> = body.get_byob_reader();
    //     Ok(reader.into_async_read())
    // }
}

#[cfg(test)]
#[wasm_bindgen_test]
async fn pass() -> Result<(), JsValue> {
    use js_sys::JsString;

    let model_repo = ApiBuilder::from_hf("jantxu/ratchet-test").build();

    let model = model_repo.get("model.safetensors").await?;

    let bytes = model.bytes().await?;

    let length = bytes.len();
    assert!(length == 8388776, "Length was {length}");
    // let model_repo = ApiBuilder::from_hf("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    //     .cached()
    //     .build();

    // let model = model_repo.get("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    // let tokenizer_repo = ApiBuilder::from_hf("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    //     .cached()
    //     .build();
    // let tokenizer = tokenizer_repo.get("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    // let repo = ApiBuilder::from_url("http://localhost:8080").build();
    // let model = repo.get("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    Ok(())
}
