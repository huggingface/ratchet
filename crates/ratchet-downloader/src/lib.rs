use js_sys::Uint8Array;
#[cfg(test)]
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

use futures_util::{AsyncReadExt, StreamExt};
use gloo::console::{debug, error as log_error};
use js_sys::JsString;
use wasm_bindgen::{prelude::*, JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use wasm_streams::ReadableStream;
use web_sys::{console, ReadableStreamGetReaderOptions, ReadableStreamReaderMode};
use winnow::{binary::bits::bytes, prelude::*, stream::Stream, Bytes, Partial};
use winnow::{binary::u32, binary::u64, combinator::preceded, Parser};

mod fetch;
pub mod huggingface;

#[cfg(test)]
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen]
pub fn js_error(message: String) -> JsError {
    JsError::new(message.as_str())
}

type BytesStream<'i> = Partial<&'i Bytes>;

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

    async fn open_stream(&self, file_name: String) -> Result<(), JsError> {
        let file_url = format!("{}/{}", self.url, file_name);
        let response = fetch::fetch(file_url.as_str()).await?;

        let raw_body = response
            .body()
            .ok_or(js_error(format!("Failed to load {}", file_name)))?;

        let mut body = ReadableStream::from_raw(raw_body);
        let reader = body.get_byob_reader();
        let mut async_read = reader.into_async_read();

        let mut buf = [0u8; 100];
        let result = async_read.read_exact(&mut buf).await?;

        let mut test = BytesStream::new(Bytes::new(&buf));

        let g1 = &test.next_token();
        let g2 = &test.next_token();
        let u = &test.next_token();
        let f = &test.next_token();
        debug!("Done!:", format!("{:?}{:?}{:?}{:?}", g1, g2, u, f));

        Ok(())
    }
}

mod gguf {
    use crate::BytesStream;
    use winnow::binary::u32;
    use winnow::binary::u64;
    use winnow::binary::Endianness;
    use winnow::Parser;

    #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Header {
        pub version: u32,
        pub tensor_count: u64,
        pub metadata_kv_count: u64,
    }

    #[inline]
    fn magic_number(input: &mut BytesStream) -> winnow::PResult<()> {
        // [TODO] Fix endianness
        (71, 71, 85, 70).parse_next(input).map(|_magic_number| ())
    }

    #[inline]
    fn version(input: &mut BytesStream) -> winnow::PResult<u32> {
        u32(Endianness::Little).parse_next(input)
    }

    #[inline]
    fn tensor_count(input: &mut BytesStream) -> winnow::PResult<u64> {
        u64(Endianness::Little).parse_next(input)
    }

    #[inline]
    fn metadata_kv_count(input: &mut BytesStream) -> winnow::PResult<u64> {
        u64(Endianness::Little).parse_next(input)
    }

    #[inline]
    fn metadata_kv(input: &mut BytesStream) -> winnow::PResult<u64> {
        u64(Endianness::Little).parse_next(input)
    }

    pub fn parse_header(input: &mut BytesStream) -> winnow::PResult<Header> {
        (magic_number, version, tensor_count, metadata_kv_count)
            .parse_next(input)
            .map(|(gguf, version, tensor_count, metadata_kv_count)| Header {
                version,
                tensor_count,
                metadata_kv_count,
            })
    }
}

pub fn to_std_error(error: winnow::error::ErrMode<winnow::error::ContextError>) -> std::io::Error {
    match error {
        winnow::error::ErrMode::Backtrack(err) => {
            std::io::Error::new(std::io::ErrorKind::Other, "Backtrack")
        }
        winnow::error::ErrMode::Cut(err) => std::io::Error::new(std::io::ErrorKind::Other, "Cut"),
        winnow::error::ErrMode::Incomplete(needed) => {
            std::io::Error::new(std::io::ErrorKind::Other, "Needed")
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use anyhow::Error;
    use winnow::Bytes;

    use crate::{gguf, to_std_error, BytesStream};

    #[test]
    fn test_parse_header() -> anyhow::Result<()> {
        let mut file = std::fs::File::open("./test-data/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")?;

        let buffer_size = 30;
        let min_buffer_growth = 100;
        let buffer_growth_factor = 2;
        let mut buffer = circular::Buffer::with_capacity(buffer_size);
        let read = file.read(buffer.space())?;
        buffer.fill(read);

        let mut input = BytesStream::new(Bytes::new(buffer.data()));

        let result = gguf::parse_header(&mut input).map_err(to_std_error)?;
        let expected = gguf::Header {
            version: 3,
            tensor_count: 201,
            metadata_kv_count: 23,
        };
        assert_eq!(result, expected);
        Ok(())
    }
}

#[cfg(test)]
#[wasm_bindgen_test]
async fn pass() -> Result<(), JsValue> {
    use js_sys::JsString;

    let model = Model::from_custom("http://localhost:8888".to_string());
    let stream = model
        .open_stream(
            "TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
                .to_string(),
        )
        .await
        .map_err(|err| {
            log_error!(err);
            JsString::from("Failed to download file")
        })?;
    Ok(())
}
