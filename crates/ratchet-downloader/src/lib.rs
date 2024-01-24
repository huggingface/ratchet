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
    use winnow::binary::{u32, u64, u8, Endianness};

    use winnow::error::{AddContext, ContextError, ErrMode, StrContext};
    use winnow::token::take;
    use winnow::Parser;

    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub struct MetadataKv {
        pub key: String,
        pub value_type: MetadataValueType,
    }
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Header {
        pub version: u32,
        pub tensor_count: u64,
        pub metadata_kv_count: u64,
        pub metadata_kv: MetadataKv,
    }

    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub enum MetadataValueType {
        // The value is a 8-bit unsigned integer.
        GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
        // The value is a 8-bit signed integer.
        GGUF_METADATA_VALUE_TYPE_INT8 = 1,
        // The value is a 16-bit unsigned little-endian integer.
        GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
        // The value is a 16-bit signed little-endian integer.
        GGUF_METADATA_VALUE_TYPE_INT16 = 3,
        // The value is a 32-bit unsigned little-endian integer.
        GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
        // The value is a 32-bit signed little-endian integer.
        GGUF_METADATA_VALUE_TYPE_INT32 = 5,
        // The value is a 32-bit IEEE754 floating point number.
        GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
        // The value is a boolean.
        // 1-byte value where 0 is false and 1 is true.
        // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
        GGUF_METADATA_VALUE_TYPE_BOOL = 7,
        // The value is a UTF-8 non-null-terminated string, with length prepended.
        GGUF_METADATA_VALUE_TYPE_STRING = 8,
        // The value is an array of other values, with the length and type prepended.
        ///
        // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
        GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
        // The value is a 64-bit unsigned little-endian integer.
        GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
        // The value is a 64-bit signed little-endian integer.
        GGUF_METADATA_VALUE_TYPE_INT64 = 11,
        // The value is a 64-bit IEEE754 floating point number.
        GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
    }

    #[inline]
    fn parse_magic_number(input: &mut BytesStream) -> winnow::PResult<()> {
        // [TODO] Fix endianness
        (71, 71, 85, 70).parse_next(input).map(|_magic_number| ())
    }

    #[inline]
    fn parse_version(input: &mut BytesStream) -> winnow::PResult<u32> {
        u32(Endianness::Little).parse_next(input)
    }

    #[inline]
    fn parse_tensor_count(input: &mut BytesStream) -> winnow::PResult<u64> {
        u64(Endianness::Little).parse_next(input)
    }

    #[inline]
    fn parse_metadata_value_type(input: &mut BytesStream) -> winnow::PResult<MetadataValueType> {
        u32(Endianness::Little)
            .parse_next(input)
            .and_then(|value| match value {
                0 => Ok(MetadataValueType::GGUF_METADATA_VALUE_TYPE_UINT8),
                1 => Ok(MetadataValueType::GGUF_METADATA_VALUE_TYPE_INT8),
                2 => Ok(MetadataValueType::GGUF_METADATA_VALUE_TYPE_UINT16),
                3 => Ok(MetadataValueType::GGUF_METADATA_VALUE_TYPE_INT16),
                4 => Ok(MetadataValueType::GGUF_METADATA_VALUE_TYPE_UINT32),
                5 => Ok(MetadataValueType::GGUF_METADATA_VALUE_TYPE_INT32),
                6 => Ok(MetadataValueType::GGUF_METADATA_VALUE_TYPE_FLOAT32),
                7 => Ok(MetadataValueType::GGUF_METADATA_VALUE_TYPE_BOOL),
                8 => Ok(MetadataValueType::GGUF_METADATA_VALUE_TYPE_STRING),
                9 => Ok(MetadataValueType::GGUF_METADATA_VALUE_TYPE_ARRAY),
                10 => Ok(MetadataValueType::GGUF_METADATA_VALUE_TYPE_UINT64),
                11 => Ok(MetadataValueType::GGUF_METADATA_VALUE_TYPE_INT64),
                12 => Ok(MetadataValueType::GGUF_METADATA_VALUE_TYPE_FLOAT64),
                other => Err(cut_error(input, &"Found invalid metadata type value.")),
            })
    }

    #[inline]
    fn parse_metadata_kv_count(input: &mut BytesStream) -> winnow::PResult<u64> {
        u64(Endianness::Little).parse_next(input)
    }

    fn parse_string(input: &mut BytesStream) -> winnow::PResult<String> {
        u64(Endianness::Little)
            .flat_map(|count| take(count))
            .parse_next(input)
            .and_then(|bytes| {
                String::from_utf8(bytes.to_vec()).map_err(|err| {
                    let error_msg = "Failed to parse string";
                    cut_error(input, error_msg)
                })
            })
    }

    fn cut_error(
        input: &mut winnow::Partial<&winnow::Bytes>,
        error_msg: &'static str,
    ) -> ErrMode<ContextError> {
        ErrMode::Cut(ContextError::new().add_context(input, StrContext::Label(error_msg)))
    }

    #[inline]
    fn parse_metadata_kv<'i>(
        metadata_kv_count: u64,
    ) -> impl Parser<BytesStream<'i>, MetadataKv, ContextError> {
        move |input: &mut BytesStream| {
            (parse_string, parse_metadata_value_type)
                .parse_next(input)
                .map(|(key, value_type)| MetadataKv { key, value_type })
        }
    }

    pub fn parse_header(input: &mut BytesStream) -> winnow::PResult<Header> {
        (
            parse_magic_number,
            parse_version,
            parse_tensor_count,
            parse_metadata_kv_count,
        )
            .flat_map(|(gguf, version, tensor_count, metadata_kv_count)| {
                parse_metadata_kv(metadata_kv_count).map(move |metadata_kv| {
                    (gguf, version, tensor_count, metadata_kv_count, metadata_kv)
                })
            })
            .parse_next(input)
            .map(
                |(gguf, version, tensor_count, metadata_kv_count, metadata_kv)| Header {
                    version,
                    tensor_count,
                    metadata_kv_count,
                    metadata_kv,
                },
            )
    }
    pub fn to_std_error(
        error: winnow::error::ErrMode<winnow::error::ContextError>,
    ) -> std::io::Error {
        match error {
            ErrMode::Backtrack(err) => std::io::Error::new(std::io::ErrorKind::Other, "Backtrack"),
            ErrMode::Cut(err) => std::io::Error::new(std::io::ErrorKind::Other, "Cut"),
            ErrMode::Incomplete(needed) => std::io::Error::new(std::io::ErrorKind::Other, "Needed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use anyhow::Error;
    use winnow::Bytes;

    use crate::{gguf, BytesStream};

    #[test]
    fn test_parse_header() -> anyhow::Result<()> {
        let mut file = std::fs::File::open("./test-data/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")?;

        let buffer_size = 100;
        let min_buffer_growth = 100;
        let buffer_growth_factor = 2;
        let mut buffer = circular::Buffer::with_capacity(buffer_size);
        let read = file.read(buffer.space())?;
        buffer.fill(read);

        let mut input = BytesStream::new(Bytes::new(buffer.data()));

        let result = gguf::parse_header(&mut input).map_err(gguf::to_std_error)?;

        println!("{:#?}", result.metadata_kv);
        println!("{:#?}", result.metadata_kv.value_type);
        assert_eq!(result.version, 3);
        assert_eq!(result.tensor_count, 201);
        assert_eq!(result.metadata_kv_count, 23);
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
