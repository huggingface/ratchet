#![feature(seek_stream_len)]
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

use std::io::Read;

use anyhow::Error;
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
    use std::io::Seek;

    use crate::BytesStream;
    use winnow::binary::{u32, u64, u8, Endianness};

    use anyhow::anyhow;
    use winnow::combinator::fail;
    use winnow::error::Needed;
    use winnow::error::{AddContext, ContextError, ErrMode, StrContext};
    use winnow::prelude;
    use winnow::stream::{Offset, Stream};
    use winnow::token::take;
    use winnow::Parser;

    #[derive(Clone, Debug)]
    pub struct MetadataKv {
        pub key: String,
        pub metadata_value: MetadataValue,
    }
    #[derive(Clone, Debug)]
    pub struct Header {
        pub version: u32,
        pub tensor_count: u64,
        pub metadata_kv: Vec<MetadataKv>,
    }

    #[derive(Clone, Debug)]
    pub struct TensorInfo {
        pub name: String,
        pub dimensions: Vec<u64>,
        pub ggml_type: GgmlType,
        pub offset: u64,
    }

    #[derive(Clone, Debug)]
    pub enum MetadataValueType {
        GgufMetadataValueTypeUint8,
        GgufMetadataValueTypeInt8,
        GgufMetadataValueTypeUint16,
        GgufMetadataValueTypeInt16,
        GgufMetadataValueTypeUint32,
        GgufMetadataValueTypeInt32,
        GgufMetadataValueTypeFloat32,
        GgufMetadataValueTypeBool,
        GgufMetadataValueTypeString,
        GgufMetadataValueTypeArray,
        GgufMetadataValueTypeUint64,
        GgufMetadataValueTypeInt64,
        GgufMetadataValueTypeFloat64,
    }

    #[derive(Clone, Debug)]
    pub enum MetadataValue {
        GgufMetadataValueUint8(u8),
        GgufMetadataValueInt8(i8),
        GgufMetadataValueUint16(u16),
        GgufMetadataValueInt16(i16),
        GgufMetadataValueUint32(u32),
        GgufMetadataValueInt32(i32),
        GgufMetadataValueFloat32(f32),
        GgufMetadataValueBool(bool),
        GgufMetadataValueString(String),
        GgufMetadataValueArray(Vec<MetadataValue>),
        GgufMetadataValueUint64(u64),
        GgufMetadataValueInt64(i64),
        GgufMetadataValueFloat64(f64),
    }

    #[derive(Clone, Debug)]
    pub enum GgmlType {
        GgmlTypeF32,
        GgmlTypeF16,
        GgmlTypeQ4_0,
        GgmlTypeQ4_1,
        GgmlTypeQ5_0,
        GgmlTypeQ5_1,
        GgmlTypeQ8_0,
        GgmlTypeQ8_1,
        // k-quantizations
        GgmlTypeQ2K,
        GgmlTypeQ3K,
        GgmlTypeQ4K,
        GgmlTypeQ5K,
        GgmlTypeQ6K,
        GgmlTypeQ8K,
        GgmlTypeI8,
        GgmlTypeI16,
        GgmlTypeI32,
        GgmlTypeCount,
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

    fn parse_metadata_value_array(input: &mut BytesStream) -> winnow::PResult<MetadataValue> {
        (parse_metadata_value_type, u64(Endianness::Little))
            .flat_map(|(metadata_value_type, length)| {
                winnow::combinator::repeat(
                    length as usize,
                    parse_metadata_value(metadata_value_type),
                )
            })
            .parse_next(input)
            .map(MetadataValue::GgufMetadataValueArray)
    }

    fn parse_metadata_value_type(input: &mut BytesStream) -> winnow::PResult<MetadataValueType> {
        u32(Endianness::Little)
            .parse_next(input)
            .and_then(|metadata_value_type| match metadata_value_type {
                0 => Ok(MetadataValueType::GgufMetadataValueTypeUint8),
                1 => Ok(MetadataValueType::GgufMetadataValueTypeInt8),
                2 => Ok(MetadataValueType::GgufMetadataValueTypeUint16),
                3 => Ok(MetadataValueType::GgufMetadataValueTypeInt16),
                4 => Ok(MetadataValueType::GgufMetadataValueTypeUint32),
                5 => Ok(MetadataValueType::GgufMetadataValueTypeInt32),
                6 => Ok(MetadataValueType::GgufMetadataValueTypeFloat32),
                7 => Ok(MetadataValueType::GgufMetadataValueTypeBool),
                8 => Ok(MetadataValueType::GgufMetadataValueTypeString),
                9 => Ok(MetadataValueType::GgufMetadataValueTypeArray),
                10 => Ok(MetadataValueType::GgufMetadataValueTypeUint64),
                11 => Ok(MetadataValueType::GgufMetadataValueTypeInt64),
                12 => Ok(MetadataValueType::GgufMetadataValueTypeFloat64),
                other => Err(cut_error(input, "Unknown metadata value type.")),
            })
    }

    #[inline]
    fn parse_metadata_value<'i>(
        metadata_value_type: MetadataValueType,
    ) -> impl Parser<BytesStream<'i>, MetadataValue, ContextError> {
        move |input: &mut BytesStream| match metadata_value_type {
            MetadataValueType::GgufMetadataValueTypeUint8 => winnow::binary::u8
                .map(MetadataValue::GgufMetadataValueUint8)
                .parse_next(input),

            MetadataValueType::GgufMetadataValueTypeInt8 => winnow::binary::i8
                .map(MetadataValue::GgufMetadataValueInt8)
                .parse_next(input),
            MetadataValueType::GgufMetadataValueTypeUint16 => {
                winnow::binary::u16(Endianness::Little)
                    .map(MetadataValue::GgufMetadataValueUint16)
                    .parse_next(input)
            }
            MetadataValueType::GgufMetadataValueTypeInt16 => {
                winnow::binary::i16(Endianness::Little)
                    .map(MetadataValue::GgufMetadataValueInt16)
                    .parse_next(input)
            }
            MetadataValueType::GgufMetadataValueTypeUint32 => {
                winnow::binary::u32(Endianness::Little)
                    .map(MetadataValue::GgufMetadataValueUint32)
                    .parse_next(input)
            }
            MetadataValueType::GgufMetadataValueTypeInt32 => {
                winnow::binary::i32(Endianness::Little)
                    .map(MetadataValue::GgufMetadataValueInt32)
                    .parse_next(input)
            }
            MetadataValueType::GgufMetadataValueTypeFloat32 => {
                winnow::binary::f32(Endianness::Little)
                    .map(MetadataValue::GgufMetadataValueFloat32)
                    .parse_next(input)
            }
            MetadataValueType::GgufMetadataValueTypeBool => winnow::binary::i8
                .map(|b| {
                    if b == 0 {
                        MetadataValue::GgufMetadataValueBool(true)
                    } else {
                        MetadataValue::GgufMetadataValueBool(false)
                    }
                })
                .parse_next(input),
            MetadataValueType::GgufMetadataValueTypeString => parse_string
                .map(MetadataValue::GgufMetadataValueString)
                .parse_next(input),
            MetadataValueType::GgufMetadataValueTypeArray => {
                parse_metadata_value_array.parse_next(input)
            }
            MetadataValueType::GgufMetadataValueTypeUint64 => {
                winnow::binary::u64(Endianness::Little)
                    .map(MetadataValue::GgufMetadataValueUint64)
                    .parse_next(input)
            }
            MetadataValueType::GgufMetadataValueTypeInt64 => {
                winnow::binary::i64(Endianness::Little)
                    .map(MetadataValue::GgufMetadataValueInt64)
                    .parse_next(input)
            }
            MetadataValueType::GgufMetadataValueTypeFloat64 => {
                winnow::binary::f64(Endianness::Little)
                    .map(MetadataValue::GgufMetadataValueFloat64)
                    .parse_next(input)
            }
        }
    }

    fn parse_metadata_value_single(input: &mut BytesStream) -> winnow::PResult<MetadataValue> {
        parse_metadata_value_type
            .flat_map(|metadata_value_type| parse_metadata_value(metadata_value_type))
            .parse_next(input)
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

    fn parse_ggml_type(input: &mut BytesStream) -> winnow::PResult<GgmlType> {
        u32(Endianness::Little)
            .parse_next(input)
            .and_then(|metadata_value_type| match metadata_value_type {
                0 => Ok(GgmlType::GgmlTypeF32),
                1 => Ok(GgmlType::GgmlTypeF16),
                2 => Ok(GgmlType::GgmlTypeQ4_0),
                3 => Ok(GgmlType::GgmlTypeQ4_1),
                // 4 & 5 have been removed
                6 => Ok(GgmlType::GgmlTypeQ5_0),
                7 => Ok(GgmlType::GgmlTypeQ5_1),
                8 => Ok(GgmlType::GgmlTypeQ8_0),
                9 => Ok(GgmlType::GgmlTypeQ8_1),
                // k-quantizations
                10 => Ok(GgmlType::GgmlTypeQ2K),
                11 => Ok(GgmlType::GgmlTypeQ3K),
                12 => Ok(GgmlType::GgmlTypeQ4K),
                13 => Ok(GgmlType::GgmlTypeQ5K),
                14 => Ok(GgmlType::GgmlTypeQ6K),
                15 => Ok(GgmlType::GgmlTypeQ8K),
                16 => Ok(GgmlType::GgmlTypeI8),
                17 => Ok(GgmlType::GgmlTypeI16),
                18 => Ok(GgmlType::GgmlTypeI32),
                19 => Ok(GgmlType::GgmlTypeCount),
                other => Err(cut_error(input, "Unknown metadata value type.")),
            })
    }

    fn parse_tensor_info(input: &mut BytesStream) -> winnow::PResult<TensorInfo> {
        (parse_string, u32(Endianness::Little))
            .flat_map(|(name, n_dimensions)| {
                let dimensions_parser =
                    winnow::combinator::repeat(n_dimensions as usize, u64(Endianness::Little));

                (dimensions_parser, parse_ggml_type, u64(Endianness::Little)).map(
                    move |(dimensions, ggml_type, offset)| TensorInfo {
                        name: name.clone(),
                        dimensions,
                        ggml_type,
                        offset,
                    },
                )
            })
            .parse_next(input)
    }

    fn cut_error(
        input: &mut winnow::Partial<&winnow::Bytes>,
        error_msg: &'static str,
    ) -> ErrMode<ContextError> {
        println!("Error: {}", error_msg);
        ErrMode::Cut(ContextError::new().add_context(input, StrContext::Label(error_msg)))
    }

    #[inline]
    fn parse_metadata_kv<'i>(
        metadata_kv_count: u64,
    ) -> impl Parser<BytesStream<'i>, MetadataKv, ContextError> {
        move |input: &mut BytesStream| {
            (parse_string, parse_metadata_value_single)
                .parse_next(input)
                .map(|(key, metadata_value)| MetadataKv {
                    key,
                    metadata_value,
                })
        }
    }

    pub fn parse_header(input: &mut BytesStream) -> winnow::PResult<Header> {
        (
            parse_magic_number,
            parse_version,
            parse_tensor_count,
            parse_metadata_kv_count,
        )
            .flat_map(|(_gguf, version, tensor_count, metadata_kv_count)| {
                winnow::combinator::repeat(
                    metadata_kv_count as usize,
                    parse_metadata_kv(metadata_kv_count),
                )
                .map(move |metadata_kv| Header {
                    version,
                    tensor_count,
                    metadata_kv,
                })
            })
            .parse_next(input)
    }

    pub fn load_gguf(mut file: std::fs::File) -> anyhow::Result<(Header, Vec<TensorInfo>)> {
        let buffer_size = 1_000_000;
        let buffer_growth_factor = 2;
        let mut buffer = circular::Buffer::with_capacity(buffer_size);

        let header = parse_with_buffer(&mut file, &mut buffer, parse_header, buffer_growth_factor)?;
        let mut tensor_infos: Vec<TensorInfo> = vec![];
        for i in 0..header.tensor_count {
            let tensor_info = parse_with_buffer(
                &mut file,
                &mut buffer,
                parse_tensor_info,
                buffer_growth_factor,
            )?;
            tensor_infos.push(tensor_info);
        }
        Ok((header, tensor_infos))
    }

    fn parse_with_buffer<O>(
        file: &mut std::fs::File,
        buffer: &mut circular::Buffer,
        mut parser: fn(&mut winnow::Partial<&winnow::Bytes>) -> Result<O, ErrMode<ContextError>>,
        buffer_growth_factor: usize,
    ) -> anyhow::Result<O> {
        use std::io::Read;
        let mut result: anyhow::Result<O> = Err(anyhow!("Failed to read file.",));
        'outer: loop {
            if buffer.available_space() == 0 {
                buffer.grow(buffer_growth_factor * buffer.capacity());
            }
            let read = file.read(buffer.space())?;

            if read == 0 {
                println!("Read 0");
                // Should be EOF since we always make sure there is `available_space`
                assert_ne!(buffer.available_space(), 0);
                assert_eq!(
                    buffer.available_data(),
                    0,
                    "leftover data: {}",
                    String::from_utf8_lossy(buffer.data())
                );
                break 'outer;
            }
            buffer.fill(read);

            'inner: loop {
                let mut input = BytesStream::new(winnow::Bytes::new(buffer.data()));

                let parser_result = parser.parse_peek(input);
                match parser_result {
                    Ok((remainder, parser_output)) => {
                        // Tell the buffer how much we read
                        let consumed = remainder.offset_from(&input);
                        buffer.consume(consumed);
                        result = Ok(parser_output);
                        break 'outer;
                    }
                    Err(ErrMode::Backtrack(e)) => {
                        let pos = buffer.position();
                        return Err(anyhow::format_err!(e.to_string()));
                    }
                    Err(ErrMode::Cut(e)) => {
                        return Err(anyhow::format_err!(e.to_string()));
                    }
                    Err(ErrMode::Incomplete(_)) => {
                        let new_capacity = buffer_growth_factor * buffer.capacity();
                        buffer.grow(new_capacity);
                        break 'inner;
                    }
                }
            }
        }
        result
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

    use crate::{gguf, BytesStream};

    #[test]
    fn test_load_gguf() -> anyhow::Result<()> {
        let mut file = std::fs::File::open("./test-data/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")?;

        let result = gguf::load_gguf(file);

        match result {
            Ok((header, tensor_info)) => {
                println!("{:#?}", tensor_info);
                assert_eq!(header.version, 3);
                assert_eq!(header.tensor_count, 201)
            }
            Err(err) => println!("Got an error: {:#?}", err),
        }

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
