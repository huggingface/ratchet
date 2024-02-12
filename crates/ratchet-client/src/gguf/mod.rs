//! Support for the GGUF file format.
//! Adapted from https://github.com/huggingface/candle/blob/main/candle-core/src/quantized/gguf_file.rs

use self::ggml::GgmlDType;
use crate::error::{self, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use futures_util::future::BoxFuture;
use futures_util::{future, stream, AsyncWrite, StreamExt};
use futures_util::{Future, FutureExt};
use ratchet::shape;
use ratchet::{Device, Shape, Tensor};
use std::collections::HashMap;
use std::pin::Pin;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncSeek, AsyncSeekExt};

pub mod ggml;
// pub mod ggml_file;
pub mod k_quants;
// pub mod utils;
pub const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Magic {
    Gguf,
}

impl TryFrom<u32> for Magic {
    type Error = crate::error::Error;
    fn try_from(value: u32) -> Result<Self> {
        let magic = match value {
            0x46554747 | 0x47475546 => Self::Gguf,
            _ => crate::bail!("unknown magic 0x{value:08x}"),
        };
        Ok(magic)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionedMagic {
    GgufV1,
    GgufV2,
    GgufV3,
}

impl VersionedMagic {
    pub async fn read<R: AsyncRead + Unpin>(reader: &mut R) -> Result<Self> {
        let magic = reader.read_u32_le().await?;
        let magic = Magic::try_from(magic)?;
        let version = reader.read_u32_le().await?;
        let versioned_magic = match (magic, version) {
            (Magic::Gguf, 1) => Self::GgufV1,
            (Magic::Gguf, 2) => Self::GgufV2,
            (Magic::Gguf, 3) => Self::GgufV3,
            _ => crate::bail!("gguf: unsupported magic/version {magic:?}/{version}"),
        };
        Ok(versioned_magic)
    }
}

#[derive(Debug)]
pub struct TensorInfo {
    pub ggml_dtype: GgmlDType,
    pub shape: Shape,
    pub offset: u64,
}

impl TensorInfo {
    pub async fn read<R: AsyncSeek + AsyncRead + Unpin>(
        &self,
        reader: &mut R,
        tensor_data_offset: u64,
        device: &Device,
    ) -> Result<Tensor> {
        // let tensor_elems = self.shape.elem_count();
        let tensor_elems = self.shape.numel();
        let block_size = self.ggml_dtype.block_size();
        if tensor_elems % block_size != 0 {
            crate::bail!(
            "the number of elements {tensor_elems} is not divisible by the block size {block_size}"
        )
        }
        let size_in_bytes = tensor_elems / block_size * self.ggml_dtype.type_size();
        let mut raw_data = vec![0u8; size_in_bytes];
        reader
            .seek(std::io::SeekFrom::Start(tensor_data_offset + self.offset))
            .await?;
        reader.read_exact(&mut raw_data).await?;
        // ggml_file::qtensor_from_ggml(
        //     self.ggml_dtype,
        //     &raw_data,
        //     // self.shape.dims().to_vec(),
        //     self.shape.to_vec(),
        //     device,
        // )
        // [TODO] Implement
        let tensor = ratchet::Tensor::randn::<f32>(shape![1, 2], device.clone());
        Ok(tensor)
    }
}

#[derive(Debug)]
pub struct Content {
    pub magic: VersionedMagic,
    pub metadata: HashMap<String, Value>,
    pub tensor_infos: HashMap<String, TensorInfo>,
    pub tensor_data_offset: u64,
}

async fn read_string<R: AsyncRead + Unpin>(
    reader: &mut R,
    magic: &VersionedMagic,
) -> Result<String> {
    let len = match magic {
        VersionedMagic::GgufV1 => reader.read_u32_le().await? as usize,
        VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => reader.read_u64_le().await? as usize,
    };
    let mut v = vec![0u8; len];
    reader.read_exact(&mut v).await?;
    // GGUF strings are supposed to be non-null terminated but in practice this happens.
    while let Some(0) = v.last() {
        v.pop();
    }
    // GGUF strings are utf8 encoded but there are cases that don't seem to be valid.
    Ok(String::from_utf8_lossy(&v).into_owned())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    // The value is a 8-bit unsigned integer.
    U8,
    // The value is a 8-bit signed integer.
    I8,
    // The value is a 16-bit unsigned little-endian integer.
    U16,
    // The value is a 16-bit signed little-endian integer.
    I16,
    // The value is a 32-bit unsigned little-endian integer.
    U32,
    // The value is a 32-bit signed little-endian integer.
    I32,
    // The value is a 64-bit unsigned little-endian integer.
    U64,
    // The value is a 64-bit signed little-endian integer.
    I64,
    // The value is a 32-bit IEEE754 floating point number.
    F32,
    // The value is a 64-bit IEEE754 floating point number.
    F64,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    Bool,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    String,
    // The value is an array of other values, with the length and type prepended.
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    Array,
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn value_type(&self) -> ValueType {
        match self {
            Self::U8(_) => ValueType::U8,
            Self::I8(_) => ValueType::I8,
            Self::U16(_) => ValueType::U16,
            Self::I16(_) => ValueType::I16,
            Self::U32(_) => ValueType::U32,
            Self::I32(_) => ValueType::I32,
            Self::U64(_) => ValueType::U64,
            Self::I64(_) => ValueType::I64,
            Self::F32(_) => ValueType::F32,
            Self::F64(_) => ValueType::F64,
            Self::Bool(_) => ValueType::Bool,
            Self::String(_) => ValueType::String,
            Self::Array(_) => ValueType::Array,
        }
    }

    pub fn to_u8(&self) -> Result<u8> {
        match self {
            Self::U8(v) => Ok(*v),
            v => crate::bail!("not a u8 {v:?}"),
        }
    }

    pub fn to_i8(&self) -> Result<i8> {
        match self {
            Self::I8(v) => Ok(*v),
            v => crate::bail!("not a i8 {v:?}"),
        }
    }

    pub fn to_u16(&self) -> Result<u16> {
        match self {
            Self::U16(v) => Ok(*v),
            v => crate::bail!("not a u16 {v:?}"),
        }
    }

    pub fn to_i16(&self) -> Result<i16> {
        match self {
            Self::I16(v) => Ok(*v),
            v => crate::bail!("not a i16 {v:?}"),
        }
    }

    pub fn to_u32(&self) -> Result<u32> {
        match self {
            Self::U32(v) => Ok(*v),
            v => crate::bail!("not a u32 {v:?}"),
        }
    }

    pub fn to_i32(&self) -> Result<i32> {
        match self {
            Self::I32(v) => Ok(*v),
            v => crate::bail!("not a i32 {v:?}"),
        }
    }

    pub fn to_u64(&self) -> Result<u64> {
        match self {
            Self::U64(v) => Ok(*v),
            v => crate::bail!("not a u64 {v:?}"),
        }
    }

    pub fn to_i64(&self) -> Result<i64> {
        match self {
            Self::I64(v) => Ok(*v),
            v => crate::bail!("not a i64 {v:?}"),
        }
    }

    pub fn to_f32(&self) -> Result<f32> {
        match self {
            Self::F32(v) => Ok(*v),
            v => crate::bail!("not a f32 {v:?}"),
        }
    }

    pub fn to_f64(&self) -> Result<f64> {
        match self {
            Self::F64(v) => Ok(*v),
            v => crate::bail!("not a f64 {v:?}"),
        }
    }

    pub fn to_bool(&self) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            v => crate::bail!("not a bool {v:?}"),
        }
    }

    pub fn to_vec(&self) -> Result<&Vec<Value>> {
        match self {
            Self::Array(v) => Ok(v),
            v => crate::bail!("not a vec {v:?}"),
        }
    }

    pub fn to_string(&self) -> Result<&String> {
        match self {
            Self::String(v) => Ok(v),
            v => crate::bail!("not a string {v:?}"),
        }
    }

    fn recursive_read_wrapper<'a, R: AsyncRead + Unpin + Send>(
        reader: &'a mut R,
        value_type: ValueType,
        magic: &'a VersionedMagic,
    ) -> BoxFuture<'a, Result<Self>> {
        Box::pin(Value::read(reader, value_type, magic))
    }

    async fn read<R: AsyncRead + Unpin + Send>(
        reader: &mut R,
        value_type: ValueType,
        magic: &VersionedMagic,
    ) -> Result<Self> {
        let v = match value_type {
            ValueType::U8 => Self::U8(reader.read_u8().await?),
            ValueType::I8 => Self::I8(reader.read_i8().await?),
            ValueType::U16 => Self::U16(reader.read_u16_le().await?),
            ValueType::I16 => Self::I16(reader.read_i16_le().await?),
            ValueType::U32 => Self::U32(reader.read_u32_le().await?),
            ValueType::I32 => Self::I32(reader.read_i32_le().await?),
            ValueType::U64 => Self::U64(reader.read_u64_le().await?),
            ValueType::I64 => Self::I64(reader.read_i64_le().await?),
            ValueType::F32 => Self::F32(reader.read_f32_le().await?),
            ValueType::F64 => Self::F64(reader.read_f64_le().await?),
            ValueType::Bool => match reader.read_u8().await? {
                0 => Self::Bool(false),
                1 => Self::Bool(true),
                b => crate::bail!("unexpected bool value {b}"),
            },
            ValueType::String => Self::String(read_string(reader, magic).await?),
            ValueType::Array => {
                let value_type = reader.read_u32_le().await?;
                let value_type = ValueType::from_u32(value_type)?;
                let len = match magic {
                    VersionedMagic::GgufV1 => reader.read_u32_le().await? as usize,
                    VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                        reader.read_u64_le().await? as usize
                    }
                };
                let mut vs = Vec::with_capacity(len);
                for _ in 0..len {
                    vs.push(Value::recursive_read_wrapper(reader, value_type, magic).await?)
                }
                Self::Array(vs)
            }
        };

        Ok(v)
    }

    //     fn write<W: std::io::Write>(&self, w: &mut W) -> Result<()> {
    //         match self {
    //             &Self::U8(v) => w.write_u8(v)?,
    //             &Self::I8(v) => w.write_i8(v)?,
    //             &Self::U16(v) => w.write_u16::<LittleEndian>(v)?,
    //             &Self::I16(v) => w.write_i16::<LittleEndian>(v)?,
    //             &Self::U32(v) => w.write_u32::<LittleEndian>(v)?,
    //             &Self::I32(v) => w.write_i32::<LittleEndian>(v)?,
    //             &Self::U64(v) => w.write_u64::<LittleEndian>(v)?,
    //             &Self::I64(v) => w.write_i64::<LittleEndian>(v)?,
    //             &Self::F32(v) => w.write_f32::<LittleEndian>(v)?,
    //             &Self::F64(v) => w.write_f64::<LittleEndian>(v)?,
    //             &Self::Bool(v) => w.write_u8(u8::from(v))?,
    //             Self::String(v) => write_string(w, v.as_str())?,
    //             Self::Array(v) => {
    //                 // The `Value` type does not enforce that all the values in an Array have the same
    //                 // type.
    //                 let value_type = if v.is_empty() {
    //                     // Doesn't matter, the array is empty.
    //                     ValueType::U32
    //                 } else {
    //                     let value_type: std::collections::HashSet<_> =
    //                         v.iter().map(|elem| elem.value_type()).collect();
    //                     if value_type.len() != 1 {
    //                         crate::bail!("multiple value-types in the same array {value_type:?}")
    //                     }
    //                     value_type.into_iter().next().unwrap()
    //                 };
    //                 w.write_u32::<LittleEndian>(value_type.to_u32())?;
    //                 w.write_u64::<LittleEndian>(v.len() as u64)?;
    //                 for elem in v.iter() {
    //                     elem.write(w)?
    //                 }
    //             }
    //         }
    //         Ok(())
    //     }
}

impl ValueType {
    fn from_u32(v: u32) -> Result<Self> {
        let v = match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            v => crate::bail!("unrecognized value-type {v:#08x}"),
        };
        Ok(v)
    }

    fn to_u32(self) -> u32 {
        match self {
            Self::U8 => 0,
            Self::I8 => 1,
            Self::U16 => 2,
            Self::I16 => 3,
            Self::U32 => 4,
            Self::I32 => 5,
            Self::F32 => 6,
            Self::Bool => 7,
            Self::String => 8,
            Self::Array => 9,
            Self::U64 => 10,
            Self::I64 => 11,
            Self::F64 => 12,
        }
    }
}

trait AsyncReadExt2 {
    async fn read_u32_into(&mut self, n: u32) -> Result<Vec<u32>>;
    async fn read_u64_into(&mut self, n: u32) -> Result<Vec<u64>>;
}

impl<T: AsyncRead + Unpin> AsyncReadExt2 for T {
    async fn read_u32_into(&mut self, n: u32) -> Result<Vec<u32>> {
        let mut result = vec![];
        for i in 0..n {
            let elem = self.read_u32_le().await.map_err(error::Error::wrap)?;
            result.push(elem)
        }
        Ok(result)
    }
    async fn read_u64_into(&mut self, n: u32) -> Result<Vec<u64>> {
        let mut result = vec![];
        for i in 0..n {
            let elem = self.read_u64_le().await.map_err(error::Error::wrap)?;
            result.push(elem)
        }

        Ok(result)
    }
}

impl Content {
    pub async fn read<R: AsyncSeek + AsyncRead + Unpin + Send>(reader: &mut R) -> Result<Self> {
        let magic = VersionedMagic::read(reader).await?;

        let tensor_count = match magic {
            VersionedMagic::GgufV1 => reader.read_u32_le().await? as usize,
            VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => reader.read_u64_le().await? as usize,
        };
        let metadata_kv_count = match magic {
            VersionedMagic::GgufV1 => reader.read_u32_le().await? as usize,
            VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => reader.read_u64_le().await? as usize,
        };

        let mut metadata = HashMap::new();
        for _idx in 0..metadata_kv_count {
            let key = read_string(reader, &magic).await?;
            let value_type = reader.read_u32_le().await?;
            let value_type = ValueType::from_u32(value_type)?;
            let value = Value::read(reader, value_type, &magic).await?;
            metadata.insert(key, value);
        }
        let mut tensor_infos = HashMap::new();
        for _idx in 0..tensor_count {
            let tensor_name = read_string(reader, &magic).await?;
            let n_dimensions = reader.read_u32_le().await?;

            let mut dimensions: Vec<usize> = match magic {
                VersionedMagic::GgufV1 => {
                    // let mut dimensions = vec![0; n_dimensions as usize];
                    // reader.read_u32_into(&mut dimensions).await?;
                    let dimensions = reader.read_u32_into(n_dimensions).await?;
                    dimensions.into_iter().map(|c| c as usize).collect()
                }
                VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                    // let mut dimensions = vec![0; n_dimensions as usize];
                    // reader.read_u64_into(&mut dimensions).await?;
                    let dimensions = reader.read_u64_into(n_dimensions).await?;
                    dimensions.into_iter().map(|c| c as usize).collect()
                }
            };

            dimensions.reverse();
            let ggml_dtype = reader.read_u32_le().await?;
            let ggml_dtype = GgmlDType::from_u32(ggml_dtype)?;
            let offset = reader.read_u64_le().await?;
            tensor_infos.insert(
                tensor_name,
                TensorInfo {
                    shape: Shape::from(dimensions),
                    offset,
                    ggml_dtype,
                },
            );
        }
        let position = reader.stream_position().await?;
        let alignment = match metadata.get("general.alignment") {
            Some(Value::U8(v)) => *v as u64,
            Some(Value::U16(v)) => *v as u64,
            Some(Value::U32(v)) => *v as u64,
            Some(Value::I8(v)) if *v >= 0 => *v as u64,
            Some(Value::I16(v)) if *v >= 0 => *v as u64,
            Some(Value::I32(v)) if *v >= 0 => *v as u64,
            _ => DEFAULT_ALIGNMENT,
        };
        let tensor_data_offset = (position + alignment - 1) / alignment * alignment;
        Ok(Self {
            magic,
            metadata,
            tensor_infos,
            tensor_data_offset,
        })
    }

    pub async fn tensor<R: AsyncSeek + AsyncRead + Unpin>(
        &self,
        reader: &mut R,
        name: &str,
        device: &Device,
    ) -> Result<Tensor> {
        let tensor_info = match self.tensor_infos.get(name) {
            Some(tensor_info) => tensor_info,
            None => crate::bail!("cannot find tensor info for {name}"),
        };
        tensor_info
            .read(reader, self.tensor_data_offset, device)
            .await
    }
}

// fn write_string<W: std::io::Write>(w: &mut W, str: &str) -> Result<()> {
//     let bytes = str.as_bytes();
//     w.write_u64::<LittleEndian>(bytes.len() as u64)?;
//     w.write_all(bytes)?;
//     Ok(())
// }

// pub fn write<W: std::io::Seek + std::io::Write>(
//     w: &mut W,
//     metadata: &[(&str, &Value)],
//     tensors: &[(&str, &QTensor)],
// ) -> Result<()> {
//     w.write_u32::<LittleEndian>(0x46554747)?;
//     w.write_u32::<LittleEndian>(2)?; // version 2.
//     w.write_u64::<LittleEndian>(tensors.len() as u64)?;
//     w.write_u64::<LittleEndian>(metadata.len() as u64)?;
//     for (name, value) in metadata.iter() {
//         write_string(w, name)?;
//         w.write_u32::<LittleEndian>(value.value_type().to_u32())?;
//         value.write(w)?;
//     }
//     let mut offset = 0usize;
//     let mut offsets = Vec::with_capacity(tensors.len());
//     for (name, tensor) in tensors.iter() {
//         write_string(w, name)?;
//         let dims = tensor.shape().dims();
//         w.write_u32::<LittleEndian>(dims.len() as u32)?;
//         for &dim in dims.iter().rev() {
//             w.write_u64::<LittleEndian>(dim as u64)?;
//         }
//         w.write_u32::<LittleEndian>(tensor.dtype().to_u32())?;
//         w.write_u64::<LittleEndian>(offset as u64)?;
//         offsets.push(offset);
//         let size_in_bytes = tensor.storage_size_in_bytes();
//         let padding = 31 - (31 + size_in_bytes) % 32;
//         offset += size_in_bytes + padding;
//     }
//     let pos = w.stream_position()? as usize;
//     let padding = 31 - (31 + pos) % 32;
//     w.write_all(&vec![0u8; padding])?;
//     let tensor_start_pos = w.stream_position()? as usize;
//     for (offset, (_name, tensor)) in offsets.iter().zip(tensors.iter()) {
//         let pos = w.stream_position()? as usize;
//         if tensor_start_pos + offset != pos {
//             crate::bail!(
//                 "internal error, unexpected current position {tensor_start_pos} {offset} {pos}"
//             )
//         }
//         let data = tensor.data()?;
//         let size_in_bytes = data.len();
//         w.write_all(&data)?;
//         let padding = 31 - (31 + size_in_bytes) % 32;
//         w.write_all(&vec![0u8; padding])?;
//     }
//     Ok(())
// }
