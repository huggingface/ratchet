//! Support for the GGUF file format.
//!
//! Spec: https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md
//!
//! Adapted from https://github.com/huggingface/candle/blob/5ebcfeaf0f5af69bb2f74385e8d6b020d4a3b8df/candle-core/src/quantized/gguf_file.rs

use crate::error::{Error, Result};
use crate::gguf::ggml::GgmlDType;
use crate::gguf::k_quants::{BlockQ4K, K_SCALE_SIZE, QK_K};
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};
use half::f16;
use ratchet::{shape, Device, Shape, Tensor, TensorDType};
use std::collections::HashMap;
use std::io::{Cursor, Read, Write};

use super::k_quants::Block;

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
    fn read<R: std::io::Read>(reader: &mut R) -> Result<Self> {
        let magic = reader.read_u32::<LittleEndian>()?;
        let magic = Magic::try_from(magic)?;
        let version = reader.read_u32::<LittleEndian>()?;
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

trait ReadHalf {
    fn read_f16(&mut self) -> Result<f16>;
}

impl<R: std::io::Seek + std::io::Read> ReadHalf for R {
    fn read_f16(&mut self) -> Result<f16> {
        let mut d = [0u8; 2];
        self.read_exact(&mut d)?;
        let f16_value = half::f16::from_le_bytes(d);
        Ok(f16_value)
    }
}
trait ReadInto<Other> {
    fn read_u8s_into(&mut self, other: &mut Other, length: usize) -> Result<()>;
}

impl<R: std::io::Seek + std::io::Read, Other: std::io::Write> ReadInto<Other> for R {
    fn read_u8s_into(&mut self, other: &mut Other, length: usize) -> Result<()> {
        let mut temp = vec![0u8; length];
        self.read_exact(&mut temp)?;
        other.write_all(&mut temp)?;
        Ok(())
    }
}

impl TensorInfo {
    pub fn read<R: std::io::Seek + std::io::Read>(
        &self,
        reader: &mut R,
        tensor_data_offset: u64,
        device: &Device,
    ) -> anyhow::Result<Block> {
        // let tensor_elems = self.shape.elem_count();
        let tensor_elems = self.shape.numel();
        let block_size = self.ggml_dtype.block_size();
        if tensor_elems % block_size != 0 {
            anyhow::bail!(
            "the number of elements {tensor_elems} is not divisible by the block size {block_size}"
        )
        }

        println!("tensor_elems={:?}", tensor_elems);
        println!("block_size={:?}", block_size);
        println!("shape={:?} ggml_dtype={:?}", self.shape, self.ggml_dtype);
        let tensor_blocks = tensor_elems / block_size;
        // let size_in_bytes = tensor_blocks * self.ggml_dtype.type_size();

        reader.seek(std::io::SeekFrom::Start(tensor_data_offset + self.offset))?;

        // [TODO] Implement
        match self.ggml_dtype {
            GgmlDType::Q4K => read_q4k(tensor_blocks, reader, device),
            _ => anyhow::bail!("Not yet implemented"),
        }
    }
}

// [TODO] Move to GgmlType
fn read_q4k<R: std::io::Seek + std::io::Read>(
    tensor_blocks: usize,
    reader: &mut R,
    device: &Device,
) -> std::prelude::v1::Result<Block, anyhow::Error> {
    let mut ds = vec![0f32; tensor_blocks];
    let mut dmins = vec![0f32; tensor_blocks];
    let mut scales = vec![0u8; tensor_blocks * K_SCALE_SIZE];
    let mut scales_cursor = Cursor::new(&mut scales);

    let mut qs = vec![0u8; tensor_blocks * QK_K / 2];
    let mut qs_cursor = Cursor::new(&mut qs);

    for _idx in 0..tensor_blocks {
        ds[_idx] = reader.read_f16()?.to_f32();
        dmins[_idx] = reader.read_f16()?.to_f32();

        reader.read_u8s_into(&mut scales_cursor, K_SCALE_SIZE)?;
        reader.read_u8s_into(&mut qs_cursor, QK_K / 2)?;
    }

    let ds_tensor = Tensor::from_data(&ds, shape![tensor_blocks], device.clone());
    let dmins_tensor = Tensor::from_data(dmins, shape![tensor_blocks], device.clone());

    let scales_tensor = Tensor::from_bytes(
        qs.as_ref(),
        ratchet::DType::U32,
        shape![tensor_blocks, K_SCALE_SIZE],
        device.clone(),
    )?;

    let qs_tensor = Tensor::from_bytes(
        qs.as_ref(),
        ratchet::DType::U32,
        shape![tensor_blocks, QK_K / 2],
        device.clone(),
    )?;
    let block_q4k: BlockQ4K = BlockQ4K {
        d: ds_tensor,
        dmin: dmins_tensor,
        scales: scales_tensor,
        qs: qs_tensor,
    };
    let block: Block = Block::BlockQ4K(block_q4k);
    Ok(block)
}

#[derive(Debug)]
pub struct Content {
    pub magic: VersionedMagic,
    pub metadata: HashMap<String, Value>,
    pub tensor_infos: HashMap<String, TensorInfo>,
    pub tensor_data_offset: u64,
}

fn read_string<R: std::io::Read>(reader: &mut R, magic: &VersionedMagic) -> Result<String> {
    let len = match magic {
        VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as usize,
        VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
            reader.read_u64::<LittleEndian>()? as usize
        }
    };
    let mut v = vec![0u8; len];
    reader.read_exact(&mut v)?;
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

    fn read<R: std::io::Read>(
        reader: &mut R,
        value_type: ValueType,
        magic: &VersionedMagic,
    ) -> Result<Self> {
        let v = match value_type {
            ValueType::U8 => Self::U8(reader.read_u8()?),
            ValueType::I8 => Self::I8(reader.read_i8()?),
            ValueType::U16 => Self::U16(reader.read_u16::<LittleEndian>()?),
            ValueType::I16 => Self::I16(reader.read_i16::<LittleEndian>()?),
            ValueType::U32 => Self::U32(reader.read_u32::<LittleEndian>()?),
            ValueType::I32 => Self::I32(reader.read_i32::<LittleEndian>()?),
            ValueType::U64 => Self::U64(reader.read_u64::<LittleEndian>()?),
            ValueType::I64 => Self::I64(reader.read_i64::<LittleEndian>()?),
            ValueType::F32 => Self::F32(reader.read_f32::<LittleEndian>()?),
            ValueType::F64 => Self::F64(reader.read_f64::<LittleEndian>()?),
            ValueType::Bool => match reader.read_u8()? {
                0 => Self::Bool(false),
                1 => Self::Bool(true),
                b => crate::bail!("unexpected bool value {b}"),
            },
            ValueType::String => Self::String(read_string(reader, magic)?),
            ValueType::Array => {
                let value_type = reader.read_u32::<LittleEndian>()?;
                let value_type = ValueType::from_u32(value_type)?;
                let len = match magic {
                    VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as usize,
                    VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                        reader.read_u64::<LittleEndian>()? as usize
                    }
                };
                let mut vs = Vec::with_capacity(len);
                for _ in 0..len {
                    vs.push(Value::read(reader, value_type, magic)?)
                }
                Self::Array(vs)
            }
        };
        Ok(v)
    }
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

impl Content {
    pub fn read<R: std::io::Seek + std::io::Read>(reader: &mut R) -> Result<Self> {
        let magic = VersionedMagic::read(reader)?;

        let tensor_count = match magic {
            VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as usize,
            VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                reader.read_u64::<LittleEndian>()? as usize
            }
        };
        let metadata_kv_count = match magic {
            VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as usize,
            VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                reader.read_u64::<LittleEndian>()? as usize
            }
        };

        let mut metadata = HashMap::new();
        for _idx in 0..metadata_kv_count {
            let key = read_string(reader, &magic)?;
            let value_type = reader.read_u32::<LittleEndian>()?;
            let value_type = ValueType::from_u32(value_type)?;
            let value = Value::read(reader, value_type, &magic)?;
            metadata.insert(key, value);
        }
        let mut tensor_infos = HashMap::new();
        for _idx in 0..tensor_count {
            let tensor_name = read_string(reader, &magic)?;
            let n_dimensions = reader.read_u32::<LittleEndian>()?;

            let mut dimensions: Vec<usize> = match magic {
                VersionedMagic::GgufV1 => {
                    let mut dimensions = vec![0; n_dimensions as usize];
                    reader.read_u32_into::<LittleEndian>(&mut dimensions)?;
                    dimensions.into_iter().map(|c| c as usize).collect()
                }
                VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                    let mut dimensions = vec![0; n_dimensions as usize];
                    reader.read_u64_into::<LittleEndian>(&mut dimensions)?;
                    dimensions.into_iter().map(|c| c as usize).collect()
                }
            };

            dimensions.reverse();
            let ggml_dtype = reader.read_u32::<LittleEndian>()?;
            let ggml_dtype = GgmlDType::from_u32(ggml_dtype)?;
            let offset = reader.read_u64::<LittleEndian>()?;
            tensor_infos.insert(
                tensor_name,
                TensorInfo {
                    shape: Shape::from(dimensions),
                    offset,
                    ggml_dtype,
                },
            );
        }
        let position = reader.stream_position()?;
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

    pub fn tensor<R: std::io::Seek + std::io::Read>(
        &self,
        reader: &mut R,
        name: &str,
        device: &Device,
    ) -> anyhow::Result<Block> {
        let tensor_info = match self.tensor_infos.get(name) {
            Some(tensor_info) => tensor_info,
            None => anyhow::bail!("cannot find tensor info for {name}"),
        };
        tensor_info.read(reader, self.tensor_data_offset, device)
    }
}
