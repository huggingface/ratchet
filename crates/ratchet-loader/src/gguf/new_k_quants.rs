// Adapted from https://github.com/huggingface/candle/blob/5ebcfeaf0f5af69bb2f74385e8d6b020d4a3b8df/candle-core/src/quantized/k_quants.rs

use anyhow::{anyhow, bail};
use half::f16;
use ratchet::gguf::Align;
use ratchet::{prelude::shape, Device, Tensor};

use byteorder::{ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Seek, SeekFrom, Write};

use super::utils::*;
use super::{ggml::GgmlDType, utils::*};
use crate::gguf::utils::WriteHalf;
use itertools::Itertools;
use ratchet::gguf;
use ratchet::GGUFDType;
use ratchet::{BufferSegment, DType};

// Default to QK_K 256 rather than 64.
pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;

pub trait GgmlType: Sized + Clone {
    const DTYPE: GgmlDType;
    const BLCK_SIZE: usize;
    const TYPE_SIZE: usize;
    type VecDotType: GgmlType;

    fn read<R: std::io::Seek + std::io::Read>(
        tensor_blocks: usize,
        reader: &mut R,
        device: &Device,
    ) -> std::prelude::v1::Result<Tensor, anyhow::Error>;

    fn write<W: std::io::Seek + std::io::Write>(
        tensor: Tensor,
        writer: &mut W,
    ) -> std::prelude::v1::Result<(), anyhow::Error>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    BlockQ4K(BlockQ4K),
    BlockF32(BlockF32),
    BlockQ6K(BlockQ6K),
}

impl Block {
    pub fn type_size(&self) -> usize {
        match &self {
            Block::BlockQ4K(blk) => BlockQ4K::TYPE_SIZE,
            Block::BlockF32(blk) => BlockF32::TYPE_SIZE,
            Block::BlockQ6K(blk) => BlockQ6K::TYPE_SIZE,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ4_0 {
    pub(crate) d: f32,
    pub(crate) qs: [u8; QK4_0 / 2],
}
// const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == 18);

#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ4_1 {
    pub(crate) d: f32,
    pub(crate) m: f32,
    pub(crate) qs: [u8; QK4_1 / 2],
}
// const _: () = assert!(std::mem::size_of::<BlockQ4_1>() == 20);

#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ5_0 {
    pub(crate) d: f32,
    pub(crate) qh: [u8; 4],
    pub(crate) qs: [u8; QK5_0 / 2],
}
// const _: () = assert!(std::mem::size_of::<BlockQ5_0>() == 22);

#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ5_1 {
    pub(crate) d: f32,
    pub(crate) m: f32,
    pub(crate) qh: [u8; 4],
    pub(crate) qs: [u8; QK5_1 / 2],
}
// const _: () = assert!(std::mem::size_of::<BlockQ5_1>() == 24);

#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ8_0 {
    pub(crate) d: f32,
    pub(crate) qs: [i8; QK8_0],
}
// const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);

#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ8_1 {
    pub(crate) d: f32,
    pub(crate) s: f32,
    pub(crate) qs: [i8; QK8_1],
}
// const _: () = assert!(std::mem::size_of::<BlockQ8_1>() == 36);

#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ2K {
    pub(crate) scales: [u8; QK_K / 16],
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) d: f32,
    pub(crate) dmin: f32,
}
// const _: () = assert!(QK_K / 16 + QK_K / 4 + 2 * 2 == std::mem::size_of::<BlockQ2K>());

#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ3K {
    pub(crate) hmask: [u8; QK_K / 8],
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) scales: [u8; 12],
    pub(crate) d: f32,
}
// const _: () = assert!(QK_K / 8 + QK_K / 4 + 12 + 2 == std::mem::size_of::<BlockQ3K>());

#[derive(Debug, Clone, PartialEq)]
// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/k_quants.h#L82
// pub struct BlockQ4K {
//     pub(crate) d: f32,
//     pub(crate) dmin: f32,
//     pub(crate) scales: [u8; K_SCALE_SIZE],
//     pub(crate) qs: [u8; QK_K / 2],
// }
// pub struct BlockQ4K {
//     pub(crate) d: Tensor,
//     pub(crate) dmin: Tensor,
//     pub(crate) scales: Tensor,
//     pub(crate) qs: Tensor,
// }
pub struct BlockQ4K {
    // inner: Tensor,
    // ds_offset: usize,
    // dmins_offset: usize,
    // scales_offset: usize,
    // qs_offset: usize,
}
// const _: () = assert!(QK_K / 2 + K_SCALE_SIZE + 2 * 2 == std::mem::size_of::<BlockQ4K>());

#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ5K {
    pub(crate) d: f32,
    pub(crate) dmin: f32,
    pub(crate) scales: [u8; K_SCALE_SIZE],
    pub(crate) qh: [u8; QK_K / 8],
    pub(crate) qs: [u8; QK_K / 2],
}
// const _: () =
//     assert!(QK_K / 8 + QK_K / 2 + 2 * 2 + K_SCALE_SIZE == std::mem::size_of::<BlockQ5K>());

// #[derive(Debug, Clone, PartialEq)]
// pub struct BlockQ6K {
//     pub(crate) ql: [u8; QK_K / 2],
//     pub(crate) qh: [u8; QK_K / 4],
//     pub(crate) scales: [i8; QK_K / 16],
//     pub(crate) d: f32,
// }
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ6K {
    pub(crate) ql: Tensor,
    pub(crate) qh: Tensor,
    pub(crate) scales: Tensor,
    pub(crate) d: Tensor,
}
// const _: () = assert!(3 * QK_K / 4 + QK_K / 16 + 2 == std::mem::size_of::<BlockQ6K>());

#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ8K {
    pub(crate) d: f32,
    pub(crate) qs: [i8; QK_K],
    pub(crate) bsums: [i16; QK_K / 16],
}
// const _: () = assert!(4 + QK_K + QK_K / 16 * 2 == std::mem::size_of::<BlockQ8K>());

impl GgmlType for BlockQ4_0 {
    const DTYPE: GgmlDType = GgmlDType::Q4_0;
    const BLCK_SIZE: usize = QK4_0;
    const TYPE_SIZE: usize = 2 + QK4_0 / 2;
    type VecDotType = BlockQ8_0;

    fn read<R: std::io::Seek + std::io::Read>(
        tensor_blocks: usize,
        reader: &mut R,
        device: &Device,
    ) -> std::prelude::v1::Result<Tensor, anyhow::Error> {
        todo!()
    }

    fn write<R: std::io::Write>(
        tensor: Tensor,
        writer: &mut R,
    ) -> std::prelude::v1::Result<(), anyhow::Error> {
        todo!()
    }
}

impl GgmlType for BlockQ4_1 {
    const DTYPE: GgmlDType = GgmlDType::Q4_1;
    const BLCK_SIZE: usize = QK4_1;
    const TYPE_SIZE: usize = 2 + 2 + QK4_1 / 2;
    type VecDotType = BlockQ8_1;

    fn read<R: std::io::Seek + std::io::Read>(
        tensor_blocks: usize,
        reader: &mut R,
        device: &Device,
    ) -> std::prelude::v1::Result<Tensor, anyhow::Error> {
        todo!()
    }

    fn write<R: std::io::Write>(
        tensor: Tensor,
        writer: &mut R,
    ) -> std::prelude::v1::Result<(), anyhow::Error> {
        todo!()
    }
}

impl GgmlType for BlockQ5_0 {
    const DTYPE: GgmlDType = GgmlDType::Q5_0;
    const BLCK_SIZE: usize = QK5_0;
    const TYPE_SIZE: usize = 2 + 4 + QK5_0 / 2;
    type VecDotType = BlockQ8_0;

    fn read<R: std::io::Seek + std::io::Read>(
        tensor_blocks: usize,
        reader: &mut R,
        device: &Device,
    ) -> std::prelude::v1::Result<Tensor, anyhow::Error> {
        todo!()
    }

    fn write<R: std::io::Write>(
        tensor: Tensor,
        writer: &mut R,
    ) -> std::prelude::v1::Result<(), anyhow::Error> {
        todo!()
    }
}

impl GgmlType for BlockQ5_1 {
    const DTYPE: GgmlDType = GgmlDType::Q5_1;
    const BLCK_SIZE: usize = QK5_1;
    const TYPE_SIZE: usize = 2 + 2 + 4 + QK5_1 / 2;
    type VecDotType = BlockQ8_1;

    fn read<R: std::io::Seek + std::io::Read>(
        tensor_blocks: usize,
        reader: &mut R,
        device: &Device,
    ) -> std::prelude::v1::Result<Tensor, anyhow::Error> {
        todo!()
    }

    fn write<R: std::io::Write>(
        tensor: Tensor,
        writer: &mut R,
    ) -> std::prelude::v1::Result<(), anyhow::Error> {
        todo!()
    }
}

impl GgmlType for BlockQ8_0 {
    const DTYPE: GgmlDType = GgmlDType::Q8_0;
    const BLCK_SIZE: usize = QK8_0;
    const TYPE_SIZE: usize = 2 + QK8_0;
    type VecDotType = BlockQ8_0;

    fn read<R: std::io::Seek + std::io::Read>(
        tensor_blocks: usize,
        reader: &mut R,
        device: &Device,
    ) -> std::prelude::v1::Result<Tensor, anyhow::Error> {
        todo!()
    }
    fn write<R: std::io::Write>(
        tensor: Tensor,
        writer: &mut R,
    ) -> std::prelude::v1::Result<(), anyhow::Error> {
        todo!()
    }
}

impl GgmlType for BlockQ8_1 {
    const DTYPE: GgmlDType = GgmlDType::Q8_1;
    const BLCK_SIZE: usize = QK8_1;
    const TYPE_SIZE: usize = 2 + 2 + QK8_1;
    type VecDotType = BlockQ8_1;

    fn read<R: std::io::Seek + std::io::Read>(
        tensor_blocks: usize,
        reader: &mut R,
        device: &Device,
    ) -> std::prelude::v1::Result<Tensor, anyhow::Error> {
        todo!()
    }
    fn write<R: std::io::Write>(
        tensor: Tensor,
        writer: &mut R,
    ) -> std::prelude::v1::Result<(), anyhow::Error> {
        todo!()
    }
}

impl GgmlType for BlockQ2K {
    const DTYPE: GgmlDType = GgmlDType::Q2K;
    const BLCK_SIZE: usize = QK_K;
    const TYPE_SIZE: usize = QK_K / 16 + QK_K / 4 + 2 + 2;
    type VecDotType = BlockQ8K;

    fn read<R: std::io::Seek + std::io::Read>(
        tensor_blocks: usize,
        reader: &mut R,
        device: &Device,
    ) -> std::prelude::v1::Result<Tensor, anyhow::Error> {
        todo!()
    }
    fn write<R: std::io::Write>(
        tensor: Tensor,
        writer: &mut R,
    ) -> std::prelude::v1::Result<(), anyhow::Error> {
        todo!()
    }
}

impl GgmlType for BlockQ3K {
    const DTYPE: GgmlDType = GgmlDType::Q3K;
    const BLCK_SIZE: usize = QK_K;
    const TYPE_SIZE: usize = QK_K / 8 + QK_K / 4 + 12 + 2;
    type VecDotType = BlockQ8K;

    fn read<R: std::io::Seek + std::io::Read>(
        tensor_blocks: usize,
        reader: &mut R,
        device: &Device,
    ) -> std::prelude::v1::Result<Tensor, anyhow::Error> {
        todo!()
    }
    fn write<R: std::io::Write>(
        tensor: Tensor,
        writer: &mut R,
    ) -> std::prelude::v1::Result<(), anyhow::Error> {
        todo!()
    }
}

pub trait Padding {
    fn align_standard(&mut self) -> usize;
}

impl<T: Clone + Default> Padding for Vec<T> {
    fn align_standard(&mut self) -> usize {
        let length = &self.len();
        let alignment = length.calculate_alignment();
        if alignment != 0 {
            let default_value: T = Default::default();
            let mut padding = vec![default_value; alignment];
            self.append(&mut padding);
            alignment
        } else {
            0
        }
    }
}

impl GgmlType for BlockQ4K {
    const DTYPE: GgmlDType = GgmlDType::Q4K;
    const BLCK_SIZE: usize = QK_K;
    const TYPE_SIZE: usize = QK_K / 2 + K_SCALE_SIZE + 2 * 2;
    type VecDotType = BlockQ8K;

    fn read<R: std::io::Seek + std::io::Read>(
        tensor_blocks: usize,
        reader: &mut R,
        device: &Device,
    ) -> std::prelude::v1::Result<Tensor, anyhow::Error> {
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

        let mut ds = ds
            .iter()
            .map(|d| d.to_le_bytes())
            .flatten()
            .collect::<Vec<u8>>();

        let mut dmins = dmins
            .iter()
            .map(|d| d.to_le_bytes())
            .flatten()
            .collect::<Vec<u8>>();

        let mut tensor_data = vec![0u32; 0];

        let ds_alignment = ds.align_standard();
        let mut ds_u32 = bytemuck::cast_slice::<u8, u32>(&ds).to_vec();
        tensor_data.append(&mut ds_u32);

        let dmins_alignment = dmins.align_standard();
        let mut dmins_u32 = bytemuck::cast_slice::<u8, u32>(&dmins).to_vec();
        tensor_data.append(&mut dmins_u32);

        let scales_alignment = scales.align_standard();
        let mut scales_u32 = bytemuck::cast_slice::<u8, u32>(&scales).to_vec();
        tensor_data.append(&mut scales_u32);

        let qs_alignment = qs.align_standard();
        let mut qs_u32 = bytemuck::cast_slice::<u8, u32>(&qs).to_vec();
        tensor_data.append(&mut qs_u32);

        let tensor_data_len = (&tensor_data).len();

        let inner = unsafe {
            Tensor::from_quantized::<u32, _>(
                &tensor_data,
                DType::GGUF(GGUFDType::Q4K),
                shape![256, 256],
                device.clone(),
            )
        };

        Ok(inner)
    }

    fn write<W: std::io::Seek + std::io::Write>(
        tensor: Tensor,
        writer: &mut W,
    ) -> std::prelude::v1::Result<(), anyhow::Error> {
        let GGUFDType::Q4K = match tensor.dt() {
            DType::GGUF(q4k @ GGUFDType::Q4K) => Ok(q4k),
            otherwise => Err(anyhow!(
                "Got an invalid datatype while trying to parse q4k: {:?}",
                otherwise
            )),
        }?;

        let tensor_blocks = tensor
            .shape()
            .get(0)
            .ok_or(anyhow!("Failed to get tensor blocks"))?
            .clone();

        let segments = gguf::GGUFDType::Q4K.segments(tensor_blocks);
        let (ds_segment, dmins_segment, scales_segment, qs_segment) = segments
            .iter()
            .collect_tuple()
            .ok_or(anyhow!("Invalid segmentation found in Q4K"))?;

        //Can't just call `to_vec` here.
        //Do something like: https://github.com/FL33TW00D/wgpu-bench/blob/master/src/quant.rs#L105
        // let tensor_data = tensor.to(&ratchet::Device::CPU)?.to_vec::<u32>()?;
        let tensor_data = unsafe { tensor.into_bytes()? };

        let ds_offset = ds_segment.offset;
        let mut ds_cursor = Cursor::new(&tensor_data);
        ds_cursor.seek(SeekFrom::Start(ds_offset))?;

        let dmins_offset = dmins_segment.offset;
        let mut dmins_cursor = Cursor::new(&tensor_data);
        dmins_cursor.seek(SeekFrom::Start(dmins_offset))?;

        let scales_offset = scales_segment.offset;
        let mut scales_data_cursor = Cursor::new(&tensor_data);
        scales_data_cursor.seek(SeekFrom::Start(scales_offset))?;

        let qs_offset = qs_segment.offset;
        let mut qs_data_cursor = Cursor::new(&tensor_data);
        qs_data_cursor.seek(SeekFrom::Start(qs_offset))?;

        for _idx in 0..tensor_blocks {
            let d_value = ds_cursor.read_f32::<LittleEndian>()?;
            let d_value = half::f16::from_f32(d_value);
            writer.write_f16(d_value)?;

            let dmin_value = dmins_cursor.read_f32::<LittleEndian>()?;
            let dmin_value = half::f16::from_f32(dmin_value);
            writer.write_f16(dmin_value)?;

            let current_scales = scales_data_cursor.read_len_bytes(K_SCALE_SIZE)?;
            writer.write_all(current_scales.as_ref())?;

            let current_qs = qs_data_cursor.read_len_bytes(QK_K / 2)?;
            writer.write_all(current_qs.as_ref())?;
        }

        Ok(())
    }
}

fn read_data_from_cursor(
    cursor: &mut Cursor<&mut Vec<u8>>,
    offset: u64,
    length: usize,
) -> anyhow::Result<Vec<u8>> {
    let mut current_scales = vec![0u8; length];
    let pos = std::io::SeekFrom::Start(offset);
    cursor.seek(pos)?;
    cursor.read_exact(&mut current_scales)?;
    Ok(current_scales)
}

// https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L928
impl GgmlType for BlockQ5K {
    const DTYPE: GgmlDType = GgmlDType::Q5K;
    const BLCK_SIZE: usize = QK_K;
    const TYPE_SIZE: usize = QK_K / 8 + QK_K / 2 + 2 * 2 + K_SCALE_SIZE;
    type VecDotType = BlockQ8K;

    fn read<R: std::io::Seek + std::io::Read>(
        tensor_blocks: usize,
        reader: &mut R,
        device: &Device,
    ) -> std::prelude::v1::Result<Tensor, anyhow::Error> {
        todo!()
    }
    fn write<R: std::io::Write>(
        tensor: Tensor,
        writer: &mut R,
    ) -> std::prelude::v1::Result<(), anyhow::Error> {
        todo!()
    }
}

impl GgmlType for BlockQ6K {
    const DTYPE: GgmlDType = GgmlDType::Q6K;
    const BLCK_SIZE: usize = QK_K;
    const TYPE_SIZE: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 4;
    type VecDotType = BlockQ6K;

    // pub struct BlockQ6K {
    //     pub(crate) ql: [u8; QK_K / 2],
    //     pub(crate) qh: [u8; QK_K / 4],
    //     pub(crate) scales: [i8; QK_K / 16],
    //     pub(crate) d: f32,
    // }

    fn read<R: std::io::Seek + std::io::Read>(
        tensor_blocks: usize,
        reader: &mut R,
        device: &Device,
    ) -> std::prelude::v1::Result<Tensor, anyhow::Error> {
        let mut qls = vec![0u8; tensor_blocks * QK_K / 2];
        let mut qls_cursor = Cursor::new(&mut qls);

        let mut qhs = vec![0u8; tensor_blocks * QK_K / 4];
        let mut qhs_cursor = Cursor::new(&mut qhs);

        // [TODO] See if we actually need i8
        let mut scales = vec![0u8; tensor_blocks * QK_K / 16];
        let mut scales_cursor = Cursor::new(&mut scales);

        let mut ds = vec![0f32; tensor_blocks];

        for _idx in 0..tensor_blocks {
            reader.read_u8s_into(&mut qls_cursor, QK_K / 2)?;
            reader.read_u8s_into(&mut qhs_cursor, QK_K / 4)?;
            reader.read_u8s_into(&mut scales_cursor, QK_K / 16)?;
            ds[_idx] = reader.read_f32::<LittleEndian>()?;
        }

        let qls_tensor = Tensor::from_bytes(
            qls.as_ref(),
            ratchet::DType::U32,
            shape![tensor_blocks, QK_K / 2 / 4],
            device.clone(),
        )?;

        let qhs_tensor = Tensor::from_bytes(
            qhs.as_ref(),
            ratchet::DType::U32,
            shape![tensor_blocks, QK_K / 4 / 4],
            device.clone(),
        )?;

        let scales_tensor = Tensor::from_bytes(
            scales.as_ref(),
            ratchet::DType::U32,
            shape![tensor_blocks, QK_K / 16 / 4],
            device.clone(),
        )?;
        let ds_tensor = Tensor::from_data(&ds, shape![tensor_blocks], device.clone());

        let block_q6k: BlockQ6K = BlockQ6K {
            ql: qls_tensor,
            qh: qhs_tensor,
            scales: scales_tensor,
            d: ds_tensor,
        };
        // Ok(Block::BlockQ6K(block_q6k))
        todo!()
    }
    fn write<R: std::io::Write>(
        tensor: Tensor,
        writer: &mut R,
    ) -> std::prelude::v1::Result<(), anyhow::Error> {
        // let BlockQ6K { ql, qh, scales, d } = self;

        // let tensor_blocks = d
        //     .shape()
        //     .get(0)
        //     .ok_or(anyhow!("Unable to get tensor blocks"))?;

        // let ql_data = ql.to(&ratchet::Device::CPU)?.to_vec::<u32>()?;
        // let mut ql_data = ql_data
        //     .iter()
        //     .map(|value| value.to_le_bytes())
        //     .flatten()
        //     .collect::<Vec<u8>>();
        // let mut ql_data_cursor = Cursor::new(&mut ql_data);

        // let qh_data = qh.to(&ratchet::Device::CPU)?.to_vec::<u32>()?;
        // let mut qh_data = qh_data
        //     .iter()
        //     .map(|value| value.to_le_bytes())
        //     .flatten()
        //     .collect::<Vec<u8>>();
        // let mut qh_data_cursor = Cursor::new(&mut qh_data);

        // let scales_data = scales.to(&ratchet::Device::CPU)?.to_vec::<u32>()?;
        // let mut scales_data = scales_data
        //     .iter()
        //     .map(|value| value.to_le_bytes())
        //     .flatten()
        //     .collect::<Vec<u8>>();
        // let mut scales_data_cursor = Cursor::new(&mut scales_data);

        // let d_data = d.to(&ratchet::Device::CPU)?.to_vec::<f32>()?;

        // for _idx in 0..*tensor_blocks {
        //     let current_ql =
        //         read_data_from_cursor(&mut ql_data_cursor, (_idx * QK_K / 2) as u64, QK_K / 2)?;
        //     writer.write_all(current_ql.as_ref())?;

        //     let current_qh =
        //         read_data_from_cursor(&mut qh_data_cursor, (_idx * QK_K / 4) as u64, QK_K / 4)?;
        //     writer.write_all(current_qh.as_ref())?;

        //     let current_scales = read_data_from_cursor(
        //         &mut scales_data_cursor,
        //         (_idx * QK_K / 16) as u64,
        //         QK_K / 16,
        //     )?;
        //     writer.write_all(current_scales.as_ref())?;
        //     writer.write_f32::<LittleEndian>(d_data[_idx])?;
        // }
        Ok(())
    }
}

impl GgmlType for BlockQ8K {
    const DTYPE: GgmlDType = GgmlDType::Q8K;
    const BLCK_SIZE: usize = QK_K;
    const TYPE_SIZE: usize = 4 + QK_K + QK_K / 16 * 2;
    type VecDotType = BlockQ8K;

    fn read<R: std::io::Seek + std::io::Read>(
        tensor_blocks: usize,
        reader: &mut R,
        device: &Device,
    ) -> std::prelude::v1::Result<Tensor, anyhow::Error> {
        todo!()
    }
    fn write<R: std::io::Write>(
        tensor: Tensor,
        writer: &mut R,
    ) -> std::prelude::v1::Result<(), anyhow::Error> {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockF32(Tensor);

impl GgmlType for BlockF32 {
    const DTYPE: GgmlDType = GgmlDType::F32;
    const BLCK_SIZE: usize = 1;
    const TYPE_SIZE: usize = 4;
    type VecDotType = BlockF32;

    fn read<R: std::io::Seek + std::io::Read>(
        tensor_blocks: usize,
        reader: &mut R,
        device: &Device,
    ) -> std::prelude::v1::Result<Tensor, anyhow::Error> {
        println!("tensor_blocks: {:?}", tensor_blocks);

        let mut data = vec![0f32; tensor_blocks];
        for _idx in 0..tensor_blocks {
            data[_idx] = reader.read_f32::<LittleEndian>()?;
        }

        let tensor = Tensor::from_data(data, shape![tensor_blocks], device.clone());
        Ok(tensor)
    }
    fn write<R: std::io::Write>(
        tensor: Tensor,
        writer: &mut R,
    ) -> std::prelude::v1::Result<(), anyhow::Error> {
        // let tensor = &self.0;
        // let tensor_data = tensor.to(&ratchet::Device::CPU)?.to_vec::<f32>()?;
        // let tensor_blocks = tensor.shape().get(0).unwrap(); // [TODO] Handle result properly

        // for _idx in 0..*tensor_blocks {
        //     writer.write_f32::<LittleEndian>(tensor_data[_idx])?;
        // }
        Ok(())
    }
}

// impl GgmlType for f16 {
//     const DTYPE: GgmlDType = GgmlDType::F16;
//     const BLCK_SIZE: usize = 1;
//     type VecDotType = f16;

// fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
//     Self::vec_dot_unopt(n, xs, ys)
// }

// fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> Result<f32> {
//     if xs.len() < n {
//         crate::bail!("size mismatch {} < {n}", xs.len())
//     }
//     if ys.len() < n {
//         crate::bail!("size mismatch {} < {n}", ys.len())
//     }
//     let mut res = 0f32;
//     unsafe { crate::cpu::vec_dot_f16(xs.as_ptr(), ys.as_ptr(), &mut res, n) };
//     Ok(res)
// }

// fn from_float(xs: &[f32], ys: &mut [Self]) -> Result<()> {
//     if xs.len() != ys.len() {
//         crate::bail!("size mismatch {} {}", xs.len(), ys.len());
//     }
//     // TODO: vectorize
//     for (x, y) in xs.iter().zip(ys.iter_mut()) {
//         *y = f16::from_f32(*x)
//     }
//     Ok(())
// }

//     fn to_float(xs: &[Self], ys: &mut [f32]) -> Result<()> {
//         if xs.len() != ys.len() {
//             crate::bail!("size mismatch {} {}", xs.len(), ys.len());
//         }
//         // TODO: vectorize
//         for (x, y) in xs.iter().zip(ys.iter_mut()) {
//             *y = x
//         }
//         Ok(())
//     }
// }
