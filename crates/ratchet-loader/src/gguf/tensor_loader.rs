// Adapted from https://github.com/huggingface/candle/blob/5ebcfeaf0f5af69bb2f74385e8d6b020d4a3b8df/candle-core/src/quantized/k_quants.rs

use anyhow::anyhow;
use ratchet::gguf::Align;
use ratchet::{prelude::shape, Device, Tensor};

use byteorder::{ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Seek, SeekFrom, Write};

use super::ggml::GgmlDType;
use super::utils::*;
use crate::gguf::utils::WriteHalf;
use itertools::Itertools;
use ratchet::gguf::{K_SCALE_SIZE, QK_K};
use ratchet::DType;
use ratchet::GGUFDType;
use ratchet::{gguf, Segments};

pub trait TensorLoader: Sized + Clone {
    const GGML_DTYPE: GgmlDType;

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

impl TensorLoader for gguf::Q4K {
    const GGML_DTYPE: GgmlDType = GgmlDType::Q4K;

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

        let mut ds = ds.iter().flat_map(|d| d.to_le_bytes()).collect::<Vec<u8>>();

        let mut dmins = dmins
            .iter()
            .flat_map(|d| d.to_le_bytes())
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

        let inner = unsafe {
            Tensor::from_quantized::<u32, _>(
                &tensor_data,
                DType::GGUF(GGUFDType::Q4K(gguf::Q4K::new())),
                shape![tensor_blocks, QK_K],
                device.clone(),
            )
        };

        Ok(inner)
    }

    fn write<W: std::io::Seek + std::io::Write>(
        tensor: Tensor,
        writer: &mut W,
    ) -> std::prelude::v1::Result<(), anyhow::Error> {
        match tensor.dt() {
            DType::GGUF(GGUFDType::Q4K(_)) => Ok(()),
            otherwise => Err(anyhow!(
                "Got an invalid datatype while trying to parse q4k: {:?}",
                otherwise
            )),
        }?;

        let tensor_blocks = *tensor
            .shape()
            .get(0)
            .ok_or(anyhow!("Failed to get tensor blocks"))?;

        let segments = gguf::Q4K::segments(tensor_blocks);
        let (ds_segment, dmins_segment, scales_segment, qs_segment) = segments
            .iter()
            .collect_tuple()
            .ok_or(anyhow!("Invalid segmentation found in Q4K"))?;

        //Can't just call `to_vec` here.
        //Do something like: https://github.com/FL33TW00D/wgpu-bench/blob/master/src/quant.rs#L105
        // let tensor_data = tensor.to(&ratchet::Device::CPU)?.to_vec::<u32>()?;
        // [TODO] Check if we can improve on this
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

impl TensorLoader for gguf::Q6K {
    const GGML_DTYPE: GgmlDType = GgmlDType::Q6K;

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

        let mut tensor_data = vec![0u32; 0];

        let qls_alignment = qls.align_standard();
        let mut qls_u32 = bytemuck::cast_slice::<u8, u32>(&qls).to_vec();
        tensor_data.append(&mut qls_u32);

        let qhs_alignment = qhs.align_standard();
        let mut qhs_u32 = bytemuck::cast_slice::<u8, u32>(&qhs).to_vec();
        tensor_data.append(&mut qhs_u32);

        let scales_alignment = scales.align_standard();
        let mut scales_u32 = bytemuck::cast_slice::<u8, u32>(&scales).to_vec();
        tensor_data.append(&mut scales_u32);

        let ds_alignment = ds.align_standard();
        let mut ds_u32 = bytemuck::cast_slice::<f32, u32>(&ds).to_vec();
        tensor_data.append(&mut ds_u32);

        let inner = unsafe {
            Tensor::from_quantized::<u32, _>(
                &tensor_data,
                DType::GGUF(GGUFDType::Q6K(gguf::Q6K::new())),
                shape![tensor_blocks, QK_K],
                device.clone(),
            )
        };

        Ok(inner)
    }

    fn write<R: std::io::Write>(
        tensor: Tensor,
        writer: &mut R,
    ) -> std::prelude::v1::Result<(), anyhow::Error> {
        match tensor.dt() {
            DType::GGUF(GGUFDType::Q6K(_)) => Ok(()),
            otherwise => Err(anyhow!(
                "Got an invalid datatype while trying to parse q6k: {:?}",
                otherwise
            )),
        }?;

        let tensor_blocks = *tensor
            .shape()
            .get(0)
            .ok_or(anyhow!("Failed to get tensor blocks"))?;

        let segments = gguf::Q6K::segments(tensor_blocks);
        let (ql_segment, qh_segment, scales_segment, d_segment) =
            segments
                .iter()
                .collect_tuple()
                .ok_or(anyhow!("Invalid segmentation found in Q4K"))?;

        // [TODO] Check if we can improve on this
        let tensor_data = unsafe { tensor.into_bytes()? };

        let ql_offset = ql_segment.offset;
        let mut ql_data_cursor = Cursor::new(&tensor_data);
        ql_data_cursor.seek(SeekFrom::Start(ql_offset))?;

        let qh_offset = qh_segment.offset;
        let mut qh_data_cursor = Cursor::new(&tensor_data);
        qh_data_cursor.seek(SeekFrom::Start(qh_offset))?;

        let scales_offset = scales_segment.offset;
        let mut scales_data_cursor = Cursor::new(&tensor_data);
        scales_data_cursor.seek(SeekFrom::Start(scales_offset))?;

        let d_offset = d_segment.offset;
        let mut d_data_cursor = Cursor::new(&tensor_data);
        d_data_cursor.seek(SeekFrom::Start(d_offset))?;

        for _idx in 0..tensor_blocks {
            let current_ql = ql_data_cursor.read_len_bytes(QK_K / 2)?;
            writer.write_all(current_ql.as_ref())?;

            let current_qh = qh_data_cursor.read_len_bytes(QK_K / 4)?;
            writer.write_all(current_qh.as_ref())?;

            let current_scales = scales_data_cursor.read_len_bytes(QK_K / 16)?;
            writer.write_all(current_scales.as_ref())?;

            let d_value = d_data_cursor.read_f32::<LittleEndian>()?;
            writer.write_f32::<LittleEndian>(d_value)?;
        }

        Ok(())
    }
}

impl TensorLoader for f32 {
    const GGML_DTYPE: GgmlDType = GgmlDType::F32;

    fn read<R: std::io::Seek + std::io::Read>(
        tensor_blocks: usize,
        reader: &mut R,
        device: &Device,
    ) -> std::prelude::v1::Result<Tensor, anyhow::Error> {
        let mut data = vec![0f32; tensor_blocks];
        data.align_standard();
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
        let tensor_data = tensor.to_vec::<f32>()?;
        let tensor_blocks = tensor
            .shape()
            .get(0)
            .ok_or(anyhow!("Failed to get tensor shape"))?;

        for _idx in 0..*tensor_blocks {
            writer.write_f32::<LittleEndian>(tensor_data[_idx])?;
        }
        Ok(())
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
