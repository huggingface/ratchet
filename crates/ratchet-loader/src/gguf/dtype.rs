#![allow(non_camel_case_types)]
use half::f16;
use ratchet::gguf::*;
use ratchet::{DType, Device, Padding, Shape, Tensor};

use crate::k_quants::*;

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;

/// # GGUF Interoperability
///
/// Supported GGUF types with the ability to be loaded into Ratchet.
pub trait GGUFInterop {
    //Associated block type
    type GGUF_TYPE: GGType;
    //Number of elements in a block
    const BLCK_NUMEL: usize;
    //Size of the block in bytes
    const TYPE_SIZE: usize = std::mem::size_of::<Self::GGUF_TYPE>();
    //Size of block in bytes for WebGPU (i.e f16 isn't supported, so f32 is used, increasing size)
    const TYPE_SIZE_WEBGPU: usize;

    //Given differences between GGUF and Ratchet, we need to "transcode" tensors from the raw GGUF
    //data into a format consumable by Ratchet.
    fn transcode(
        data: &[Self::GGUF_TYPE],
        n_blocks: usize,
        shape: Shape,
        device: &Device,
    ) -> anyhow::Result<Tensor>;
}

impl GGUFInterop for Q4K {
    type GGUF_TYPE = BlockQ4K;
    const BLCK_NUMEL: usize = QK_K;
    const TYPE_SIZE_WEBGPU: usize = Self::TYPE_SIZE + 4;

    fn transcode(
        data: &[Self::GGUF_TYPE],
        n_blocks: usize,
        shape: Shape,
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        let mut ds_bytes = Vec::with_capacity(n_blocks * 4);
        let mut dmins_bytes = Vec::with_capacity(n_blocks * 4);
        let mut scales_bytes = Vec::with_capacity(n_blocks * K_SCALE_SIZE);
        let mut qs_bytes = Vec::with_capacity(n_blocks * QK_K / 2);

        for block in data {
            ds_bytes.extend_from_slice(bytemuck::bytes_of(&block.d.to_f32()));
            dmins_bytes.extend_from_slice(bytemuck::bytes_of(&block.dmin.to_f32()));
            scales_bytes.extend_from_slice(bytemuck::cast_slice(&block.scales));
            qs_bytes.extend_from_slice(bytemuck::cast_slice(&block.qs));
        }

        let _ = ds_bytes.align_standard();
        let _ = dmins_bytes.align_standard();
        let _ = scales_bytes.align_standard();
        let _ = qs_bytes.align_standard();

        ds_bytes.append(&mut dmins_bytes);
        ds_bytes.append(&mut scales_bytes);
        ds_bytes.append(&mut qs_bytes);

        Tensor::from_bytes(
            &ds_bytes,
            DType::GGUF(GGUFDType::Q4K(Q4K)),
            shape,
            Device::CPU,
        )
    }
}

impl GGUFInterop for Q6K {
    type GGUF_TYPE = BlockQ6K;
    const BLCK_NUMEL: usize = QK_K;
    const TYPE_SIZE_WEBGPU: usize = Self::TYPE_SIZE + 2;

    fn transcode(
        data: &[Self::GGUF_TYPE],
        n_blocks: usize,
        shape: Shape,
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        let mut ql_bytes = Vec::with_capacity(n_blocks * QK_K / 2);
        let mut qh_bytes = Vec::with_capacity(n_blocks * QK_K / 4);
        let mut scales_bytes = Vec::with_capacity(n_blocks * QK_K / 16);
        let mut d_bytes = Vec::with_capacity(n_blocks * 4);

        for block in data {
            ql_bytes.extend_from_slice(bytemuck::cast_slice(&block.ql));
            qh_bytes.extend_from_slice(bytemuck::cast_slice(&block.qh));
            scales_bytes.extend_from_slice(bytemuck::cast_slice(&block.scales));
            d_bytes.extend_from_slice(bytemuck::bytes_of(&block.d.to_f32()));
        }

        let _ = ql_bytes.align_standard();
        let _ = qh_bytes.align_standard();
        let _ = scales_bytes.align_standard();
        let _ = d_bytes.align_standard();

        ql_bytes.append(&mut qh_bytes);
        ql_bytes.append(&mut scales_bytes);
        ql_bytes.append(&mut d_bytes);

        Tensor::from_bytes(
            &ql_bytes,
            DType::GGUF(GGUFDType::Q6K(Q6K)),
            shape,
            Device::CPU,
        )
    }
}

impl GGUFInterop for Q8_0 {
    type GGUF_TYPE = BlockQ8_0;
    const BLCK_NUMEL: usize = QK8_0;
    const TYPE_SIZE_WEBGPU: usize = Self::TYPE_SIZE + 2;

    fn transcode(
        data: &[Self::GGUF_TYPE],
        n_blocks: usize,
        shape: Shape,
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        //TODO: these should be uninit
        let mut qs_bytes = Vec::with_capacity(n_blocks * QK8_0);
        let mut ds_bytes = Vec::with_capacity(n_blocks * 4);

        for block in data {
            ds_bytes.extend_from_slice(bytemuck::bytes_of(&block.d.to_f32()));
            qs_bytes.extend_from_slice(bytemuck::cast_slice(&block.qs));
        }

        let _ = ds_bytes.align_standard();
        let _ = qs_bytes.align_standard();

        qs_bytes.append(&mut ds_bytes);
        Tensor::from_bytes(&qs_bytes, DType::WQ8, shape, Device::CPU)
    }
}

impl GGUFInterop for f32 {
    type GGUF_TYPE = f32;
    const BLCK_NUMEL: usize = 1;
    const TYPE_SIZE_WEBGPU: usize = 4;

    fn transcode(
        data: &[Self::GGUF_TYPE],
        n_blocks: usize,
        shape: Shape,
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        Ok(Tensor::from_data(data, shape, device.clone()))
    }
}

impl GGUFInterop for f16 {
    type GGUF_TYPE = f16;

    const BLCK_NUMEL: usize = 1;

    const TYPE_SIZE_WEBGPU: usize = 4;

    fn transcode(
        data: &[Self::GGUF_TYPE],
        n_blocks: usize,
        shape: Shape,
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        let f32_data = data.iter().map(|f| f.to_f32()).collect::<Vec<_>>();
        Ok(Tensor::from_data(f32_data, shape, device.clone()))
    }
}
