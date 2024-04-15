use half::f16;
use ratchet::{DType, Device, Shape, Tensor};

use crate::{
    gguf::tensor_loader::Padding,
    k_quants::{BlockQ8_0, GgmlType, QK8_0},
};
use anyhow::Context;

/// # Transcoder
/// Transcode GGML quantized types into Ratchet quantized types.
///
/// Can be done on the fly (but perhaps shouldn't be) for transparent compatibility.
pub struct GGTranscoder {}

impl GGTranscoder {
    pub fn transcode<G: GgmlType + Send + Sync + 'static>(
        data: &[G],
        n_blocks: usize,
        shape: Shape,
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        match G::DTYPE {
            crate::GgmlDType::F32 => {
                //It's just F32 dog
                let data: &[f32] = unsafe { std::mem::transmute(data) };
                Ok(Tensor::from_data(data, shape, device.clone()))
            }
            crate::GgmlDType::F16 => {
                //Cast whilst WGPU doesn't support f16
                let data: &[f16] = unsafe { std::mem::transmute(data) };
                let f32_data = data.iter().map(|f| f.to_f32()).collect::<Vec<_>>();
                Ok(Tensor::from_data(f32_data, shape, device.clone()))
            }
            crate::GgmlDType::Q8_0 => {
                let data: &[BlockQ8_0] = unsafe { std::mem::transmute(data) };

                let mut qs_bytes = Vec::with_capacity(n_blocks * QK8_0);
                let mut ds_bytes = Vec::with_capacity(n_blocks * 4);

                for b in data {
                    ds_bytes.extend_from_slice(bytemuck::bytes_of(&b.d.to_f32()));
                    qs_bytes.extend_from_slice(bytemuck::cast_slice(&b.qs));
                }

                let _ = ds_bytes.align_standard();
                let _ = qs_bytes.align_standard();

                qs_bytes.append(&mut ds_bytes);

                println!(
                    "Creating WQ8 tensor of shape: {:?} with num bytes: {}",
                    shape,
                    qs_bytes.len()
                );
                Tensor::from_bytes(&qs_bytes, DType::WQ8, shape, device.clone())
            }
            _ => todo!(),
        }
    }
}
