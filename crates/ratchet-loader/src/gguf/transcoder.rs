use half::f16;
use ratchet::{DType, Device, Quantization, Quantizer, Shape, Tensor};

use super::ggml::GgmlDType;

/// # Transcoder
/// Transcode GGML quantized types into Ratchet quantized types.
///
/// Can be done on the fly (but perhaps shouldn't be) for transparent compatibility.
pub struct GGTranscoder {}

impl GGTranscoder {
    pub fn transcode(
        src_dtype: GgmlDType,
        dst_dtype: DType,
        raw_data: &[u8],
        shape: Shape,
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        log::info!("Transcoding from {:?} to {:?}", src_dtype, dst_dtype);
        match (src_dtype, dst_dtype) {
            (GgmlDType::F32, DType::F32) => {
                Tensor::from_bytes(raw_data, dst_dtype, shape, device.clone())
            }
            (GgmlDType::F16, DType::F32) => {
                //Cast whilst WGPU doesn't support f16
                let f16_data = bytemuck::cast_slice::<u8, f16>(raw_data);
                let f32_data = f16_data.iter().map(|f| f.to_f32()).collect::<Vec<_>>();
                Ok(Tensor::from_data(f32_data, shape, device.clone()))
            }
            (GgmlDType::F32, DType::WQ8) => {
                let quantizer = Quantizer::new(Quantization::SInt8);
                let unquant = Tensor::from_bytes(raw_data, dst_dtype, shape, device.clone())?;
                Ok(quantizer.quantize(unquant))
            }
            _ => todo!(),
        }
    }
}
