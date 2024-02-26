use std::{collections::HashSet, path::Path};

use ratchet::{Device, Quantization, Quantizer};

use crate::GGMLCompatible;

pub struct Converter;

impl Converter {
    pub fn convert<P: AsRef<Path>, M: GGMLCompatible>(
        src_path: P,
        dst_quant: Quantization,
        to_quant: HashSet<&str>,
    ) -> anyhow::Result<()> {
        let mut reader = std::io::BufReader::new(std::fs::File::open(src_path).unwrap());
        let src = M::load_ggml(&mut reader)?;

        let mut writer = std::io::BufWriter::new(std::fs::File::create("q.bin")?);
        M::write_header(&src.header, &mut writer)?;

        let quantizer = Quantizer::new(dst_quant);

        let mut total_write = 0;
        for (name, _) in &src.tensors {
            let loaded = src.load_tensor(&name, &mut reader, &Device::CPU)?;
            let to_write = if to_quant.iter().any(|suffix| name.ends_with(suffix)) {
                log::info!("Quantizing {}", name);
                quantizer.quantize(loaded)
            } else {
                loaded
            };
            total_write += M::write_tensor(&name, to_write, &mut writer)?;
        }
        log::info!("Total tensor data written: {} bytes", total_write);
        Ok(())
    }
}
