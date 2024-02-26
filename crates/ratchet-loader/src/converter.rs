use std::{collections::HashSet, path::Path};

use ratchet::{Device, Quantization, Quantizer};

use crate::GGMLCompatible;

pub struct Converter;

impl Converter {
    pub fn convert<P: AsRef<Path>, M: GGMLCompatible>(
        src_path: P,
        dst_quant: Quantization,
        to_quant: HashSet<String>,
    ) -> anyhow::Result<()> {
        let mut reader = std::io::BufReader::new(std::fs::File::open(src_path).unwrap());
        let src = M::load_ggml(&mut reader)?;

        let mut writer = std::io::BufWriter::new(std::fs::File::create("q.bin")?);
        M::write_header(&src.header, &mut writer)?;

        let quantizer = Quantizer::new(dst_quant);

        for (name, _) in &src.tensors {
            let loaded = src.load_tensor(&name, &mut reader, &Device::CPU)?;
            let to_write = if to_quant.contains(name) {
                quantizer.quantize(loaded)
            } else {
                loaded
            };
            M::write_tensor(&name, to_write, &mut writer)?;
        }

        Ok(())
    }
}
