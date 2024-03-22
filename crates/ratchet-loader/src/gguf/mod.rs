pub mod ggml;
pub mod gguf;
pub mod k_quants;
pub mod utils;
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};
pub use k_quants::GgmlType;
use std::io::Seek;

#[cfg(test)]
mod tests {

    use std::io::{Cursor, Read, SeekFrom};

    use super::*;
    use ratchet::{Device, DeviceRequest};

    fn read_expected_data<R: std::io::Seek + std::io::Read>(
        reader: &mut R,
        offset: u64,
        length: usize,
    ) -> anyhow::Result<Vec<u8>> {
        let mut expected_data = vec![0u8; length];

        reader.seek(SeekFrom::Start(offset))?;
        reader.read_exact(&mut expected_data)?;
        Ok(expected_data)
    }

    #[tokio::test]
    async fn test_read_q4k() -> anyhow::Result<()> {
        const Q4K_GGUF: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test-data/nano-llama-q4k.gguf"
        ));

        let mut reader = std::io::BufReader::new(std::io::Cursor::new(Q4K_GGUF.to_vec()));
        let device = Device::request_device(DeviceRequest::GPU)?;
        let content = gguf::Content::read(&mut reader)?;

        println!(
            "{:?}",
            content
                .tensor_infos
                .keys()
                .into_iter()
                .filter(|key| (*key).starts_with("blk.0"))
                .map(|key| content.tensor_infos.get(key).map(|ti| (key, ti)))
                .flatten()
                .collect::<Vec<_>>()
        );

        let blk0_k_weight = "blk.0.attn_k.weight";
        let blk0_k_weight_info = content.tensor_infos.get(blk0_k_weight).unwrap();

        let blk0_k_weight_blockq4k = match content.tensor(&mut reader, blk0_k_weight, &device)? {
            k_quants::Block::BlockQ4K(q4k) => q4k,
            k_quants::Block::BlockF32(_) => panic!("Not possible"),
        };

        // let model_data = file.gguf().await?;
        //
        let q4k_len = blk0_k_weight_info.shape.get(0).unwrap() * k_quants::BlockQ4K::TYPE_SIZE;

        let mut expected_q4k_data = vec![0u8; q4k_len];

        reader.seek(SeekFrom::Start(
            content.tensor_data_offset + blk0_k_weight_info.offset,
        ))?;
        reader.read_exact(&mut expected_q4k_data)?;

        let mut actual_q4k_data = vec![0u8; q4k_len];
        let mut actual_q4k_data_cursor = Cursor::new(&mut actual_q4k_data);
        blk0_k_weight_blockq4k.write(&mut actual_q4k_data_cursor)?;

        assert_eq!(
            expected_q4k_data, actual_q4k_data,
            "{:?} not equal",
            blk0_k_weight
        );
        let blk0_norm_weight = "blk.0.attn_norm.weight";
        let blk0_norm_weight_info = content.tensor_infos.get(blk0_norm_weight).unwrap();

        let f32_len = blk0_norm_weight_info.shape.get(0).unwrap() * k_quants::BlockF32::TYPE_SIZE;

        let blk0_norm_weight_blockf32 =
            match content.tensor(&mut reader, blk0_norm_weight, &device)? {
                k_quants::Block::BlockF32(blk_f32) => blk_f32,
                _ => panic!("Not possible"),
            };

        let mut expected_f32_data = vec![0u8; f32_len];

        reader.seek(SeekFrom::Start(
            content.tensor_data_offset + blk0_norm_weight_info.offset,
        ))?;
        reader.read_exact(&mut expected_f32_data)?;

        let mut actual_f32_data = vec![0u8; f32_len];
        let mut actual_f32_data_cursor = Cursor::new(&mut actual_f32_data);
        blk0_norm_weight_blockf32.write(&mut actual_f32_data_cursor)?;

        assert_eq!(
            expected_f32_data, actual_f32_data,
            "{:?} not equal",
            blk0_norm_weight
        );

        Ok(())
    }
}
