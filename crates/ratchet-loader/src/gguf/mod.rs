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

    use self::k_quants::Block;

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

    fn read_actual_data<GGUF: GgmlType>(block: &GGUF, length: usize) -> anyhow::Result<Vec<u8>> {
        let mut actual_data = vec![0u8; length];
        let mut actual_f32_data_cursor = Cursor::new(&mut actual_data);
        block.write(&mut actual_f32_data_cursor)?;

        Ok(actual_data)
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

        let tensor_infos = content
            .tensor_infos
            .keys()
            .into_iter()
            .filter(|key| (*key).starts_with("blk.0"))
            .map(|key| content.tensor_infos.get(key).map(|ti| (key, ti)))
            .flatten()
            .collect::<Vec<_>>();
        // for (tensor_name, tensor_info) in tensor_infos {
        //     let block = content.tensor(&mut reader, tensor_name, &device)?;

        //     let block_length = block.type_size();
        //     let length = tensor_info.shape.get(0).unwrap() * block_length;

        //     let expected_data = read_expected_data(
        //         &mut reader,
        //         content.tensor_data_offset + tensor_info.offset,
        //         length,
        //     )?;

        //     let actual_data = match block {
        //         Block::BlockQ4K(blk) => read_actual_data(&blk, length),
        //         Block::BlockF32(blk) => read_actual_data(&blk, length),
        //         Block::BlockQ6K(blk) => read_actual_data(&blk, length),
        //     }?;
        //     println!("Checking {:?}", tensor_name);

        //     assert_eq!(expected_data, actual_data, "{:?} not equal", tensor_name);
        // }

        let blk0_k_weight = "blk.0.attn_k.weight";
        let blk0_k_weight_info = content.tensor_infos.get(blk0_k_weight).unwrap();

        let blk0_k_weight_blockq4k = match content.tensor(&mut reader, blk0_k_weight, &device)? {
            k_quants::Block::BlockQ4K(q4k) => q4k,
            _ => panic!("Not possible"),
        };

        // let model_data = file.gguf().await?;
        //
        let q4k_len = blk0_k_weight_info.shape.get(0).unwrap() * k_quants::BlockQ4K::TYPE_SIZE;

        let expected_q4k_data = read_expected_data(
            &mut reader,
            content.tensor_data_offset + blk0_k_weight_info.offset,
            q4k_len,
        )?;

        let actual_q4k_data = read_actual_data(&blk0_k_weight_blockq4k, q4k_len)?;

        // assert_eq!(
        //     expected_q4k_data, actual_q4k_data,
        //     "{:?} not equal",
        //     blk0_k_weight
        // );
        // let blk0_norm_weight = "blk.0.attn_norm.weight";
        // let blk0_norm_weight_info = content.tensor_infos.get(blk0_norm_weight).unwrap();

        // let f32_len = blk0_norm_weight_info.shape.get(0).unwrap() * k_quants::BlockF32::TYPE_SIZE;

        // let blk0_norm_weight_blockf32 =
        //     match content.tensor(&mut reader, blk0_norm_weight, &device)? {
        //         k_quants::Block::BlockF32(blk_f32) => blk_f32,
        //         _ => panic!("Not possible"),
        //     };

        // let expected_f32_data = read_expected_data(
        //     &mut reader,
        //     content.tensor_data_offset + blk0_norm_weight_info.offset,
        //     f32_len,
        // )?;

        // let actual_f32_data = read_actual_data(&blk0_norm_weight_blockf32, f32_len)?;

        // assert_eq!(
        //     expected_f32_data, actual_f32_data,
        //     "{:?} not equal",
        //     blk0_norm_weight
        // );

        // let blk0_v_weight = "blk.0.attn_v.weight";
        // let blk0_v_weight_info = content.tensor_infos.get(blk0_v_weight).unwrap();

        // let q6k_len = blk0_v_weight_info.shape.get(0).unwrap() * k_quants::BlockQ6K::TYPE_SIZE;

        // let blk0_v_weight_block_q6k = match content.tensor(&mut reader, blk0_v_weight, &device)? {
        //     k_quants::Block::BlockQ6K(blk_q6k) => blk_q6k,
        //     _ => panic!("Not possible"),
        // };

        // let expected_q6k_data = read_expected_data(
        //     &mut reader,
        //     content.tensor_data_offset + blk0_v_weight_info.offset,
        //     q6k_len,
        // )?;

        // let actual_q6k_data = read_actual_data(&blk0_v_weight_block_q6k, q6k_len)?;

        // assert_eq!(
        //     expected_q6k_data, actual_q6k_data,
        //     "{:?} not equal",
        //     blk0_v_weight
        // );

        Ok(())
    }
}
