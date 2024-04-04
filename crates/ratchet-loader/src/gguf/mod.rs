pub mod ggml;
pub mod gguf;
pub mod tensor_loader;
pub mod transcoder;
pub mod utils;
use ratchet::gguf::*;
pub use tensor_loader::TensorLoader;

#[cfg(test)]
mod tests {

    use std::io::{Cursor, Read, SeekFrom};

    use super::*;
    use ratchet::{Device, DeviceRequest, Tensor};

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

    fn read_actual_data<GGUF: TensorLoader>(
        tensor: Tensor,
        length: usize,
    ) -> anyhow::Result<Vec<u8>> {
        let mut actual_data = vec![0u8; length];
        let mut actual_data_cursor = Cursor::new(&mut actual_data);
        GGUF::write(tensor, &mut actual_data_cursor)?;

        Ok(actual_data)
    }

    #[tokio::test]
    async fn test_read_q4k() -> anyhow::Result<()> {
        const Q4K_GGUF: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test-data/nano-llama-q4k.gguf"
        ));

        let mut reader = std::io::BufReader::new(std::io::Cursor::new(Q4K_GGUF.to_vec()));
        let device = Device::request_device(DeviceRequest::CPU)?;
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

        for (tensor_name, tensor_info) in tensor_infos {
            let tensor = content.tensor(&mut reader, tensor_name, &device)?;

            let (type_size, block_size) = match tensor.dt() {
                ratchet::DType::F32 => Ok((f32::TYPE_SIZE, f32::BLCK_SIZE)),
                ratchet::DType::GGUF(GGUFDType::Q4K(_)) => Ok((Q4K::TYPE_SIZE, Q4K::BLCK_SIZE)),
                ratchet::DType::GGUF(GGUFDType::Q6K(_)) => Ok((Q6K::TYPE_SIZE, Q6K::BLCK_SIZE)),
                _ => Err(anyhow::anyhow!("Invalid dtype")),
            }?;

            let tensor_blocks = tensor_info.shape.numel() / block_size;
            let length_in_bytes = tensor_blocks * type_size;

            let expected_data = read_expected_data(
                &mut reader,
                content.tensor_data_offset + tensor_info.offset,
                length_in_bytes,
            )?;

            let actual_data = match tensor.dt() {
                ratchet::DType::F32 => read_actual_data::<f32>(tensor, length_in_bytes),
                ratchet::DType::GGUF(GGUFDType::Q4K(_)) => {
                    read_actual_data::<Q4K>(tensor, length_in_bytes)
                }
                ratchet::DType::GGUF(GGUFDType::Q6K(_)) => {
                    read_actual_data::<Q6K>(tensor, length_in_bytes)
                }
                _ => Err(anyhow::anyhow!("Invalid dtype")),
            }?;
            println!(
                "Checking {:?}: {:?}",
                tensor_name,
                expected_data.eq(&actual_data)
            );

            assert_eq!(expected_data, actual_data, "{:?} not equal", tensor_name);
        }
        Ok(())
    }

    #[tokio::test]
    #[ignore = "todo"]
    async fn test_read_f16() -> anyhow::Result<()> {
        let model_path = concat!(
            env!("CARGO_RUSTC_CURRENT_DIR"),
            "/models/microsoft/phi-2/phi2-f16.gguf"
        );
        let mut reader = std::io::BufReader::new(std::fs::File::open(model_path)?);
        let device = Device::request_device(DeviceRequest::GPU)?;
        let content = gguf::Content::read(&mut reader)?;

        let first_blk = content
            .tensor_infos
            .keys()
            .into_iter()
            .filter(|key| (*key).starts_with("blk.0"))
            .map(|key| content.tensor_infos.get(key).map(|ti| (key, ti)))
            .flatten()
            .collect::<Vec<_>>();

        for b in first_blk {
            println!("{:?}", b);
        }

        let tensor_infos = content
            .tensor_infos
            .keys()
            .into_iter()
            .filter(|key| (*key).starts_with("blk.0"))
            .map(|key| content.tensor_infos.get(key).map(|ti| (key, ti)))
            .flatten()
            .collect::<Vec<_>>();

        tensor_infos.iter().for_each(|(key, ti)| {
            println!("{:?} {:?}", key, ti);
        });

        let zeroth = tensor_infos[0];
        let tensor = content.tensor(&mut reader, zeroth.0, &device);
        println!("{:?}", tensor);

        Ok(())
    }
}
