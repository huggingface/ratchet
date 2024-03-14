pub mod ggml;
pub mod gguf;
pub mod k_quants;
pub mod utils;

#[cfg(test)]
mod tests {
    use super::*;
    use ratchet::{Device, DeviceRequest};

    #[tokio::test]
    async fn test_read_q4k() -> anyhow::Result<()> {
        const Q4K_GGUF: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test-data/nano-llama-q4k.gguf"
        ));

        let mut reader = std::io::BufReader::new(std::io::Cursor::new(Q4K_GGUF.to_vec()));
        let device = Device::request_device(DeviceRequest::GPU)?;
        let result = gguf::Content::read(&mut reader)?;

        // println!("{:?}", result);

        let q4k_block = result.tensor(&mut reader, "blk.0.attn_k.weight", &device)?;
        // let model_data = file.gguf().await?;

        println!("{:?}", q4k_block);
        Ok(())
    }

    // #[tokio::test]
    // async fn test_gguf() -> anyhow::Result<()> {
    //     const DUMMY_GGUF: &[u8] =
    //         include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/test-data/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"));

    //     let mut reader = std::io::BufReader::new(std::io::Cursor::new(DUMMY_GGUF.to_vec()));

    //     let result = gguf::Content::read(&mut reader)?;

    //     let empty_value = gguf::Value::String(String::from(""));
    //     let metadata = result
    //         .metadata
    //         .keys()
    //         .filter(|key| !(*key).starts_with("tokenizer"))
    //         .map(|key| {
    //             (
    //                 key,
    //                 result.metadata.get(key).unwrap_or_else(|| &empty_value),
    //             )
    //         })
    //         .collect::<Vec<_>>();

    //     println!("{:?}", metadata);

    //     println!(
    //         "{:?}",
    //         result
    //             .tensor_infos
    //             .keys()
    //             .filter(|key| (*key).starts_with("blk.0"))
    //             .collect::<Vec<_>>()
    //     );

    //     let device = Device::request_device(DeviceRequest::GPU)?;
    //     // let tensor = result.tensor(&mut reader, "blk.0.ffn_norm.weight", &device);
    //     let tensor = result.tensor(&mut reader, "blk.0.attn_k.weight", &device);
    //     // // let model_data = file.gguf().await?;

    //     // println!("{:?}", tensor);
    //     Ok(())
    // }
}
