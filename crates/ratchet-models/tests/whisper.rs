#![cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

use ndarray::{s, Axis};
use ndarray_stats::QuantileExt;
use ratchet::{shape, Device, DeviceRequest, Tensor};
use ratchet_hub::{ApiBuilder, RepoType};
use ratchet_loader::GGMLCompatible;
use ratchet_models::{DecodingOptionsBuilder, Whisper, WhisperDecoder, WhisperEncoder};
use ratchet_nn::Module;
use std::path::PathBuf;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

fn log_init() {
    console_error_panic_hook::set_once();
    log::set_max_level(log::LevelFilter::Off);
    console_log::init_with_level(log::Level::Warn).unwrap();
}

#[wasm_bindgen_test]
async fn tiny_encoder() -> Result<(), JsValue> {
    log_init();
    let model_repo = ApiBuilder::from_hf("FL33TW00D-HF/whisper-tiny", RepoType::Model).build();
    let model_data = model_repo.get("tiny_f32.bin").await?;

    let ground_repo = ApiBuilder::from_hf("FL33TW00D-HF/ratchet-util", RepoType::Dataset).build();
    let input_npy = ground_repo.get("jfk_tiny_encoder_input.npy").await?;
    let ground_npy = ground_repo.get("jfk_tiny_encoder_hs.npy").await?;

    let mut reader = std::io::BufReader::new(std::io::Cursor::new(model_data.to_vec()));
    let gg = Whisper::load_ggml(&mut reader).unwrap();

    let device = Device::request_device(DeviceRequest::GPU).await.unwrap();

    let input_data = &input_npy.to_vec();
    let input = Tensor::from_npy_bytes::<f32>(input_data, &device).unwrap();
    let ground = Tensor::from_npy_bytes::<f32>(&ground_npy.to_vec(), &Device::CPU).unwrap();

    let encoder = WhisperEncoder::load(&gg, &mut reader, &device).unwrap();
    let result = encoder.forward(input).unwrap().resolve().unwrap();
    let ours = result.to(&Device::CPU).await.unwrap();
    ground.all_close(&ours, 1e-3, 1e-3).unwrap();
    Ok(())
}

#[wasm_bindgen_test]
async fn tiny_decoder() -> Result<(), JsValue> {
    let model_repo = ApiBuilder::from_hf("FL33TW00D-HF/whisper-tiny", RepoType::Model).build();
    let model_data = model_repo.get("tiny_f32.bin").await?;

    let ground_repo = ApiBuilder::from_hf("FL33TW00D-HF/ratchet-util", RepoType::Dataset).build();
    let hs_data = ground_repo.get("jfk_tiny_encoder_hs.npy").await?;

    let mut reader = std::io::BufReader::new(std::io::Cursor::new(model_data.to_vec()));
    let gg_disk = Whisper::load_ggml(&mut reader).unwrap();
    assert_eq!(gg_disk.tensors.len(), 167);

    let device = Device::request_device(DeviceRequest::GPU).await.unwrap();
    let audio_ctx = Tensor::from_npy_bytes::<f32>(&hs_data.to_vec(), &device).unwrap();
    let mut decoder = WhisperDecoder::load(&gg_disk, &mut reader, &device).unwrap();

    let mut tokens = vec![50258, 50259, 50359];
    let mut all_tokens = tokens.clone();
    let mut all_logits = vec![];
    while tokens[tokens.len() - 1] != 50257 {
        let token_t = Tensor::from_data(tokens.clone(), shape![1, tokens.len()], device.clone());
        let result = decoder
            .forward([audio_ctx.clone(), token_t])
            .unwrap()
            .resolve()
            .unwrap();

        let our_logits = result.to(&Device::CPU).await.unwrap();
        all_logits.push(our_logits.clone());
        let nd_logits = our_logits.to_ndarray_view::<f32>();
        let sliced = nd_logits.slice(s![.., -1.., ..51865]).remove_axis(Axis(1));
        decoder.cache_mut().update(tokens.len());

        tokens = sliced
            .map_axis(Axis(1), |row| row.argmax_skipnan().unwrap())
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        println!("Token: {:?}", tokens);
        all_tokens.extend(tokens.clone());
    }

    let ground_tokens = vec![
        50258, 50259, 50359, 50363, 400, 370, 452, 7177, 6280, 1029, 406, 437, 428, 1941, 393, 360,
        337, 291, 1029, 437, 291, 393, 360, 337, 428, 1941, 13, 50257,
    ];
    assert_eq!(all_tokens, ground_tokens);
    Ok(())
}
