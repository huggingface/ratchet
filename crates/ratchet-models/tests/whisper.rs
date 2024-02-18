#![cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

use ndarray::{s, Axis};
use ndarray_stats::QuantileExt;

use ratchet::{shape, Device, DeviceRequest, Tensor};
use ratchet_client::{ApiBuilder, RepoType};
use ratchet_loader::GGMLCompatible;
use ratchet_models::{Whisper, WhisperDecoder, WhisperEncoder};
use ratchet_nn::Module;

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
    let model_repo = ApiBuilder::from_hf("ggerganov/whisper.cpp", RepoType::Model).build();
    let model = model_repo.get("ggml-tiny.bin").await?;
    let model_data = model.to_uint8().await?;

    let ground_repo = ApiBuilder::from_hf("FL33TW00D-HF/ratchet-util", RepoType::Dataset).build();
    let input_npy = ground_repo.get("jfk_tiny_encoder_input.npy").await?;
    let ground_npy = ground_repo.get("jfk_tiny_encoder_output.npy").await?;

    let mut reader = std::io::BufReader::new(std::io::Cursor::new(model_data.to_vec()));
    let gg = Whisper::load_ggml(&mut reader).unwrap();

    let device = Device::request_device(DeviceRequest::GPU).await.unwrap();

    let input_data = &input_npy.to_uint8().await?.to_vec();
    let input = Tensor::from_npy_bytes::<f32>(input_data, &device).unwrap();
    let ground =
        Tensor::from_npy_bytes::<f32>(&ground_npy.to_uint8().await?.to_vec(), &Device::CPU)
            .unwrap();

    let encoder = WhisperEncoder::load(&gg, &mut reader, &device).unwrap();
    let result = encoder.forward(&input).unwrap();
    result.resolve().unwrap();
    let ours = result.to(&Device::CPU).await.unwrap();
    ground.all_close(&ours, 1e-3, 1e-3).unwrap();
    Ok(())
}

#[wasm_bindgen_test]
async fn tiny_decoder() -> Result<(), JsValue> {
    let model_repo = ApiBuilder::from_hf("ggerganov/whisper.cpp", RepoType::Model).build();
    let model = model_repo.get("ggml-tiny.bin").await?;
    let model_data = model.to_uint8().await?;

    let ground_repo = ApiBuilder::from_hf("FL33TW00D-HF/ratchet-util", RepoType::Dataset).build();
    // let input_npy = ground_repo.get("jfk_tiny_encoder_input.npy").await?;
    let ground_npy = ground_repo.get("jfk_tiny_encoder_output.npy").await?;

    let mut reader = std::io::BufReader::new(std::io::Cursor::new(model_data.to_vec()));
    let gg = Whisper::load_ggml(&mut reader).unwrap();

    let device = Device::request_device(DeviceRequest::GPU).await.unwrap();

    // let input_data = &input_npy.to_uint8().await?.to_vec();
    // let input = Tensor::from_npy_bytes::<f32>(input_data, &device).unwrap();
    let ground =
        Tensor::from_npy_bytes::<f32>(&ground_npy.to_uint8().await?.to_vec(), &Device::CPU)
            .unwrap();

    let decoder = WhisperDecoder::load(&gg, &mut reader, &device).unwrap();

    let mut tokens = vec![50258, 50259, 50359];
    let mut token = -1;
    while token != 50257 {
        let token_t =
            Tensor::from_data(tokens.clone(), shape![1, tokens.len()], device.clone());

        let our_ground = match ground.to(&device).await {
            Ok(ground) => ground,
            Err(e) => return Err(JsValue::from_str(&e.to_string())),
        };

        let result = decoder.forward(&[our_ground, token_t]).unwrap();
        result.resolve().unwrap();

        let our_logits = match result.to(&Device::CPU).await {
            Ok(logits) => logits,
            Err(e) => return Err(JsValue::from_str(&e.to_string())),
        };

        let nd_logits = our_logits.to_ndarray_view::<f32>();
        let sliced = nd_logits.slice(s![.., -1.., ..51865]).remove_axis(Axis(1));

        token = sliced
            .map_axis(Axis(1), |row| row.argmax_skipnan().unwrap())
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>()[0];
        console_log!("Token: {}", token);
        tokens.push(token);
    }

    Ok(())
}

