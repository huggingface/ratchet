#![cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

use ratchet::{Device, DeviceRequest, Tensor};
use ratchet_client::{ApiBuilder, RepoType};
use ratchet_loader::GGMLCompatible;
use ratchet_models::{Whisper, WhisperEncoder};
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
    let ground_npy = ground_repo.get("jfk_tiny_encoder_hs.npy").await?;

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
