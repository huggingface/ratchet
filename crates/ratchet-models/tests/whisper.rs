#![cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

use ratchet::{shape, Device, DeviceRequest, Tensor};
use ratchet_client::ApiBuilder;
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
async fn chrome_tiny_encoder() -> Result<(), JsValue> {
    log_init();
    let model_repo = ApiBuilder::from_hf("ggerganov/whisper.cpp").build();
    let model = model_repo.get("ggml-tiny.bin").await?;
    let uint8array = model.to_uint8().await?;

    let mut reader = std::io::BufReader::new(std::io::Cursor::new(uint8array.to_vec()));
    let gg = Whisper::load_ggml(&mut reader).unwrap();
    log::error!("Loaded Whisper to GG");

    let device = Device::request_device(DeviceRequest::GPU).await.unwrap();
    log::error!("Got device");
    let encoder = WhisperEncoder::load(&gg, &mut reader, &device).unwrap();
    log::error!("Loaded WhisperEncoder");
    let input = Tensor::randn::<f32>(shape![1, 80, 3000], device);
    log::error!("Generated input");
    let result = encoder.forward(&input).unwrap();
    log::error!("Forwarded input");
    result.resolve().unwrap();
    log::error!("Resolved result");
    let ours = result.to(&Device::CPU).await;
    log::error!("OURS: {:?}", ours);
    log::warn!("OURS: {:?}", ours);
    Ok(())
}
