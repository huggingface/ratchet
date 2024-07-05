#![cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

use ndarray::{s, Axis};
use ndarray_stats::QuantileExt;
use ratchet::{shape, Device, DeviceRequest, Tensor};
use ratchet_hub::{ApiBuilder, RepoType};
use ratchet_loader::gguf::gguf;
use ratchet_models::whisper::{Config, Whisper, WhisperDecoder, WhisperEncoder};
use ratchet_nn::Module;
use std::path::PathBuf;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

fn log_init() {
    console_error_panic_hook::set_once();
    log::set_max_level(log::LevelFilter::Off);
    console_log::init_with_level(log::Level::Debug).unwrap();
}

/*
#[wasm_bindgen_test]
async fn distil_large_v3_encoder() -> Result<(), JsValue> {
    log_init();
    let model_repo =
        ApiBuilder::from_hf("FL33TW00D-HF/distil-whisper-large-v3", RepoType::Model).build();
    let model_data = model_repo.get("distil-large-v3_q8_0.gguf").await?;
    let config_data = model_repo.get("config.json").await?;

    let ground_repo = ApiBuilder::from_hf("FL33TW00D-HF/ratchet-util", RepoType::Dataset).build();
    let input_npy = ground_repo.get("distil_large_v3_q80_mm0_input.npy").await?;
    let ground_npy = ground_repo.get("distil_large_v3_q80_mm0_hs.npy").await?;

    let mut reader = std::io::BufReader::new(std::io::Cursor::new(model_data.to_vec()));
    let header = gguf::Header::read(&mut reader).unwrap();
    let config: Config = serde_json::from_slice(&config_data.to_vec()).unwrap();

    let device = Device::request_device(DeviceRequest::GPU).await.unwrap();

    let input_data = &input_npy.to_vec();
    let input = Tensor::from_npy_bytes::<f32>(input_data, &device).unwrap();
    let ground = Tensor::from_npy_bytes::<f32>(&ground_npy.to_vec(), &Device::CPU).unwrap();

    let encoder = WhisperEncoder::load(&header, &config, &mut reader, &device).unwrap();
    let result = encoder
        .schedule(input)
        .unwrap()
        .full()
        .unwrap()
        .resolve()
        .unwrap();
    let ours = result.to(&Device::CPU).await.unwrap();
    ground.all_close(&ours, 1e-3, 1e-3).unwrap();
    Ok(())
}*/

/*
#[wasm_bindgen_test]
async fn distil_large_v3_decoder() -> Result<(), JsValue> {
    log_init();
    let model_repo =
        ApiBuilder::from_hf("FL33TW00D-HF/distil-whisper-large-v3", RepoType::Model).build();
    let model_data = model_repo.get("distil-large-v3_q8_0.gguf").await?;
    let config_data = model_repo.get("config.json").await?;

    let ground_repo = ApiBuilder::from_hf("FL33TW00D-HF/ratchet-util", RepoType::Dataset).build();
    let hs_data = ground_repo.get("distil_large_v3_q80_mm0_hs.npy").await?;

    let mut reader = std::io::BufReader::new(std::io::Cursor::new(model_data.to_vec()));
    let header = gguf::Header::read(&mut reader).unwrap();
    let config: Config = serde_json::from_slice(&config_data.to_vec()).unwrap();

    let device = Device::request_device(DeviceRequest::GPU).await.unwrap();
    let audio_ctx = Tensor::from_npy_bytes::<f32>(&hs_data.to_vec(), &device)
        .unwrap()
        .half()
        .unwrap()
        .resolve()
        .unwrap();
    log::debug!("Audio Context: {:?}", audio_ctx);
    let mut decoder = WhisperDecoder::load(&header, &config, &mut reader, &device).unwrap();

    let mut tokens = vec![50258, 50259, 50360];
    let mut all_tokens = tokens.clone();
    let mut all_logits = vec![];
    while tokens[tokens.len() - 1] != 50257 {
        let token_t = Tensor::from_data(tokens.clone(), shape![1, tokens.len()], device.clone());
        let result = decoder
            .schedule([audio_ctx.clone(), token_t])
            .unwrap()
            .resolve_debug()
            .unwrap();

        let our_logits = result.to(&Device::CPU).await.unwrap();
        all_logits.push(our_logits.clone());
        let nd_logits = our_logits.to_ndarray_view::<f32>();
        log::info!("Logits: {:?}", nd_logits);

        let sliced = nd_logits.slice(s![.., -1.., ..51866]).remove_axis(Axis(1));
        decoder.cache_mut().update(tokens.len());

        tokens = sliced
            .map_axis(Axis(1), |row| row.argmax_skipnan().unwrap())
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        println!("Token: {:?}", tokens);
        panic!();
        all_tokens.extend(tokens.clone());
    }

    let ground_tokens = vec![
        50258, 50259, 50360, 50365, 639, 307, 264, 4532, 3479, 587, 11, 15578, 264, 881, 2062, 847,
        11, 34674, 5932, 30340, 295, 3123, 4397, 608, 1652, 13, 50517, 50530, 6947, 472, 575,
        12023, 4365, 11, 20899, 11, 10445, 11, 18356, 11, 4225, 4782, 11, 50624, 50626, 1804, 4651,
        3123, 4397, 34922, 8963, 862, 6352, 13, 50695, 50701, 821, 311, 257, 3804, 5214, 11, 2610,
        5214, 11, 6383, 11, 2643, 5214, 11, 293, 544, 13, 50797, 50807, 10246, 8963, 2436, 2965,
        281, 747, 604, 1081, 13, 50875, 50881, 400, 456, 366, 867, 34674, 862, 365, 11, 293, 1184,
        472, 1487, 365, 1080, 1065, 2121, 11377, 11, 51002, 51007, 4532, 3479, 5864, 293, 1019, 11,
        5456, 4122, 300, 30686, 25038, 1286, 13, 51120, 51135, 30062, 264, 6582, 29814, 412, 264,
        10155, 11, 1849, 1426, 11, 587, 11, 264, 3874, 34544, 412, 264, 7267, 3096, 13, 51243,
        51246, 18463, 428, 1032, 412, 264, 1032, 5675, 13, 51287, 51290, 30062, 264, 16629, 7283,
        13, 51320, 51328, 400, 613, 862, 6352, 3318, 1214, 281, 1254, 257, 3123, 4397, 34922, 8963,
        1081, 6352, 11, 370, 27985, 11, 51504, 51504, 370, 6239, 11, 370, 44078, 1688, 356, 9942,
        11, 291, 603, 528, 281, 8963, 552, 439, 13, 51623, 51635, 25642, 12089, 1652, 366, 1027,
        14759, 490, 7336, 836, 65, 13, 51743, 51743, 440, 4356, 436, 366, 11, 264, 1101, 436, 366,
        13, 51834,
    ];
    assert_eq!(all_tokens, ground_tokens);
    Ok(())
}*/

/*
#[wasm_bindgen_test]
async fn tiny_encoder() -> Result<(), JsValue> {
    log_init();
    let model_repo = ApiBuilder::from_hf("FL33TW00D-HF/whisper-tiny", RepoType::Model).build();
    let model_data = model_repo.get("tiny_f32.gguf").await?;
    let config_data = model_repo.get("config.json").await?;

    let ground_repo = ApiBuilder::from_hf("FL33TW00D-HF/ratchet-util", RepoType::Dataset).build();
    let input_npy = ground_repo.get("jfk_tiny_encoder_input.npy").await?;
    let ground_npy = ground_repo.get("jfk_tiny_encoder_hs.npy").await?;

    let mut reader = std::io::BufReader::new(std::io::Cursor::new(model_data.to_vec()));
    let header = gguf::Header::read(&mut reader).unwrap();
    let config: Config = serde_json::from_slice(&config_data.to_vec()).unwrap();

    let device = Device::request_device(DeviceRequest::GPU).await.unwrap();

    let input_data = &input_npy.to_vec();
    let input = Tensor::from_npy_bytes::<f32>(input_data, &device).unwrap();
    let ground = Tensor::from_npy_bytes::<f32>(&ground_npy.to_vec(), &Device::CPU).unwrap();

    let encoder = WhisperEncoder::load(&header, &config, &mut reader, &device).unwrap();
    let result = encoder.schedule(input).unwrap().resolve().unwrap();
    let ours = result.to(&Device::CPU).await.unwrap();
    ground.all_close(&ours, 1e-3, 1e-3).unwrap();
    Ok(())
}
*/

#[wasm_bindgen_test]
async fn tiny_decoder() -> Result<(), JsValue> {
    log_init();
    let model_repo =
        ApiBuilder::from_hf("FL33TW00D-HF/distil-whisper-large-v3", RepoType::Model).build();
    let model_data = model_repo.get("distil-large-v3_q8_0.gguf").await?;
    let config_data = model_repo.get("config.json").await?;

    let ground_repo = ApiBuilder::from_hf("FL33TW00D-HF/ratchet-util", RepoType::Dataset).build();
    let hs_data = ground_repo.get("distil_large_v3_q80_mm0_hs.npy").await?;

    let mut reader = std::io::BufReader::new(std::io::Cursor::new(model_data.to_vec()));
    let header = gguf::Header::read(&mut reader).unwrap();
    let config: Config = serde_json::from_slice(&config_data.to_vec()).unwrap();

    let device = Device::request_device(DeviceRequest::GPU).await.unwrap();

    let audio_ctx_cpu = Tensor::from_npy_bytes::<f32>(&hs_data.to_vec(), &Device::CPU).unwrap();
    log::debug!("Audio Context: {:?}", audio_ctx_cpu);

    let audio_ctx = Tensor::from_npy_bytes::<f32>(&hs_data.to_vec(), &device)
        .unwrap()
        .cast(device.compute_precision())
        .unwrap();
    let mut decoder = WhisperDecoder::load(&header, &config, &mut reader, &device).unwrap();

    let mut tokens = vec![50258, 50259, 50360];
    let mut all_tokens = tokens.clone();
    let mut all_logits = vec![];
    let vocab_size = 51866;
    while tokens[tokens.len() - 1] != 50257 {
        let token_t = Tensor::from_data(tokens.clone(), shape![1, tokens.len()], device.clone());
        let result = decoder
            .schedule([audio_ctx.clone(), token_t])
            .unwrap()
            .resolve()
            .unwrap();

        let our_logits = result.to(&Device::CPU).await.unwrap();
        all_logits.push(our_logits.clone());
        let nd_logits = our_logits.to_ndarray_view::<f32>();
        log::debug!("ND LOGITS: {:?}", nd_logits);
        let sliced = nd_logits
            .slice(s![.., -1.., ..vocab_size])
            .remove_axis(Axis(1));
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
