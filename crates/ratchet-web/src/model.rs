#![cfg(target_arch = "wasm32")]
use crate::db::*;
use ratchet_hub::{ApiBuilder, RepoType};
use ratchet_models::{transcribe, DecodingOptions, Whisper};
use wasm_bindgen::prelude::*;

#[derive(Debug)]
pub enum WebModel {
    Whisper(Whisper),
}

impl WebModel {
    pub async fn run(&mut self, input: JsValue) -> Result<JsValue, JsValue> {
        match self {
            WebModel::Whisper(model) => {
                let input: WhisperInputs = serde_wasm_bindgen::from_value(input)?;
                let options = serde_wasm_bindgen::from_value(input.decode_options)?;
                let result = transcribe(model, input.audio, options).await.unwrap();
                serde_wasm_bindgen::to_value(&result).map_err(|e| e.into())
            }
        }
    }

    pub async fn from_stored(stored: StoredModel) -> Result<WebModel, anyhow::Error> {
        match stored.repo_id.as_str() {
            "ggerganov/whisper.cpp" => Ok(WebModel::Whisper(
                Whisper::from_bytes(&stored.bytes.to_vec()).await?,
            )),
            _ => Err(anyhow::anyhow!("Unknown model type")),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct WhisperInputs {
    pub audio: Vec<f32>,
    #[serde(with = "serde_wasm_bindgen::preserve")]
    pub decode_options: JsValue,
}

#[wasm_bindgen]
#[derive(Debug)]
pub struct Model {
    inner: WebModel,
}

#[wasm_bindgen]
impl Model {
    /// The main JS entrypoint into the library.
    ///
    /// Loads a model with the provided ID.
    /// This key should be an enum of supported models.
    #[wasm_bindgen]
    pub async fn load(key: ModelKey) -> Result<Model, JsValue> {
        let model_repo = ApiBuilder::from_hf(&key.repo_id(), RepoType::Model).build();
        let db = RatchetDB::open().await.map_err(|e| {
            let e: JsError = e.into();
            Into::<JsValue>::into(e)
        })?;
        log::warn!("Loading model: {:?}", key);
        if let None = db.get_model(&key).await.map_err(|e| {
            let e: JsError = e.into();
            Into::<JsValue>::into(e)
        })? {
            log::warn!("Model not found in db, fetching from remote");
            let model_data = model_repo.get(&key.model_id()).await?;
            let bytes = model_data.to_uint8().await?;
            let model = StoredModel::new(&key, bytes);
            db.put_model(&key, model).await.unwrap();
        }
        let model = db.get_model(&key).await.unwrap().unwrap();
        Ok(Model {
            inner: WebModel::from_stored(model).await.unwrap(),
        })
    }

    /// User-facing method to run the model.
    ///
    /// Untyped input is required unfortunately.
    pub async fn run(&mut self, input: JsValue) -> Result<JsValue, JsValue> {
        self.inner.run(input).await
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    use super::*;
    use ratchet_hub::{ApiBuilder, RepoType};
    use ratchet_models::DecodingOptionsBuilder;
    use wasm_bindgen_test::*;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    fn log_init() {
        console_error_panic_hook::set_once();
        let logger = fern::Dispatch::new()
            .format(|out, message, record| {
                out.finish(format_args!(
                    "{}[{}][{}] {}",
                    chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"),
                    record.target(),
                    record.level(),
                    message
                ))
            })
            .level_for("tokenizers", log::LevelFilter::Off)
            .level(log::LevelFilter::Info)
            .chain(fern::Output::call(console_log::log))
            .apply();
        match logger {
            Ok(_) => log::info!("Logging initialized."),
            Err(error) => eprintln!("Error initializing logging: {:?}", error),
        }
    }

    fn load_sample(bytes: &[u8]) -> Vec<f32> {
        let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes)).unwrap();
        reader
            .samples::<i16>()
            .map(|x| x.unwrap() as f32 / 32768.0)
            .collect::<Vec<_>>()
    }

    #[wasm_bindgen_test]
    async fn browser_end_to_end() -> Result<(), JsValue> {
        log_init();
        let key = ModelKey::new(
            "ggerganov/whisper.cpp".to_string(),
            "ggml-tiny.bin".to_string(),
        );
        let mut model = Model::load(key).await.unwrap();
        log::warn!("Model: {:?}", model);

        let data_repo = ApiBuilder::from_hf("FL33TW00D-HF/ratchet-util", RepoType::Dataset).build();
        let response = data_repo.get("jfk.wav").await?;
        let audio_bytes = response.to_uint8().await?;
        let sample = load_sample(&audio_bytes.to_vec());

        let decode_options = DecodingOptionsBuilder::default().build();

        let input = WhisperInputs {
            audio: sample,
            decode_options,
        };
        let input = serde_wasm_bindgen::to_value(&input).unwrap();
        let result = model.run(input).await.unwrap();
        log::warn!("Result: {:?}", result);
        Ok(())
    }
}
