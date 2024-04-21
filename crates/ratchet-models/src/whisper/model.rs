use ratchet::{shape, Device, Tensor};
use ratchet_loader::gguf::gguf::Header;
use ratchet_nn::Module;
use std::io::{BufRead, Seek};

use ndarray::{s, Dimension};
use ndarray_stats::QuantileExt;
use ratchet::NDArrayExt;

#[cfg(not(target_arch = "wasm32"))]
use hf_hub::api::sync::Api;

#[cfg(target_arch = "wasm32")]
use {ratchet_hub::ApiBuilder, ratchet_hub::RepoType, wasm_bindgen::JsError};

use crate::registry::WhisperVariants;
use crate::whisper::{options::Language, task::DecodingTask, tokenizer::WhisperTokenizer};

use super::encoder::WhisperEncoder;
use super::spectrogram::SpectrogramGenerator;
use super::{config::Config, decoder::WhisperDecoder};

#[derive(Debug)]
pub struct Whisper {
    pub specgen: SpectrogramGenerator,
    pub encoder: WhisperEncoder,
    pub decoder: WhisperDecoder,
    pub config: Config,
    pub device: Device,
}

impl Whisper {
    pub fn load<R: BufRead + Seek>(
        header: Header,
        reader: &mut R,
        device: Device,
    ) -> anyhow::Result<Self> {
        let mel_bytes = Self::fetch_resource(WhisperVariants::Tiny, "melfilters.bytes")?;
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
            &mel_bytes,
            &mut mel_filters,
        );
        let specgen = SpectrogramGenerator::new(mel_filters);

        let config: Config =
            serde_json::from_slice(&Self::fetch_resource(WhisperVariants::Tiny, "config.json")?)?;
        let encoder = WhisperEncoder::load(&header, &config, reader, &device)?;
        let decoder = WhisperDecoder::load(&header, &config, reader, &device)?;

        Ok(Self {
            specgen,
            encoder,
            decoder,
            config,
            device,
        })
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn fetch_resource(
        variant: WhisperVariants,
        resource: &str,
    ) -> Result<Vec<u8>, JsError> {
        let repo_id = variant.repo_id();
        let model_repo = ApiBuilder::from_hf(repo_id, RepoType::Model).build();
        model_repo.get(resource).await?
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn fetch_resource(variant: WhisperVariants, resource: &str) -> anyhow::Result<Vec<u8>> {
        let api = Api::new().unwrap();
        let repo_id = variant.repo_id();

        let repo = api.model(repo_id.to_string());
        Ok(std::fs::read(repo.get(resource).unwrap())?)
    }
}

impl Whisper {
    pub fn is_multilingual(&self) -> bool {
        self.config.n_vocab >= 51865
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn detect_language(&mut self, mel: Tensor) -> anyhow::Result<Language> {
        let audio_ctx = self.encoder.schedule(mel)?.resolve()?;
        let sot = Tensor::from_data([WhisperTokenizer::SOT], shape![1, 1], self.device.clone());

        let logits = self.decoder.schedule([audio_ctx, sot])?.resolve()?;
        self.decoder.reset();

        let cpu_logits = logits.to(&Device::CPU)?;
        let logits = DecodingTask::slice_logits(cpu_logits, self.config.n_vocab as usize);

        let device = logits.device().clone();
        let mut nd_logits = logits.into_ndarray::<f32>();

        let languages_end = if self.config.n_vocab == 51865 {
            50358
        } else if self.config.n_vocab == 51866 {
            50359
        } else {
            panic!("Unsupported number of tokens")
        };

        nd_logits
            .slice_mut(s![.., ..WhisperTokenizer::LANGUAGES_BEGIN])
            .map_inplace(move |el| *el = f32::NEG_INFINITY);

        nd_logits
            .slice_mut(s![.., languages_end..])
            .map_inplace(move |el| *el = f32::NEG_INFINITY);

        let language_tokens_probs = nd_logits.softmax(nd_logits.ndim() - 1);

        let argmax_dims = language_tokens_probs.argmax_skipnan().unwrap();
        let argmax: u32 = argmax_dims[argmax_dims.ndim() - 1] as _;
        let lang_t = Tensor::from_data([argmax], shape![1], device);

        Ok(Language::Token(lang_t.item()))
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn detect_language(&mut self, mel: Tensor) -> anyhow::Result<Language> {
        let audio_ctx = self.encoder.schedule(mel)?.resolve()?;
        let sot = Tensor::from_data([WhisperTokenizer::SOT], shape![1, 1], self.device.clone());

        let logits = self.decoder.schedule([audio_ctx, sot])?.resolve()?;
        self.decoder.reset();

        let cpu_logits = logits.to(&Device::CPU).await?;
        let logits = DecodingTask::slice_logits(cpu_logits, self.config.n_vocab as usize);

        let device = logits.device().clone();
        let mut nd_logits = logits.into_ndarray::<f32>();

        let languages_end = if self.config.n_vocab == 51865 {
            50358
        } else if self.config.n_vocab == 51866 {
            50359
        } else {
            panic!("Unsupported number of tokens")
        };

        nd_logits
            .slice_mut(s![.., ..WhisperTokenizer::LANGUAGES_BEGIN])
            .map_inplace(move |el| *el = f32::NEG_INFINITY);

        nd_logits
            .slice_mut(s![.., languages_end..])
            .map_inplace(move |el| *el = f32::NEG_INFINITY);

        let language_tokens_probs = nd_logits.softmax(nd_logits.ndim() - 1);

        let argmax_dims = language_tokens_probs.argmax_skipnan().unwrap();
        let argmax: u32 = argmax_dims[argmax_dims.ndim() - 1] as _;
        let lang_t = Tensor::from_data([argmax], shape![1], device);

        Ok(Language::Token(lang_t.item()))
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::path::PathBuf;

    use hf_hub::api::sync::Api;
    use ratchet::{Device, DeviceRequest};
    use ratchet_loader::gguf::gguf;

    use crate::whisper::{
        model::Whisper, options::DecodingOptionsBuilder, transcribe::transcribe,
        transcript::StreamedSegment,
    };

    fn log_init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    fn load_sample(path: PathBuf) -> Vec<f32> {
        let mut reader = hound::WavReader::open(path).unwrap();
        reader
            .samples::<i16>()
            .map(|x| x.unwrap() as f32 / 32768.0)
            .collect::<Vec<_>>()
    }

    const MM0_Q8_GROUND: [u32; 191] = [
        50364, 639, 307, 264, 4532, 3479, 13460, 264, 881, 34674, 5932, 30340, 295, 5116, 2065,
        5729, 13, 50524, 50524, 1981, 472, 575, 12023, 4365, 337, 257, 1702, 6034, 3028, 1523,
        1804, 4651, 4532, 3479, 50668, 50668, 8963, 6742, 300, 1619, 257, 3804, 5214, 2610, 5214,
        6383, 2643, 5214, 293, 544, 2176, 50816, 50816, 8963, 21800, 281, 747, 604, 1081, 293, 456,
        366, 867, 34674, 3190, 281, 862, 365, 309, 1184, 50948, 50948, 472, 1487, 365, 1080, 1065,
        2121, 11377, 4532, 3479, 5864, 293, 1019, 5456, 4122, 300, 51084, 51084, 544, 20095, 1286,
        13, 51134, 51134, 30062, 264, 13436, 574, 412, 264, 10155, 35310, 587, 264, 3874, 14701,
        1068, 281, 264, 7267, 3096, 2541, 428, 1032, 51264, 51264, 281, 818, 5675, 5300, 264,
        16629, 7283, 293, 613, 3190, 3318, 1214, 281, 1254, 257, 4532, 3479, 1002, 51424, 51424,
        4532, 3479, 8963, 6742, 300, 311, 1270, 257, 7195, 5870, 370, 6239, 13600, 370, 309, 1177,
        380, 51552, 51552, 321, 2607, 1488, 68, 322, 257, 8963, 264, 16026, 4532, 8379, 293, 4532,
        3479, 8963, 6742, 300, 311, 51696, 50364, 3718, 14759, 490, 3114, 996, 264, 4356, 436, 366,
        264, 1101, 436, 366, 13, 50500,
    ];

    #[test]
    pub fn whisper_end_to_end() {
        log_init();
        let api = Api::new().unwrap();
        let model = api.model("FL33TW00D-HF/whisper-tiny".to_string());
        let model_path = model.get("tiny_q8_0.gguf").unwrap();
        println!("PATH: {:?}", model_path.display());

        let dataset = api.dataset("FL33TW00D-HF/ratchet-util".to_string());
        let audio_path = dataset.get("mm0.wav").unwrap();
        let samples = load_sample(audio_path);

        let options = DecodingOptionsBuilder::new().build();
        let mut reader = std::io::BufReader::new(std::fs::File::open(model_path).unwrap());
        let header = gguf::Header::read(&mut reader).unwrap();

        let device = Device::request_device(DeviceRequest::GPU).unwrap();

        let mut whisper = Whisper::load(header, &mut reader, device).unwrap();

        let empty_cb: Option<fn(StreamedSegment)> = None;
        let transcript = transcribe(&mut whisper, samples, options, empty_cb).unwrap();

        let all_tokens = transcript
            .segments
            .iter()
            .flat_map(|s| s.tokens.clone().into_iter())
            .collect::<Vec<_>>();
        assert_eq!(all_tokens, MM0_Q8_GROUND);

        println!("{}", transcript.formatted.unwrap());
        println!("Processing time: {:?}", transcript.processing_time);
    }

    /*
    #[test]
    pub fn convert_ggml_f32_to_wq8() {
        log_init();
        let api = Api::new().unwrap();
        let model = api.model("ggerganov/whisper.cpp".to_string());
        let src_path = model.get("ggml-tiny.bin").unwrap();

        let to_quant = HashSet::from([
            "attn.query.weight",
            "attn.key.weight",
            "attn.value.weight",
            "attn.out.weight",
            "cross_attn.query.weight",
            "cross_attn.key.weight",
            "cross_attn.value.weight",
            "cross_attn.out.weight",
            "mlp.0.weight",
            "mlp.2.weight",
            "token_embedding.weight",
        ]);

        let mut dst_path = src_path.clone();
        dst_path.pop();
        dst_path = dst_path.join("tiny_q8.bin");
        println!("DST: {:?}", dst_path);

        let v3 = false;
        let pad_size = if v3 { 6 } else { 7 };
        let to_pad = HashMap::from([(
            "decoder.token_embedding.weight",
            vec![[0, pad_size], [0, 0]],
        )]);
        let quantization = Quantization::None;
        Converter::convert::<_, Whisper>(src_path, dst_path, quantization, to_quant, to_pad)
            .unwrap();
    }*/
}
