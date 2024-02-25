use std::io::{BufRead, Seek, SeekFrom};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ratchet::{shape, Device, Tensor};
use ratchet_loader::{GGMLCompatible, GGMLFormat, GGMLModel, LoadError};
use ratchet_nn::Module;

use crate::{
    DecodingTask, Language, LogitMutator, SelectLanguage, SpectrogramGenerator, WhisperDecoder,
    WhisperEncoder, WhisperTokenizer,
};

pub struct WhisperGGMLHeader {
    pub format: GGMLFormat,
    pub hparams: HyperParameters,
    pub filters: MelFilters,
    pub n_tokens: i32,
}

#[derive(Debug, Clone)]
pub struct HyperParameters {
    pub n_vocab: i32,
    pub n_audio_ctx: i32,
    pub n_audio_state: i32,
    pub n_audio_head: i32,
    pub n_audio_layer: i32,
    pub n_text_ctx: i32,
    pub n_text_state: i32,
    pub n_text_head: i32,
    pub n_text_layer: i32,
    pub n_mels: i32,
    pub ftype: i32,
}

impl HyperParameters {
    pub fn read<R: BufRead>(reader: &mut R) -> Result<Self, std::io::Error> {
        let n_vocab = reader.read_i32::<LittleEndian>()?;
        let n_audio_ctx = reader.read_i32::<LittleEndian>()?;
        let n_audio_state = reader.read_i32::<LittleEndian>()?;
        let n_audio_head = reader.read_i32::<LittleEndian>()?;
        let n_audio_layer = reader.read_i32::<LittleEndian>()?;
        let n_text_ctx = reader.read_i32::<LittleEndian>()?;
        let n_text_state = reader.read_i32::<LittleEndian>()?;
        let n_text_head = reader.read_i32::<LittleEndian>()?;
        let n_text_layer = reader.read_i32::<LittleEndian>()?;
        let n_mels = reader.read_i32::<LittleEndian>()?;
        let ftype = reader.read_i32::<LittleEndian>()?;
        Ok(Self {
            n_vocab,
            n_audio_ctx,
            n_audio_state,
            n_audio_head,
            n_audio_layer,
            n_text_ctx,
            n_text_state,
            n_text_head,
            n_text_layer,
            n_mels,
            ftype,
        })
    }

    pub fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_i32::<LittleEndian>(self.n_vocab)?;
        writer.write_i32::<LittleEndian>(self.n_audio_ctx)?;
        writer.write_i32::<LittleEndian>(self.n_audio_state)?;
        writer.write_i32::<LittleEndian>(self.n_audio_head)?;
        writer.write_i32::<LittleEndian>(self.n_audio_layer)?;
        writer.write_i32::<LittleEndian>(self.n_text_ctx)?;
        writer.write_i32::<LittleEndian>(self.n_text_state)?;
        writer.write_i32::<LittleEndian>(self.n_text_head)?;
        writer.write_i32::<LittleEndian>(self.n_text_layer)?;
        writer.write_i32::<LittleEndian>(self.n_mels)?;
        writer.write_i32::<LittleEndian>(self.ftype)?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct MelFilters {
    pub n_mel: i32,
    pub n_fft: i32,
    pub mels: Vec<f32>,
}

impl MelFilters {
    pub fn read<R: BufRead>(reader: &mut R) -> Result<Self, std::io::Error> {
        let n_mel = reader.read_i32::<LittleEndian>()?;
        let n_fft = reader.read_i32::<LittleEndian>()?;

        let mels = (0..(n_mel * n_fft))
            .map(|_| reader.read_f32::<LittleEndian>())
            .collect::<Result<Vec<f32>, std::io::Error>>()?;

        Ok(Self { n_mel, n_fft, mels })
    }

    pub fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_i32::<LittleEndian>(self.n_mel)?;
        writer.write_i32::<LittleEndian>(self.n_fft)?;
        for mel in &self.mels {
            writer.write_f32::<LittleEndian>(*mel)?;
        }
        Ok(())
    }
}

pub struct Whisper {
    pub specgen: SpectrogramGenerator,
    pub encoder: WhisperEncoder,
    pub decoder: WhisperDecoder,
    pub hparams: HyperParameters,
    pub device: Device,
}

impl Whisper {
    pub fn load<R: BufRead + Seek>(
        disk_model: &GGMLModel<Whisper>,
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let encoder = WhisperEncoder::load(disk_model, reader, device)?;
        let decoder = WhisperDecoder::load(disk_model, reader, device)?;
        //TODO: remove clones
        let generator = crate::SpectrogramGenerator::new(disk_model.header.filters.mels.clone());
        Ok(Self {
            specgen: generator,
            encoder,
            decoder,
            hparams: disk_model.header.hparams.clone(),
            device: device.clone(),
        })
    }
}

impl GGMLCompatible for Whisper {
    type ModelHeader = WhisperGGMLHeader;

    fn load_header<R: BufRead + Seek>(reader: &mut R) -> Result<Self::ModelHeader, LoadError> {
        let format = GGMLFormat::read(reader)?;
        let hparams = HyperParameters::read(reader)?;
        let filters = MelFilters::read(reader)?;
        let n_tokens = reader.read_i32::<LittleEndian>()?;
        for _ in 0..n_tokens {
            let token_len = reader.read_u32::<LittleEndian>()?;
            reader.seek(SeekFrom::Current(token_len as i64))?;
        }
        Ok(Self::ModelHeader {
            format,
            hparams,
            filters,
            n_tokens,
        })
    }

    fn write_header<W: std::io::Write>(
        header: &Self::ModelHeader,
        writer: &mut W,
    ) -> std::io::Result<()> {
        header.format.write(writer)?;
        header.hparams.write(writer)?;
        header.filters.write(writer)?;
        writer.write_i32::<LittleEndian>(header.n_tokens)?;
        for _ in 0..header.n_tokens {
            writer.write_u32::<LittleEndian>(0)?;
        }
        Ok(())
    }
}

impl Whisper {
    pub fn is_multilingual(&self) -> bool {
        self.hparams.n_vocab == 51865
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn detect_language(&mut self, mel: Tensor) -> anyhow::Result<Language> {
        let audio_ctx = self.encoder.forward(&mel)?.resolve()?;
        let sot = Tensor::from_data(&[WhisperTokenizer::SOT], shape![1, 1], self.device.clone());

        let logits = self.decoder.forward(&[audio_ctx, sot])?.resolve()?;
        self.decoder.reset();

        let cpu_logits = logits.to(&Device::CPU)?;
        let logits = DecodingTask::slice_logits(cpu_logits);

        let selector = SelectLanguage {};
        let lang_t = selector.apply(logits, None).unwrap();
        Ok(Language::Token(lang_t.item()))
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn detect_language(&mut self, mel: Tensor) -> anyhow::Result<Language> {
        let audio_ctx = self.encoder.forward(&mel)?.resolve()?;
        let sot = Tensor::from_data(&[WhisperTokenizer::SOT], shape![1, 1], self.device.clone());

        let logits = self.decoder.forward(&[audio_ctx, sot])?.resolve()?;
        self.decoder.reset();

        let cpu_logits = logits.to(&Device::CPU).await?;
        let logits = DecodingTask::slice_logits(cpu_logits);

        let selector = SelectLanguage {};
        let lang_t = selector.apply(logits, None).unwrap();
        Ok(Language::Token(lang_t.item()))
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::path::PathBuf;

    use crate::{transcribe, DecodingOptionsBuilder, Whisper};
    use hf_hub::api::sync::Api;
    use ratchet::{Device, DeviceRequest};
    use ratchet_loader::GGMLCompatible;

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

    #[test]
    pub fn whisper_end_to_end() {
        log_init();
        let api = Api::new().unwrap();
        let model = api.model("ggerganov/whisper.cpp".to_string());
        let model_path = model.get("ggml-tiny.bin").unwrap();

        let dataset = api.dataset("FL33TW00D-HF/ratchet-util".to_string());
        let audio_path = dataset.get("mm0.wav").unwrap();
        let samples = load_sample(audio_path);

        let options = DecodingOptionsBuilder::new().build();
        let mut reader = std::io::BufReader::new(std::fs::File::open(model_path).unwrap());
        let gg_disk = Whisper::load_ggml(&mut reader).unwrap();

        let device = Device::request_device(DeviceRequest::GPU).unwrap();

        let mut whisper = Whisper::load(&gg_disk, &mut reader, &device).unwrap();

        let transcript = transcribe(&mut whisper, samples, options).unwrap();
        println!("{}", transcript.formatted.unwrap());
        println!("Processing time: {:?}", transcript.processing_time);
    }
}
