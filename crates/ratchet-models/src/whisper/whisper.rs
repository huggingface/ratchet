use std::io::{BufRead, Seek, SeekFrom};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ratchet::{Device, Tensor};
use ratchet_loader::{GGMLCompatible, GGMLFormat, LoadError};

use crate::{Language, SpectrogramGenerator, WhisperDecoder, WhisperEncoder};

pub struct WhisperGGMLHeader {
    pub format: GGMLFormat,
    pub hparams: HyperParameters,
    pub filters: MelFilters,
    pub n_tokens: i32,
}

#[derive(Debug)]
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

    pub fn detect_language(&self, mel: Tensor) -> anyhow::Result<Language> {
        todo!()
    }
}
