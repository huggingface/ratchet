use byteorder::{LittleEndian, ReadBytesExt};
use ratchet::Shape;
use std::{
    collections::HashMap,
    io::{BufRead, Seek, SeekFrom},
    mem::MaybeUninit,
};

use crate::{GgmlDType, LoadError};

trait ReadBytesCustom: ReadBytesExt {
    /// Extends to read an exact number of bytes.
    fn read_bytes_with_len(&mut self, len: usize) -> std::io::Result<Vec<u8>> {
        let mut buf: Vec<MaybeUninit<u8>> = Vec::with_capacity(len);
        unsafe {
            buf.set_len(len);
        }
        let buf_slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, len) };
        self.read_exact(buf_slice)?;
        let buf = unsafe { std::mem::transmute::<_, Vec<u8>>(buf) };
        Ok(buf)
    }
}
impl<T: std::io::BufRead> ReadBytesCustom for T {}

pub const MAGIC_GGML: u32 = 0x67676d6c;
pub const MAGIC_GGJT: u32 = 0x67676a74;
pub const MAGIC_GGLA: u32 = 0x67676C61;
pub const MAGIC_GGMF: u32 = 0x67676d66;
pub const MAGIC_GGSN: u32 = 0x6767736e;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GGMLFormat {
    GGML(u32),
    GGJT(u32, u32),
    GGLA(u32, u32),
    GGMF(u32, u32),
    GGSN(u32, u32),
}

impl GGMLFormat {
    pub fn read<R: BufRead + Seek>(reader: &mut R) -> Result<GGMLFormat, LoadError> {
        let magic = reader.read_u32::<byteorder::LittleEndian>()?;
        match magic {
            MAGIC_GGML => Ok(GGMLFormat::GGML(magic)),
            _ => {
                let version = reader.read_u32::<byteorder::LittleEndian>()?;
                match magic {
                    MAGIC_GGJT if (1..=3).contains(&version) => {
                        Ok(GGMLFormat::GGJT(magic, version))
                    }
                    MAGIC_GGLA if version == 1 => Ok(GGMLFormat::GGLA(magic, version)),
                    MAGIC_GGMF if version == 1 => Ok(GGMLFormat::GGMF(magic, version)),
                    _ => Err(LoadError::InvalidFormat(magic)),
                }
            }
        }
    }

    fn align32(&self) -> bool {
        match self {
            Self::GGML(_) => false,
            Self::GGJT(_, _) => true,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
pub struct HParams {
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

impl HParams {
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
}

#[derive(Debug, Clone)]
pub struct TensorHeader {
    pub name: String,
    pub shape: Shape,
    pub dtype: GgmlDType,
    pub start_offset: u64,
    pub numel: usize,
}

impl TensorHeader {
    fn new(name: String, shape: Shape, dtype: GgmlDType, start_offset: u64) -> Self {
        let numel = shape.numel();
        Self {
            name,
            shape,
            dtype,
            start_offset,
            numel,
        }
    }

    fn data_size(&self) -> usize {
        self.numel * self.dtype.type_size() / self.dtype.block_size()
    }

    pub fn read_data<R: BufRead + Seek>(&self, reader: &mut R) -> std::io::Result<Vec<u8>> {
        let n_bytes = self.data_size();
        let mut buf: Vec<MaybeUninit<u8>> = Vec::with_capacity(n_bytes);
        unsafe {
            buf.set_len(n_bytes);
        }
        let buf_slice =
            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n_bytes) };

        reader.seek(SeekFrom::Start(self.start_offset))?;
        reader.read_exact(buf_slice)?;
        let buf = unsafe { std::mem::transmute::<_, Vec<u8>>(buf) };
        Ok(buf)
    }
}

pub struct GGMLLoader;

impl GGMLLoader {
    pub fn load<R: BufRead + Seek>(
        reader: &mut R,
    ) -> Result<HashMap<String, TensorHeader>, LoadError> {
        let mut tensor_map = HashMap::new();
        let last_position = reader.seek(std::io::SeekFrom::End(0))?;
        reader.seek(std::io::SeekFrom::Start(0))?;
        let _format = GGMLFormat::read(reader)?;
        let hparams = HParams::read(reader)?;
        let filters = MelFilters::read(reader)?;
        let n_tokens = reader.read_i32::<LittleEndian>()?;
        for _ in 0..n_tokens {
            let token_len = reader.read_u32::<LittleEndian>()?;
            reader.seek(SeekFrom::Current(token_len as i64))?;
        }

        while reader.stream_position()? != last_position {
            let header = Self::load_single(reader)?;
            tensor_map.insert(header.name.clone(), header);
        }
        Ok(tensor_map)
    }

    fn load_single<R: BufRead + Seek>(reader: &mut R) -> Result<TensorHeader, LoadError> {
        let n_dims: usize = reader.read_i32::<LittleEndian>()?.try_into()?;
        let name_len = reader.read_i32::<LittleEndian>()?;
        let dtype = reader.read_u32::<LittleEndian>()?;

        let mut dims = vec![0u32; n_dims as usize];
        reader.read_u32_into::<LittleEndian>(&mut dims)?;
        dims.reverse();

        let name = String::from_utf8(reader.read_bytes_with_len(name_len as _)?)?;
        let dtype = GgmlDType::try_from(dtype).map_err(|_| LoadError::UnsupportedDType {
            name: name.clone(),
            dtype,
        })?;

        let start_offset = reader.stream_position()?;
        let shape: Shape = dims.into();
        let header = TensorHeader::new(name, shape, dtype.into(), start_offset);
        let data_size = header.data_size() as u64;
        reader.seek(SeekFrom::Start(start_offset + data_size))?;
        Ok(header)
    }
}

#[cfg(test)]
mod tests {
    use super::GGMLLoader;
    use hf_hub::api::sync::Api;

    #[test]
    fn load_ggml() {
        let api = Api::new().unwrap();
        let model = api.model("ggerganov/whisper.cpp".to_string());
        let path = model.get("ggml-tiny.bin").unwrap();

        let tensors = GGMLLoader::load(&mut std::io::BufReader::new(
            std::fs::File::open(path).unwrap(),
        ))
        .unwrap();
        assert_eq!(tensors.len(), 167);
    }
}
