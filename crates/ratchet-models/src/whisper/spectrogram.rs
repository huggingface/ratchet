//Adapted from: https://github.com/tanmayb123/OpenAI-Whisper-CoreML
use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt;
use num::complex::Complex;
use ratchet::Tensor;
use realfft::{RealFftPlanner, RealToComplex};
use std::f32::consts::PI;
use std::sync::Arc;

pub static SAMPLE_RATE: usize = 16000;
pub static N_FFT: usize = 400;
pub static HOP_LENGTH: usize = 160;
pub static CHUNK_LENGTH: usize = 30;
pub static N_AUDIO_CTX: usize = 1500; //same for all
pub static N_SAMPLES: usize = SAMPLE_RATE * CHUNK_LENGTH; // 480000
pub static N_FRAMES: usize = N_SAMPLES / HOP_LENGTH; // 3000
pub static FFT_PAD: usize = N_FFT / 2;

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("Audio must be 30 seconds long (with stft padding): {0} != {1}")]
    InvalidLength(usize, usize),
    #[error("Invalid audio provided: {0}")]
    InvalidAudio(#[from] anyhow::Error),
}

pub struct SpectrogramGenerator {
    fft_plan: Arc<dyn RealToComplex<f32>>,
    hann_window: Array1<f32>,
    mels: Array2<f32>,
}

impl std::fmt::Debug for SpectrogramGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpectrogramGenerator").finish()
    }
}

impl SpectrogramGenerator {
    pub fn new(mels: Vec<f32>) -> Self {
        let mut planner = RealFftPlanner::new();
        let n_mels = mels.len() / (N_FFT / 2 + 1);
        Self {
            fft_plan: planner.plan_fft_forward(N_FFT),
            hann_window: Self::hann_window(),
            mels: Array2::from_shape_vec((n_mels, N_FFT / 2 + 1), mels).unwrap(),
        }
    }

    fn hann_window() -> Array1<f32> {
        let window = (0..N_FFT)
            .map(|i| (i as f32 * 2.0 * PI) / N_FFT as f32)
            .map(|i| (1.0 - i.cos()) / 2.0)
            .collect::<Vec<_>>();
        Array1::from(window)
    }

    fn fft(&self, audio: &[f32]) -> Vec<Complex<f32>> {
        let mut input = Array1::from_vec(audio.to_vec());
        input *= &self.hann_window;
        let mut spectrum = self.fft_plan.make_output_vec();
        self.fft_plan
            .process(input.as_slice_mut().unwrap(), &mut spectrum)
            .unwrap();
        spectrum
    }

    fn mel_spectrogram(&self, audio: &[f32]) -> Tensor {
        let n_frames = (audio.len() - N_FFT) / HOP_LENGTH;
        let right_padding = N_SAMPLES + FFT_PAD; //padding is all 0s, so we can ignore it

        let mut spectrogram = Array2::<f32>::zeros((201, n_frames));
        for i in (0..audio.len() - right_padding).step_by(HOP_LENGTH) {
            if i / HOP_LENGTH >= n_frames {
                break;
            }
            let fft = self.fft(&audio[i..i + N_FFT]);
            let spectrogram_col = fft.iter().map(|c| c.norm_sqr()).collect::<Array1<f32>>();
            spectrogram
                .column_mut(i / HOP_LENGTH)
                .assign(&spectrogram_col);
        }

        let mut mel_spec = self.mels.dot(&spectrogram);
        mel_spec.mapv_inplace(|x| x.max(1e-10).log10());
        let max = *mel_spec.max().unwrap();
        mel_spec.mapv_inplace(|x| (x.max(max - 8.0) + 4.0) / 4.0);
        let expanded = mel_spec.insert_axis(ndarray::Axis(0));
        Tensor::from(expanded.into_dyn())
    }

    pub fn generate(&self, audio: Vec<f32>) -> Result<Tensor, AudioError> {
        if audio.is_empty() {
            return Err(AudioError::InvalidAudio(anyhow::anyhow!(
                "Audio must be non-empty"
            )));
        }
        let padded = Self::pad_audio(audio, N_SAMPLES);
        Ok(self.mel_spectrogram(&padded))
    }

    //The padding done by OAI is as follows:
    //1. First explicitly pad with (CHUNK_LENGTH * SAMPLE_RATE) (480,000) zeros
    //2. Then perform a reflection padding of FFT_PAD (200) samples on each side
    //   This must be done with care, because we have already performed the explicit padding
    //   the pre-padding will contain non-zero values, but the post-padding must be zero
    pub fn pad_audio(audio: Vec<f32>, padding: usize) -> Vec<f32> {
        let padded_len = FFT_PAD + audio.len() + padding + FFT_PAD;
        let mut padded_samples = vec![0.0; padded_len];

        let mut reflect_padding = vec![0.0; FFT_PAD];
        for i in 0..FFT_PAD {
            reflect_padding[i] = audio[FFT_PAD - i];
        }

        padded_samples[0..FFT_PAD].copy_from_slice(&reflect_padding);
        padded_samples[FFT_PAD..(FFT_PAD + audio.len())].copy_from_slice(&audio);
        padded_samples
    }
}

#[cfg(all(test, feature = "pyo3", not(target_arch = "wasm32")))]
mod tests {
    use super::SpectrogramGenerator;
    use hf_hub::api::sync::Api;
    use ratchet::test_util::run_py_prg;
    use ratchet::DType;
    use std::path::PathBuf;

    const MAX_DIFF: f32 = 5e-5;

    pub fn load_npy(path: PathBuf) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap();
        npyz::NpyFile::new(&bytes[..]).unwrap().into_vec().unwrap()
    }

    fn load_sample(path: PathBuf) -> Vec<f32> {
        let mut reader = hound::WavReader::open(path).unwrap();
        reader
            .samples::<i16>()
            .map(|x| x.unwrap() as f32 / 32768.0)
            .collect::<Vec<_>>()
    }

    #[test]
    fn spectrogram_matches() {
        let api = Api::new().unwrap();
        let repo = api.dataset("FL33TW00D-HF/ratchet-util".to_string());
        let gb0 = repo.get("erwin_jp.wav").unwrap();
        let mels = repo.get("mel_filters_128.npy").unwrap();
        let prg = format!(
            r#"
import whisper
import numpy as np
def ground_truth():
    audio = whisper.load_audio("{}")
    return whisper.log_mel_spectrogram(audio, n_mels=128, padding=480000).numpy()[np.newaxis]
"#,
            gb0.to_str().unwrap()
        );
        let ground_truth = run_py_prg(prg.to_string(), &[], &[], DType::F32).unwrap();
        let generator = SpectrogramGenerator::new(load_npy(mels));
        let result = generator.generate(load_sample(gb0)).unwrap();
        ground_truth.all_close(&result, MAX_DIFF, MAX_DIFF).unwrap();
    }
}
