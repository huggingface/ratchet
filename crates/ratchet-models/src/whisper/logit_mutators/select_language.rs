use ndarray::{s, Dimension};
use ndarray_stats::QuantileExt;
use ratchet::{prelude::shape, NDArrayExt, Tensor};

use crate::{LogitMutator, WhisperTokenizer};
pub struct SelectLanguage;

impl LogitMutator for SelectLanguage {
    fn apply(&self, logits: Tensor, _: Option<&Tensor>) -> Result<Tensor, anyhow::Error> {
        let device = logits.device().clone();
        let mut nd_logits = logits.into_ndarray::<f32>();
        nd_logits
            .slice_mut(s![.., ..WhisperTokenizer::LANGUAGES_BEGIN])
            .map_inplace(move |el| *el = f32::NEG_INFINITY);

        nd_logits
            .slice_mut(s![.., WhisperTokenizer::LANGUAGES_END..])
            .map_inplace(move |el| *el = f32::NEG_INFINITY);

        let language_tokens_probs = nd_logits.softmax(nd_logits.ndim() - 1);

        let argmax_dims = language_tokens_probs.argmax_skipnan().unwrap();
        let argmax: u32 = argmax_dims[argmax_dims.ndim() - 1] as _;
        Ok(Tensor::from_data([argmax], shape![1], device))
    }
}
