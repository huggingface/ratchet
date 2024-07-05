use crate::bert::Bert;
use tokenizers::Tokenizer;

#[cfg(not(target_arch = "wasm32"))]
pub fn embed(model: &mut Bert, prompt: String, callback: impl Fn(String)) -> anyhow::Result<()> {
    use super::Bert;

    Ok(())
}
