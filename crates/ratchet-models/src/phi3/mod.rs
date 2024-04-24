mod attn;
mod generate;
mod mlp;
mod model;

pub use model::Phi3;

#[cfg(target_arch = "wasm32")]
pub use generate::generate;
