mod attn;
mod mlp;
mod model;

#[cfg(target_arch = "wasm32")]
pub use model::infer;
pub use model::Phi2;
