#![feature(trait_alias)]
mod dtype;
mod gpu;
mod storage;
mod tensor;

pub use dtype::*;
pub use storage::*;
pub use tensor::*;

use smallvec::SmallVec;
pub type RVec<T> = SmallVec<[T; 4]>;
