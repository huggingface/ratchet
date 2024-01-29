#![allow(non_snake_case)]
mod compiled_op;
mod device;
mod dtype;
mod enforcer;
mod executable;
mod gpu;
mod kernels;
mod op;
mod ops;
mod quant;
mod shape;
mod storage;
mod strides;
mod tensor;
mod tensor_id;

pub use compiled_op::*;
pub use device::*;
pub use dtype::*;
pub use enforcer::*;
pub use executable::*;
pub use kernels::*;
pub use op::*;
pub use ops::*;
pub use quant::*;
pub use shape::*;
pub use storage::*;
pub use strides::*;
pub use tensor::*;
pub use tensor_id::*;

use smallvec::SmallVec;
pub type RVec<T> = SmallVec<[T; 4]>;
pub type DRVec<T> = SmallVec<[T; 8]>; //Double RVec
pub type RawGPUBuffer = wgpu::Buffer;

//https://github.com/sonos/tract/blob/main/data/src/macros.rs#L2
#[macro_export]
macro_rules! rvec {
    (@one $x:expr) => (1usize);
    ($elem:expr; $n:expr) => ({
        $crate::RVec::from_elem($elem, $n)
    });
    ($($x:expr),*$(,)*) => ({
        let count = 0usize $(+ rvec![@one $x])*;
        #[allow(unused_mut)]
        let mut vec = $crate::RVec::new();
        if count <= vec.inline_size() {
            $(vec.push($x);)*
            vec
        } else {
            $crate::RVec::from_vec(vec![$($x,)*])
        }
    });
}

#[macro_export]
macro_rules! drvec {
    (@one $x:expr) => (1usize);
    ($elem:expr; $n:expr) => ({
        $crate::DRVec::from_elem($elem, $n)
    });
    ($($x:expr),*$(,)*) => ({
        let count = 0usize $(+ rvec![@one $x])*;
        #[allow(unused_mut)]
        let mut vec = $crate::DRVec::new();
        if count <= vec.inline_size() {
            $(vec.push($x);)*
            vec
        } else {
            $crate::DRVec::from_vec(vec![$($x,)*])
        }
    });
}

#[macro_export]
macro_rules! shape {
    ($($x:expr),*$(,)*) => ({
        use $crate::rvec;
        $crate::Shape::new(rvec![$($x,)*])
    });
}

pub mod prelude {
    pub use crate::{rvec, shape, Device, Tensor};
}

#[cfg(test)]
pub mod test_util {
    use crate::Tensor;
    use regex::Regex;
    #[cfg(feature = "pyo3")]
    use {
        numpy::PyArrayDyn,
        pyo3::{prelude::*, types::PyTuple},
    };

    /// It's a bit of a hack, but it's useful for testing.
    #[cfg(feature = "pyo3")]
    pub fn run_py_prg(prg: String, args: &[&Tensor]) -> anyhow::Result<Tensor> {
        let re = Regex::new(r"def\s+(\w+)\s*\(").unwrap();
        let func = match re.captures(&prg) {
            Some(caps) => caps.get(1).map(|m| m.as_str()).unwrap(),
            None => return Err(anyhow::anyhow!("No function name found")),
        };

        Python::with_gil(|py| {
            let prg = PyModule::from_code(py, &prg, "x.py", "x")?;
            let py_args = PyTuple::new(py, args.iter().map(|arg| arg.to_py::<f32>(&py)));
            let py_result: &PyArrayDyn<f32> = prg.getattr(func)?.call1(py_args)?.extract()?;
            Ok(Tensor::from(py_result))
        })
    }
}
