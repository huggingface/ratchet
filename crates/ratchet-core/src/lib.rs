#![allow(non_snake_case)]
mod compiled_op;
mod device;
mod dtype;
mod enforcer;
mod executable;
mod gpu;
mod kernels;
mod ndarray_ext;
mod op;
mod ops;
mod plot;
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
pub use gpu::*;
pub use kernels::*;
pub use ndarray_ext::*;
pub use op::*;
pub use ops::*;
pub use quant::*;
pub use shape::*;
pub use storage::*;
pub use strides::*;
pub use tensor::*;
pub use tensor_id::*;

#[cfg(feature = "plotting")]
pub use plot::render_to_file;

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
    pub use crate::{rvec, shape, Device, DeviceRequest, Tensor};
}

#[cfg(feature = "pyo3")]
pub mod test_util {
    use crate::{DType, Tensor, TensorDType};
    use half::f16;
    use regex::Regex;
    use {
        numpy::PyArrayDyn,
        pyo3::{prelude::*, types::PyTuple},
    };

    /// It's a bit of a hack, but it's useful for testing.
    pub fn run_py_prg(
        prg: String,
        tensors: &[&Tensor],
        args: &[&dyn ToPyObject],
        dst_dtype: DType,
    ) -> anyhow::Result<Tensor> {
        let re = Regex::new(r"def\s+(\w+)\s*\(").unwrap();
        let func = match re.captures(&prg) {
            Some(caps) => caps.get(1).map(|m| m.as_str()).unwrap(),
            None => return Err(anyhow::anyhow!("No function name found")),
        };

        Python::with_gil(|py| {
            let prg = PyModule::from_code(py, &prg, "x.py", "x")?;
            let py_tensors = tensors.iter().map(|t| match t.dt() {
                DType::F32 => t.to_py::<f32>(&py).to_object(py),
                DType::I32 => t.to_py::<i32>(&py).to_object(py),
                DType::F16 => t.to_py::<f16>(&py).to_object(py),
                _ => unimplemented!(),
            });
            let py_args = py_tensors
                .chain(args.iter().map(|a| a.to_object(py)))
                .collect::<Vec<_>>();
            let py_args = PyTuple::new(py, py_args);
            let py_result = prg.getattr(func)?.call1(py_args)?;
            let result: Tensor = match dst_dtype {
                DType::F32 => py_result.extract::<&PyArrayDyn<f32>>()?.into(),
                DType::F16 => py_result.extract::<&PyArrayDyn<f16>>()?.into(),
                DType::I32 => py_result.extract::<&PyArrayDyn<i32>>()?.into(),
                DType::U32 => py_result.extract::<&PyArrayDyn<u32>>()?.into(),
                _ => unimplemented!(),
            };
            Ok(result)
        })
    }
}
