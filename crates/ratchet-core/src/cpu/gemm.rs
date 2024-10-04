use crate::{
    cpu_store_result, CPUOperation, DType, InvariantError, Matmul, MatmulSpec, OperationError,
    Shape, Tensor, TensorDType,
};
use anyhow::{anyhow, Result};
use core::str::FromStr;
use gemm::{gemm, Parallelism};
use half::{bf16, f16};
use std::num::NonZeroUsize;

fn get_num_threads() -> NonZeroUsize {
    // Respond to the same environment variable as rayon.
    match std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| usize::from_str(&s).ok())
    {
        Some(x) if x > 0 => NonZeroUsize::new(x).unwrap(),
        Some(_) | None => std::thread::available_parallelism()
            .unwrap_or_else(|_| NonZeroUsize::new(1usize).unwrap()),
    }
}

fn get_parallelism() -> Parallelism {
    match get_num_threads().get() {
        1 => Parallelism::None,
        n => Parallelism::Rayon(n),
    }
}

fn calculate_skips(
    lhs_shape: &Shape,
    lhs_strides: &[isize],
    rhs_shape: &Shape,
    rhs_strides: &[isize],
    rank: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(usize, usize)> {
    let lhs_skip: usize = match lhs_strides[..rank - 2] {
        [s1, stride] if s1 == stride * lhs_shape[1] as isize => stride as usize,
        [_, stride] if lhs_shape[0] == 1 => stride as usize,
        [stride, _] if lhs_shape[1] == 1 => stride as usize,
        [stride] => stride as usize,
        [] => m * k,
        _ => Err(anyhow!("non-contiguous lhs"))?,
    };
    let rhs_skip: usize = match rhs_strides[..rank - 2] {
        [s1, stride] if s1 == stride * rhs_shape[1] as isize => stride as usize,
        [_, stride] if rhs_shape[0] == 1 => stride as usize,
        [stride, _] if rhs_shape[1] == 1 => stride as usize,
        [stride] => stride as usize,
        [] => n * k,
        _ => Err(anyhow!("non-contiguous rhs"))?,
    };
    Ok((lhs_skip, rhs_skip))
}

fn gemm_impl<T: TensorDType>(
    spec: MatmulSpec,
    lhs: &[T],
    rhs: &[T],
) -> Result<Vec<T>, OperationError> {
    let lhs_shape = spec.lhs_shape();
    let rhs_shape = spec.rhs_shape();
    let lhs_strides = spec.lhs_strides();
    let rhs_strides = spec.rhs_strides();
    let dst_strides = spec.dst_strides();
    let b = spec.stacks();
    let m = spec.m();
    let n = spec.n();
    let k = spec.k();

    let lhs_strides = lhs_strides.to_vec();
    let rhs_strides = rhs_strides.to_vec();
    let rank = lhs_shape.rank();

    let lhs_cs = lhs_strides[rank - 1];
    let lhs_rs = lhs_strides[rank - 2];

    let rhs_cs = rhs_strides[rank - 1];
    let rhs_rs = rhs_strides[rank - 2];

    let (lhs_skip, rhs_skip) = calculate_skips(
        lhs_shape,
        &lhs_strides,
        rhs_shape,
        &rhs_strides,
        rank,
        m,
        n,
        k,
    )?;
    let dst_skip: usize = m * n;
    let dst_rs = dst_strides[0];
    let dst_cs = dst_strides[1];

    let mut dst = vec![T::zero(); b * m * n];

    for step in 0..b {
        let lhs_p = &lhs[step * lhs_skip..];
        let rhs_p = &rhs[step * rhs_skip..];
        let dst_p = &mut dst[step * dst_skip..];
        unsafe {
            gemm(
                m,
                n,
                k,
                dst_p.as_mut_ptr(),
                dst_cs,
                dst_rs,
                false,
                lhs_p.as_ptr(),
                lhs_cs,
                lhs_rs,
                rhs_p.as_ptr(),
                rhs_cs,
                rhs_rs,
                T::zero(),
                T::one(),
                false,
                false,
                false,
                get_parallelism(),
            )
        }
    }
    Ok(dst)
}

impl CPUOperation for Matmul {
    fn apply_cpu(&self, dst: Tensor) -> Result<Tensor, OperationError> {
        fn run_gemm<T: TensorDType>(
            spec: MatmulSpec,
            lhs: &Tensor,
            rhs: &Tensor,
            dst: &Tensor,
        ) -> Result<(), OperationError> {
            let lhs = lhs.to_vec::<T>()?;
            let rhs = rhs.to_vec::<T>()?;

            let result = if spec.trans_dst() {
                gemm_impl::<T>(spec, &rhs, &lhs)?
            } else {
                gemm_impl::<T>(spec, &lhs, &rhs)?
            };
            cpu_store_result(dst, &result);
            Ok(())
        }
        let spec = self.compute_spec();

        let Matmul { lhs, rhs, .. } = self;

        match self.lhs.dt() {
            DType::F32 => run_gemm::<f32>(spec, lhs, rhs, &dst),
            DType::F16 => run_gemm::<f16>(spec, lhs, rhs, &dst),
            DType::BF16 => run_gemm::<bf16>(spec, lhs, rhs, &dst),
            dtype => Err(InvariantError::UnsupportedDType(dtype))?,
        }?;
        Ok(dst)
    }
}
