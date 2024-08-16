use crate::{
    cpu_store_result, shape, CPUOperation, Matmul, MatmulSpec, OperationError, Shape, Strides,
    Tensor, TensorDType,
};
use anyhow::{anyhow, Result};
use core::str::FromStr;
use gemm::{gemm, Parallelism};
use std::num::NonZeroUsize;

pub fn get_num_threads() -> NonZeroUsize {
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

fn calculate_ab_skip(
    lhs_shape: &Shape,
    lhs_strides: &[isize],
    rhs_shape: &Shape,
    rhs_strides: &[isize],
    rank: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(usize, usize)> {
    let rhs_strides = rhs_strides.to_vec();

    let a_skip: usize = match lhs_strides[..rank - 2] {
        [s1, stride] if s1 == stride * lhs_shape[1] as isize => stride as usize,
        [_, stride] if lhs_shape[0] == 1 => stride as usize,
        [stride, _] if lhs_shape[1] == 1 => stride as usize,
        [stride] => stride as usize,
        [] => m * k,
        _ => Err(anyhow!("non-contiguous lhs"))?,
    };
    let b_skip: usize = match rhs_strides[..rank - 2] {
        [s1, stride] if s1 == stride * rhs_shape[1] as isize => stride as usize,
        [_, stride] if rhs_shape[0] == 1 => stride as usize,
        [stride, _] if rhs_shape[1] == 1 => stride as usize,
        [stride] => stride as usize,
        [] => n * k,
        _ => Err(anyhow!("non-contiguous rhs"))?,
    };
    Ok((a_skip, b_skip))
}

fn gemm_impl<T: TensorDType>(
    spec: MatmulSpec,
    lhs: &[T],
    rhs: &[T],
) -> Result<Vec<T>, OperationError> {
    let lhs_shape = spec.lhs_shape();
    let rhs_shape = spec.rhs_shape();
    let dst_shape = spec.dst_shape();
    let lhs_strides = Strides::from(lhs_shape);
    let rhs_strides = Strides::from(rhs_shape);
    let dst_strides = Strides::from(dst_shape);
    let b = spec.stacks();
    let m = spec.m();
    let n = spec.n();
    let k = spec.k();

    let lhs_strides = lhs_strides.to_vec();
    let rhs_strides = rhs_strides.to_vec();
    let rank = lhs_shape.rank();

    println!("lhs_strides: {lhs_strides:?}, rhs_strides: {rhs_strides:?}, rank: {rank}");

    let lhs_cs = lhs_strides[rank - 1];
    let lhs_rs = lhs_strides[rank - 2];

    let rhs_cs = rhs_strides[rank - 1];
    let rhs_rs = rhs_strides[rank - 2];

    let (a_skip, b_skip) = calculate_ab_skip(
        lhs_shape,
        &lhs_strides,
        rhs_shape,
        &rhs_strides,
        rank,
        m,
        n,
        k,
    )?;
    let c_skip: usize = m * n;
    let dst_rs = dst_strides[0];
    let dst_cs = dst_strides[1];

    let mut dst = vec![T::zero(); b * m * n];

    let num_threads = get_num_threads().get();
    let parallelism = if num_threads > 1 {
        Parallelism::Rayon(num_threads)
    } else {
        Parallelism::None
    };

    println!("b: {b}, m: {m}, n: {n}, k: {k}");
    println!("dst_cs: {dst_cs}, dst_rs: {dst_rs}, a_skip: {a_skip}, b_skip: {b_skip}");
    println!("lhs_cs: {lhs_cs}, lhs_rs: {lhs_rs}, rhs_cs: {rhs_cs}, rhs_rs: {rhs_rs}");

    for step in 0..b {
        let lhs_p = &lhs[step * a_skip..];
        let rhs_p = &rhs[step * b_skip..];
        let dst_p = &mut dst[step * c_skip..];
        unsafe {
            gemm(
                m,
                n,
                k,
                dst_p.as_mut_ptr(),
                dst_cs as isize,
                dst_rs as isize,
                false,
                lhs_p.as_ptr(),
                lhs_cs as isize,
                lhs_rs as isize,
                rhs_p.as_ptr(),
                rhs_cs as isize,
                rhs_rs as isize,
                T::zero(),
                T::one(),
                false,
                false,
                false,
                parallelism,
            )
        }
    }
    Ok(dst)
}

impl CPUOperation for Matmul {
    fn apply(&self, dst_tensor: Tensor) -> Result<Tensor, OperationError> {
        let spec = self.compute_spec();

        let Matmul {
            lhs,
            rhs,
            bias,
            trans_lhs,
            trans_rhs,
            trans_dst,
        } = self;

        let lhs = lhs.to_vec::<f32>()?;
        let rhs = rhs.to_vec::<f32>()?;

        let result = gemm_impl::<f32>(spec, &lhs, &rhs)?;
        println!("{result:?}");

        cpu_store_result(&dst_tensor, &result);
        Ok(dst_tensor)
    }
}
