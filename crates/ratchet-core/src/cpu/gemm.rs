use crate::{
    cpu_store_result, shape, CPUOperation, Matmul, MatmulSpec, OperationError, Shape, Strides,
    Tensor, TensorDType,
};
use anyhow::{anyhow, Result};
use gemm::{gemm, Parallelism};
use num_cpus;
use std::str::FromStr;

pub fn get_num_threads() -> usize {
    // Respond to the same environment variable as rayon.
    match std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| usize::from_str(&s).ok())
    {
        Some(x) if x > 0 => x,
        Some(_) | None => num_cpus::get(),
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

fn stride_contiguous(strides: &Strides) -> Vec<isize> {
    let mut stride: Vec<_> = strides
        .to_vec()
        .iter()
        .rev()
        .scan(1, |prod, u| {
            let prod_pre_mult = *prod;
            *prod *= u;
            Some(prod_pre_mult)
        })
        .collect();
    stride.reverse();
    stride
}
fn gemm_impl<T: TensorDType>(
    batches: usize,
    lhs: &[T],
    lhs_shape: &Shape,
    lhs_strides: &Strides,
    rhs: &[T],
    rhs_shape: &Shape,
    rhs_strides: &Strides,
    trans_lhs: bool,
    trans_rhs: bool,
) -> Result<Vec<T>, OperationError> {
    let b = batches;
    let m = if trans_lhs {
        lhs_shape[1]
    } else {
        lhs_shape[0]
    };
    let n = if trans_rhs {
        rhs_shape[0]
    } else {
        rhs_shape[1]
    };
    let k = if trans_lhs {
        lhs_shape[0]
    } else {
        lhs_shape[1]
    };

    println!("b: {b}, m: {m}, n: {n}, k: {k}");

    let lhs_strides = lhs_strides.to_vec();
    let rhs_strides = rhs_strides.to_vec();
    let rank = lhs_shape.rank();

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
    let dst_strides = stride_contiguous(&Strides::from(&shape![m, n]));
    let dst_rs = dst_strides[0];
    let dst_cs = dst_strides[1];

    let mut dst = vec![T::zero(); b * m * n];
    /*
    let num_threads = get_num_threads();
    let parallelism = if num_threads > 1 {
        Parallelism::Rayon(num_threads)
    } else {
        Parallelism::None
    };
    */
    let parallelism = Parallelism::None;

    println!("b: {b}, m: {m}, n: {n}, k: {k}, dst_cs: {dst_cs}, dst_rs: {dst_rs}, a_skip: {a_skip}, b_skip: {b_skip}");

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
            trans_out,
        } = self;

        let lhs_shape = spec.lhs_shape();
        let rhs_shape = spec.rhs_shape();
        let lhs_strides = lhs.strides();
        let rhs_strides = rhs.strides();
        let batches = spec.stacks();

        let lhs = lhs.to_vec::<f32>()?;
        let rhs = rhs.to_vec::<f32>()?;

        let result = gemm_impl::<f32>(
            batches,
            &lhs,
            lhs_shape,
            lhs_strides,
            &rhs,
            rhs_shape,
            rhs_strides,
            *trans_lhs,
            *trans_rhs,
        )?;

        cpu_store_result(&dst_tensor, &result);
        Ok(dst_tensor)
    }
}
