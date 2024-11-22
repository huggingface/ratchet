use crate::cpu::binary::{add, mul, sub};
use crate::cpu::reindex::broadcast;
use crate::cpu::unary::unary_map_inplace;
use crate::cpu::utils::cpu_store_result;
use crate::reindex::broadcast_vector;
use crate::{
    shape, CPUOperation, DType, GroupNorm, InvariantError, Norm, NormOp, OperationError, Shape,
    Tensor, TensorDType,
};
use core::iter::Sum;
use half::{bf16, f16};
use num::Float;
use num_traits::NumOps;

impl CPUOperation for NormOp {
    fn apply_cpu(&self, dst: Tensor) -> Result<Tensor, OperationError> {
        match self {
            NormOp::LayerNorm(n) => apply_layer_norm(n, dst),
            NormOp::RMSNorm(n) => apply_rms_norm(n, dst),
            NormOp::GroupNorm(g) => apply_group_norm(g, dst),
        }
    }
}

fn apply_layer_norm(
    Norm {
        input,
        scale,
        bias,
        eps,
    }: &Norm,
    dst: Tensor,
) -> Result<Tensor, OperationError> {
    if input.dt() != scale.dt() {
        return Err(InvariantError::DTypeMismatch {
            expected: input.dt(),
            actual: scale.dt(),
        }
        .into());
    }
    if let Some(b) = bias {
        if b.dt() != input.dt() {
            return Err(InvariantError::DTypeMismatch {
                expected: input.dt(),
                actual: b.dt(),
            }
            .into());
        }
    }

    match input.dt() {
        DType::F32 => layer_norm::<f32>(input, scale, bias, *eps, &dst)?,
        DType::F16 => layer_norm::<f16>(input, scale, bias, *eps, &dst)?,
        DType::BF16 => layer_norm::<bf16>(input, scale, bias, *eps, &dst)?,
        _ => todo!(),
    };

    Ok(dst)
}

fn layer_norm<T>(
    input: &Tensor,
    scale: &Tensor,
    bias: &Option<Tensor>,
    eps: f32,
    dst: &Tensor,
) -> Result<(), OperationError>
where
    T: TensorDType + Float + NumOps + for<'a> Sum<&'a T>,
{
    let src_shape = input.shape();
    let rank = input.rank();
    let N = src_shape[rank - 1];
    let norm_shape = shape!(N);

    let input = input.to_vec::<T>()?;
    let scale = scale.to_vec::<T>()?;
    let bias = match bias {
        Some(b) => Some(b.to_vec::<T>()?),
        None => None,
    };

    let mut x = input.clone();

    let mu = mean(&x, src_shape, rank - 1);
    let mut mu2 = mu.clone();
    square(&mut mu2);
    let mut x2 = input.clone();
    square(&mut x2);
    let mut x2 = mean(&x2, src_shape, rank - 1);

    sub(&mut x2, &mu2);

    let mut mu_b = vec![T::zero(); x.len()];
    broadcast_vector(&mu, &mut mu_b);
    sub(&mut x, &mu_b);

    let eps_vec = vec![T::from(eps).unwrap(); x2.len()];
    add(&mut x2, &eps_vec);
    rsqrt(&mut x2);

    let mut v = vec![T::zero(); x.len()];
    broadcast_vector(&x2, &mut v);
    mul(&mut x, &v);

    let scale_b = broadcast(&scale, &norm_shape, src_shape);
    mul(&mut x, &scale_b);

    if let Some(bias) = bias {
        let bias_b = broadcast(&bias, &norm_shape, src_shape);
        add(&mut x, &bias_b);
    }

    cpu_store_result(&dst, &x);

    Ok(())
}

fn apply_rms_norm(
    Norm {
        input,
        scale,
        bias,
        eps,
    }: &Norm,
    dst: Tensor,
) -> Result<Tensor, OperationError> {
    if input.dt() != scale.dt() {
        return Err(InvariantError::DTypeMismatch {
            expected: input.dt(),
            actual: scale.dt(),
        }
        .into());
    }
    match input.dt() {
        DType::F32 => rms_norm::<f32>(input, scale, *eps, &dst)?,
        DType::F16 => rms_norm::<f16>(input, scale, *eps, &dst)?,
        DType::BF16 => rms_norm::<bf16>(input, scale, *eps, &dst)?,
        _ => todo!(),
    };

    Ok(dst)
}

fn rms_norm<T>(input: &Tensor, scale: &Tensor, eps: f32, dst: &Tensor) -> Result<(), OperationError>
where
    T: TensorDType + Float + NumOps + for<'a> Sum<&'a T>,
{
    let src_shape = input.shape();
    let rank = input.rank();
    let N = src_shape[rank - 1];

    let mut x = input.to_vec::<T>()?;
    let scale = scale.to_vec::<T>()?;

    let mut x2 = x.clone();
    square(&mut x2);
    let mut x2 = mean(&x2, src_shape, rank - 1);
    let eps_vec = vec![T::from(eps).unwrap(); x2.len()];
    add(&mut x2, &eps_vec);
    rsqrt(&mut x2);

    let mut v = vec![T::zero(); x.len()];
    broadcast_vector(&x2, &mut v);
    mul(&mut x, &v);

    let scale_b = broadcast(&scale, &shape!(N), src_shape);
    mul(&mut x, &scale_b);

    cpu_store_result(&dst, &x);

    Ok(())
}

fn apply_group_norm(
    GroupNorm {
        norm:
            Norm {
                input,
                scale,
                bias,
                eps,
            },
        num_groups,
    }: &GroupNorm,
    dst: Tensor,
) -> Result<Tensor, OperationError> {
    if input.dt() != scale.dt() {
        return Err(InvariantError::DTypeMismatch {
            expected: input.dt(),
            actual: scale.dt(),
        }
        .into());
    }
    if let Some(b) = bias {
        if b.dt() != input.dt() {
            return Err(InvariantError::DTypeMismatch {
                expected: input.dt(),
                actual: b.dt(),
            }
            .into());
        }
    }

    match input.dt() {
        DType::F32 => group_norm::<f32>(input, scale, bias, *eps, *num_groups, &dst)?,
        DType::F16 => group_norm::<f16>(input, scale, bias, *eps, *num_groups, &dst)?,
        DType::BF16 => group_norm::<bf16>(input, scale, bias, *eps, *num_groups, &dst)?,
        _ => todo!(),
    };
    Ok(dst)
}

fn group_norm<T>(
    input: &Tensor,
    scale: &Tensor,
    bias: &Option<Tensor>,
    eps: f32,
    num_groups: usize,
    dst: &Tensor,
) -> Result<(), OperationError>
where
    T: TensorDType + Float + NumOps + for<'a> Sum<&'a T>,
{
    let mut src_shape = Shape::promote(input.shape().clone(), 4).to_vec();
    src_shape[0] = num_groups;
    src_shape[1] = src_shape[1] / num_groups;
    let src_shape = &Shape::from(src_shape);

    let N = src_shape[3];

    let norm_shape = shape!(N);

    let input = input.to_vec::<T>()?;
    let scale = scale.to_vec::<T>()?;
    let bias = match bias {
        Some(b) => Some(b.to_vec::<T>()?),
        None => None,
    };

    let mut x = input.clone();

    let mu = mean(&x, src_shape, 3);
    let mut mu2 = mu.clone();
    square(&mut mu2);
    let mut x2 = input.clone();
    square(&mut x2);
    let mut x2 = mean(&x2, src_shape, 3);

    sub(&mut x2, &mu2);

    let mut mu_b = vec![T::zero(); x.len()];
    broadcast_vector(&mu, &mut mu_b);
    sub(&mut x, &mu_b);

    let eps_vec = vec![T::from(eps).unwrap(); x2.len()];
    add(&mut x2, &eps_vec);
    rsqrt(&mut x2);

    let mut v = vec![T::zero(); x.len()];
    broadcast_vector(&x2, &mut v);
    mul(&mut x, &v);

    let scale_b = broadcast(&scale, &norm_shape, src_shape);
    mul(&mut x, &scale_b);

    if let Some(bias) = bias {
        let bias_b = broadcast(&bias, &norm_shape, src_shape);
        add(&mut x, &bias_b);
    }

    cpu_store_result(&dst, &x);

    Ok(())
}

#[inline]
fn square<T: TensorDType + NumOps>(src: &mut [T]) {
    unary_map_inplace(src, |x| x * x)
}

#[inline]
fn rsqrt<T: TensorDType + Float>(src: &mut [T]) {
    unary_map_inplace(src, |x| <T as TensorDType>::one() / x.sqrt())
}

#[inline]
fn mean<T>(src: &[T], shape: &Shape, dim: usize) -> Vec<T>
where
    T: TensorDType + Float + NumOps + for<'a> Sum<&'a T>,
{
    assert_eq!(src.len(), shape.numel());
    let mean_dim = shape.numel() / shape[dim];
    let mut result = vec![T::zero(); mean_dim];
    let step = src.len() / mean_dim;
    let n = T::from(step as f32).unwrap();

    (0..src.len())
        .step_by(step)
        .enumerate()
        .for_each(|(i, chunk)| {
            result[i] = src[chunk..chunk + step].iter().sum::<T>() / n;
        });
    result
}

#[cfg(test)]
mod tests {
    use crate::cpu::norm::{mean, square};
    use crate::shape;

    #[test]
    fn debug_square() {
        let mut a = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        square(&mut a);
        println!("{a:?}");
    }
    #[test]
    fn debug_mean() {
        let a = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = shape!(2, 3);
        let result = mean(&a, &shape, 1);
        println!("{result:?}");
    }
}
