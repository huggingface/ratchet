use crate::cpu::binary::{add, mul, sub};
use crate::cpu::reindex::broadcast;
use crate::cpu::unary::unary_map_inplace;
use crate::cpu::utils::cpu_store_result;
use crate::reindex::broadcast_vector;
use crate::{
    shape, CPUOperation, Conv, DType, GroupNorm, InvariantError, Norm, NormOp, OperationError,
    Shape, Tensor, TensorDType,
};
use core::iter::Sum;
use half::{bf16, f16};
use num::Float;
use num_traits::NumOps;
use rand::distributions::weighted;

impl CPUOperation for Conv {
    fn apply_cpu(&self, dst: Tensor) -> Result<Tensor, OperationError> {
        let Conv {
            input,
            weight,
            bias,
            stride,
            padding,
        } = self;
        if input.dt() != weight.dt() {
            return Err(InvariantError::DTypeMismatch {
                expected: input.dt(),
                actual: weight.dt(),
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
            DType::F32 => conv::<f32>(input, weight, bias, *stride, *padding, &dst)?,
            _ => todo!(),
        }

        Ok(dst)
    }
}

fn conv<T>(
    input: &Tensor,
    weight: &Tensor,
    bias: &Option<Tensor>,
    stride: usize,
    padding: usize,
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
    let weighted = weight.to_vec::<T>()?;
    let bias = match bias {
        Some(b) => Some(b.to_vec::<T>()?),
        None => None,
    };

    cpu_store_result(&dst, &input);

    Ok(())
}
