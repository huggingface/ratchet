use crate::cpu::utils::cpu_store_result;
use crate::{CPUOperation, DType, OperationError, Softmax, Tensor, TensorDType};
use half::{bf16, f16};
use num::Float;
use num_traits::NumAssignOps;

impl CPUOperation for Softmax {
    fn apply_cpu(&self, dst: Tensor) -> Result<Tensor, OperationError> {
        let Softmax { input, dim } = self;
        match input.dt() {
            DType::F32 => softmax::<f32>(input, *dim, &dst)?,
            DType::F16 => softmax::<f16>(input, *dim, &dst)?,
            DType::BF16 => softmax::<bf16>(input, *dim, &dst)?,
            _ => todo!(),
        }

        Ok(dst)
    }
}

fn softmax<T>(input: &Tensor, dim: usize, dst: &Tensor) -> Result<(), OperationError>
where
    T: TensorDType + Float + NumAssignOps,
{
    let src_shape = input.shape();
    let mut input = input.to_vec::<T>()?;
    let N = src_shape[dim];
    input.chunks_mut(N).for_each(|chunk| {
        let mut sum = T::zero();
        for j in 0..N {
            chunk[j] = chunk[j].exp();
            sum += chunk[j];
        }
        for j in 0..N {
            chunk[j] /= sum;
        }
    });

    cpu_store_result(dst, &input);

    Ok(())
}
