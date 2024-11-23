use crate::cpu::reindex::{nd_index_to_offset, offset_to_ndindex};
use crate::cpu::utils::cpu_store_result;
use crate::{
    CPUOperation, DType, IndexWrite, OperationError, SmallVec, Softmax, Tensor, TensorDType,
};
use half::{bf16, f16};
use num::Float;
use num_traits::NumAssignOps;

impl CPUOperation for IndexWrite {
    fn apply_cpu(&self, dst_real: Tensor) -> Result<Tensor, OperationError> {
        let IndexWrite {
            dst,
            src,
            write_start,
        } = self;
        match src.dt() {
            DType::F32 => index_write::<f32>(src, write_start, &dst_real)?,
            DType::F16 => index_write::<f16>(src, write_start, &dst_real)?,
            DType::BF16 => index_write::<bf16>(src, write_start, &dst_real)?,
            _ => todo!(),
        }

        Ok(dst_real)
    }
}

fn index_write<T>(
    input: &Tensor,
    write_start: &SmallVec<[usize; 4]>,
    dst: &Tensor,
) -> Result<(), OperationError>
where
    T: TensorDType + Float + NumAssignOps,
{
    let src_shape = input.shape();
    let mut input = input.to_vec::<T>()?;

    cpu_store_result(dst, &input);

    Ok(())
}
