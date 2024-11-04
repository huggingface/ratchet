use super::utils::cpu_store_result;
use crate::{CPUOperation, DType, OperationError, Reindex, Slice, Strides, Tensor, TensorDType};
use half::{bf16, f16};

impl CPUOperation for Reindex {
    fn apply_cpu(&self, dst: Tensor) -> Result<Tensor, OperationError> {
        match self {
            Reindex::Slice(s) => s.apply_cpu(dst),
            _ => todo!(),
        }
    }
}

impl CPUOperation for Slice {
    fn apply_cpu(&self, dst: Tensor) -> Result<Tensor, OperationError> {
        match dst.dt() {
            DType::F32 => apply_slice::<f32>(self, dst),
            DType::BF16 => apply_slice::<bf16>(self, dst),
            DType::F16 => apply_slice::<f16>(self, dst),
            DType::I32 => apply_slice::<i32>(self, dst),
            DType::U32 => apply_slice::<u32>(self, dst),
            _ => todo!(),
        }
    }
}

fn apply_slice<T: TensorDType>(s: &Slice, dst: Tensor) -> Result<Tensor, OperationError> {
    let (start, stop): (Vec<_>, Vec<_>) = s.indices().iter().map(|r| (r.start, r.end)).unzip();
    let result = slice(&s.src.to_vec::<T>()?, s.src.strides(), &start, &stop);

    cpu_store_result(&dst, &result);
    Ok(dst)
}

pub(crate) fn slice<T: TensorDType>(
    src: &[T],
    src_strides: &Strides,
    start: &[usize],
    stop: &[usize],
) -> Vec<T> {
    assert!(start.len() == stop.len());
    assert!(start.len() == src_strides.rank());
    start.iter().zip(stop.iter()).for_each(|(s, t)| {
        assert!(s < t);
    });

    let dst_shape: Vec<usize> = stop.iter().zip(start.iter()).map(|(s, t)| s - t).collect();
    let dst_numel: usize = dst_shape.iter().product();

    let mut dst = vec![T::zero(); dst_numel];

    for i in 0..dst_numel {
        let mut src_index = 0;
        let mut tmp = i;
        for d in 0..dst_shape.len() {
            let coord = tmp / dst_shape[d + 1..].iter().product::<usize>().max(1);
            tmp %= dst_shape[d + 1..].iter().product::<usize>().max(1);
            src_index += (coord + start[d]) * src_strides[d] as usize;
        }
        dst[i] = src[src_index];
    }

    dst
}
