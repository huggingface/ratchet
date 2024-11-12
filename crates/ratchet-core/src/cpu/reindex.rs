use super::utils::{
    cpu_store_result, TensorIterator,
    TensorIterator::{Contiguous, Strided},
};
use crate::{
    Broadcast, CPUOperation, DType, OperationError, Reindex, Shape, Slice, Strides, Tensor,
    TensorDType,
};
use half::{bf16, f16};

impl CPUOperation for Reindex {
    fn apply_cpu(&self, dst: Tensor) -> Result<Tensor, OperationError> {
        match self {
            Reindex::Slice(s) => s.apply_cpu(dst),
            Reindex::Broadcast(b) => b.apply_cpu(dst),
            _ => todo!(),
        }
    }
}
impl CPUOperation for Broadcast {
    fn apply_cpu(&self, dst: Tensor) -> Result<Tensor, OperationError> {
        match dst.dt() {
            DType::F32 => apply_broadcast::<f32>(self, dst),
            DType::BF16 => apply_broadcast::<bf16>(self, dst),
            DType::F16 => apply_broadcast::<f16>(self, dst),
            DType::I32 => apply_broadcast::<i32>(self, dst),
            DType::U32 => apply_broadcast::<u32>(self, dst),
            _ => todo!(),
        }
    }
}

fn apply_broadcast<T: TensorDType>(b: &Broadcast, dst: Tensor) -> Result<Tensor, OperationError> {
    let result = broadcast(&b.src.to_vec::<T>()?, b.src.shape(), b.to());
    cpu_store_result(&dst, &result);
    Ok(dst)
}

fn get_contiguous_offsets(
    shape: &Shape,
    strides: &Strides,
) -> Option<(usize, usize, usize, usize)> {
    let mut left_broadcast = 1;
    let mut right_broadcast = 1;
    let dims = shape.to_vec();
    let strides = strides.to_vec();
    let mut start_cont = 0;
    let mut end_cont = dims.len();
    for (&s, &d) in strides.iter().zip(dims.iter()) {
        if s != 0 {
            break;
        }
        start_cont += 1;
        left_broadcast *= d;
    }
    if start_cont == dims.len() {
        return Some((0, 1, left_broadcast, 1));
    }
    for (&s, &d) in strides.iter().zip(dims.iter()).rev() {
        if s != 0 {
            break;
        }
        end_cont -= 1;
        right_broadcast *= d;
    }
    // Check that the inner dims are contiguous
    let strides = &strides[start_cont..end_cont];
    let dims = &dims[start_cont..end_cont];
    let mut len = 1;
    for (&stride, &dim) in strides.iter().zip(dims.iter()).rev() {
        if stride as usize != len {
            return None;
        }
        len *= dim;
    }

    Some((0, len, left_broadcast, right_broadcast))
}

fn offset_to_ndindex(offset: usize, strides: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; strides.len()];
    let mut remaining = offset;

    for i in 0..strides.len() - 1 {
        let stride = strides[i];
        let idx = remaining / stride;
        indices[i] = idx;
        remaining -= idx * stride;
    }
    indices[strides.len() - 1] = remaining;
    indices
}

fn nd_index_to_offset(ndindex: &[usize], strides: &[usize]) -> usize {
    ndindex.iter().zip(strides.iter()).map(|(x, y)| x * y).sum()
}

pub(crate) fn broadcast<T: TensorDType>(src: &[T], src_shape: &Shape, dst_shape: &Shape) -> Vec<T> {
    let src_strides = Strides::from(src_shape);
    let dst_strides = Strides::from(dst_shape);
    let mut result = vec![T::zero(); dst_shape.numel()];

    let dst_shape = dst_shape.to_vec();

    let src_strides: Vec<usize> = src_strides.as_slice().iter().map(|x| *x as usize).collect();
    let dst_strides: Vec<usize> = dst_strides.as_slice().iter().map(|x| *x as usize).collect();

    if src_shape.is_scalar() {
        // Life is simple
        result.fill(src[0]);
    } else if src_shape.is_vector() {
        // If from is a vector and the first dimension is the broadcasting dimension
        if src_shape[0] > 1 && src_shape[0] == dst_shape[0] {
            let chunk_size = result.len() / src_shape.numel();

            (0..result.len())
                .step_by(chunk_size)
                .enumerate()
                .for_each(|(i, chunk)| {
                    result[chunk..chunk + chunk_size].fill(src[i]);
                });
        } else {
            for i in 0..result.len() {
                let dst_index = offset_to_ndindex(i, &dst_strides);
                let src_index: Vec<usize> = (0..src_shape.len())
                    .map(|x| if src_shape[x] == 1 { 0 } else { dst_index[x] })
                    .collect();
                let src_offset = nd_index_to_offset(&src_index, &src_strides);
                result[i] = src[src_offset]
            }
        }
    } else {
        for i in 0..result.len() {
            let dst_index = offset_to_ndindex(i, &dst_strides);
            let src_index: Vec<usize> = (0..src_shape.len())
                .map(|x| if src_shape[x] == 1 { 0 } else { dst_index[x] })
                .collect();
            let src_offset = nd_index_to_offset(&src_index, &src_strides);
            result[i] = src[src_offset]
        }
    }

    result
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

    let mut dst_dots = vec![];
    for d in 0..dst_shape.len() {
        dst_dots.push(dst_shape[d + 1..].iter().product::<usize>().max(1));
    }

    for i in 0..dst.len() {
        let mut src_index = 0;
        let mut tmp = i;
        for d in 0..dst_shape.len() {
            let coord = tmp / dst_dots[d];
            tmp %= dst_dots[d];
            src_index += (coord + start[d]) * src_strides[d] as usize;
        }
        dst[i] = src[src_index];
    }

    dst
}
