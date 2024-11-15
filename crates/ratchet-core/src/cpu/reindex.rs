use super::utils::{
    cpu_store_result, TensorIterator,
    TensorIterator::{Contiguous, Strided},
};
use crate::{
    Broadcast, CPUOperation, DType, OperationError, Permute, Reindex, Shape, Slice, Strides,
    Tensor, TensorDType,
};
use half::{bf16, f16};
use ndarray::ShapeBuilder;
use std::ops::Range;

impl CPUOperation for Reindex {
    fn apply_cpu(&self, dst: Tensor) -> Result<Tensor, OperationError> {
        match self {
            Reindex::Permute(p) => p.apply_cpu(dst),
            Reindex::Slice(s) => s.apply_cpu(dst),
            Reindex::Broadcast(b) => b.apply_cpu(dst),
            _ => todo!(),
        }
    }
}

impl CPUOperation for Permute {
    fn apply_cpu(&self, dst: Tensor) -> Result<Tensor, OperationError> {
        match dst.dt() {
            DType::F32 => apply_permute::<f32>(self, dst),
            DType::BF16 => apply_permute::<bf16>(self, dst),
            DType::F16 => apply_permute::<f16>(self, dst),
            DType::I32 => apply_permute::<i32>(self, dst),
            DType::U32 => apply_permute::<u32>(self, dst),
            _ => todo!(),
        }
    }
}

fn apply_permute<T: TensorDType>(p: &Permute, dst: Tensor) -> Result<Tensor, OperationError> {
    let perm: [usize; 4] = p.promote().try_into().unwrap();
    let Permute { src, dims } = p;
    let result = permute(&src.to_vec::<T>()?, src.shape(), dst.shape(), perm);
    cpu_store_result(&dst, &result);
    Ok(dst)
}

// TODO: Optimize.
// This generic implementation is almost a direct copy from the gpu impl,
// and can definitely be way more performant.
fn permute<T: TensorDType>(
    src: &[T],
    src_shape: &Shape,
    dst_shape: &Shape,
    perm: [usize; 4],
) -> Vec<T> {
    let mut result = vec![T::zero(); src_shape.numel()];

    // We now know that these will always be len 4, same as gpu impl.
    let src_shape = &Shape::promote(src_shape.clone(), 4);
    let dst_shape = &Shape::promote(dst_shape.clone(), 4);

    let src_strides = &Strides::from(src_shape);
    let dst_strides = &Strides::from(dst_shape);

    let src_shape: [usize; 4] = src_shape.try_into().unwrap();
    let src_strides: [usize; 4] = src_strides.try_into().unwrap();
    let dst_strides: [usize; 4] = dst_strides.try_into().unwrap();

    for i in 0..result.len() {
        let dst_index = offset_to_ndindex(i, dst_strides);
        let mut src_index = [0; 4];
        src_index[perm[0]] = dst_index[0];
        src_index[perm[1]] = dst_index[1];
        src_index[perm[2]] = dst_index[2];
        src_index[perm[3]] = dst_index[3];
        let src_offset = nd_index_to_offset(src_index, src_strides);
        result[i] = src[src_offset]
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

pub(crate) fn broadcast<T: TensorDType>(src: &[T], src_shape: &Shape, dst_shape: &Shape) -> Vec<T> {
    let mut result = vec![T::zero(); dst_shape.numel()];

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
            generic_broadcast(src, &mut result, src_shape, dst_shape)
        }
    } else {
        generic_broadcast(src, &mut result, src_shape, dst_shape)
    }

    result
}

// TODO: Optimize.
// This generic implementation is almost a direct copy from the gpu impl,
// and can definitely be way more performant.
fn generic_broadcast<T: TensorDType>(
    src: &[T],
    result: &mut [T],
    src_shape: &Shape,
    dst_shape: &Shape,
) {
    // We now know that these will always be len 4, same as gpu impl.
    let src_shape = &Shape::promote(src_shape.clone(), 4);
    let dst_shape = &Shape::promote(dst_shape.clone(), 4);

    let src_strides = &Strides::from(src_shape);
    let dst_strides = &Strides::from(dst_shape);

    let src_shape: [usize; 4] = src_shape.try_into().unwrap();
    let src_strides: [usize; 4] = src_strides.try_into().unwrap();
    let dst_strides: [usize; 4] = dst_strides.try_into().unwrap();

    fn select(a: [usize; 4], b: [usize; 4], t: [bool; 4]) -> [usize; 4] {
        let mut result = [0; 4];
        result[0] = if t[0] { a[0] } else { b[0] };
        result[1] = if t[1] { a[1] } else { b[1] };
        result[2] = if t[2] { a[2] } else { b[2] };
        result[3] = if t[3] { a[3] } else { b[3] };
        result
    }

    let shape_onedim_lookup: [bool; 4] = [
        src_shape[0] != 1,
        src_shape[1] != 1,
        src_shape[2] != 1,
        src_shape[3] != 1,
    ];
    for i in 0..result.len() {
        let dst_index = offset_to_ndindex(i, dst_strides);
        let src_index = select(dst_index, [0; 4], shape_onedim_lookup);
        let src_offset = nd_index_to_offset(src_index, src_strides);
        result[i] = src[src_offset]
    }
}

#[inline]
fn offset_to_ndindex(offset: usize, strides: [usize; 4]) -> [usize; 4] {
    let mut indices = [0; 4];
    let mut remaining = offset;

    let idx = remaining / strides[0];
    indices[0] = idx;
    remaining -= idx * strides[0];

    let idx = remaining / strides[1];
    indices[1] = idx;
    remaining -= idx * strides[1];

    let idx = remaining / strides[2];
    indices[2] = idx;
    remaining -= idx * strides[2];

    indices[3] = remaining;
    indices
}

#[inline]
fn nd_index_to_offset(ndindex: [usize; 4], strides: [usize; 4]) -> usize {
    ndindex[0] * strides[0]
        + ndindex[1] * strides[1]
        + ndindex[2] * strides[2]
        + ndindex[3] * strides[3]
}
