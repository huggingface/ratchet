use super::utils::{cpu_store_result, StridedIterator};
use crate::{
    CPUOperation, DType, OperationError, Permute, Reindex, Shape, Slice, Strides, Tensor,
    TensorDType,
};
use half::{bf16, f16};
use ndarray::ShapeBuilder;
use pyo3::ffi::PyExc_FutureWarning;
use std::ops::Range;

impl CPUOperation for Reindex {
    fn apply_cpu(&self, dst: Tensor) -> Result<Tensor, OperationError> {
        match self {
            Reindex::Permute(p) => {
                println!("Permute: {:?}", p);
                p.apply_cpu(dst)
            }
            Reindex::Slice(s) => s.apply_cpu(dst),
            _ => todo!(),
        }
    }
}

impl CPUOperation for Permute {
    fn apply_cpu(&self, dst: Tensor) -> Result<Tensor, OperationError> {
        match dst.dt() {
            DType::F32 => apply_permute::<f32>(self, dst),
            _ => todo!(),
        }
    }
}

fn apply_permute<T: TensorDType>(
    Permute { src, dims }: &Permute,
    dst: Tensor,
) -> Result<Tensor, OperationError> {
    let result = permute(&src.to_vec::<T>()?, src.shape(), dims);
    cpu_store_result(&dst, &result);
    Ok(dst)
}

fn get_strided_index(idx: usize, num_dims: usize, dims: &[usize], strides: &[isize]) -> usize {
    let mut idx = idx; // 2
    let mut strided_i: usize = 0;
    println!("strides: {strides:?}");
    println!("dims: {dims:?}");
    print!("{idx} -> ");
    for d in 0..num_dims {
        let dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx] as usize;
        idx /= dims[dim_idx];
        print!("{idx}|{dim_idx}|{strided_i}, ");
    }
    print!("\n");
    return strided_i;
}

pub(crate) fn permute<T: TensorDType>(src: &[T], shape: &Shape, dims: &[usize]) -> Vec<T> {
    /*
    // simplify shape
    // 1. remove dimensions with size 0..1
    // 2. consecutive dimensions can be merged
    // 3. remove dimensions that are not in the permutation
    let mut ranges: Vec<Range<usize>> = vec![];
    let mut dims_s: Vec<usize> = vec![];

    let mut start = 0;
    let mut end = 0;
    for i in 0..dims.len() - 1 {
        if dims[i] + 1 == dims[i + 1] {
            end = i;
        } else {
            ranges.push(start..end);
            start = end;
        }
    }
    ranges.push(start..);

    println!("ranges: {:?}", ranges);
    println!("dims_s: {:?}", dims_s);

    if ranges.len() <= 1 {
        return src.to_vec();
    }

    */
    let mut dst = vec![T::zero(); shape.numel()];

    let strides = Strides::from(shape).to_vec();

    let mut p_shape = vec![0; shape.rank()];
    for i in 0..shape.rank() {
        p_shape[dims[i]] = shape[i];
    }

    for i in 0..src.len() {
        let strided_idx = get_strided_index(i, dims.len(), p_shape.as_slice(), &strides);
        println!("{i} -> {strided_idx}");
        dst[i] = src[strided_idx];
    }
    dst
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
