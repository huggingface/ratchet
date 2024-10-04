mod binary;
pub mod gemm;
mod unary;

use crate::{
    dequantize, Binary, BinaryOp, CPUBuffer, Cast, Concat, DType, IndexSelect, InvariantError,
    LazyOp, OpGuards, Operation, OperationError, RVec, Storage, StorageView, Tensor, TensorDType,
};
use anyhow::anyhow;
use bytemuck::NoUninit;
use core::marker::PhantomData;
use half::{bf16, f16};

pub fn cpu_binary(binary: Binary, dst: Tensor) -> Result<Tensor, OperationError> {
    match dst.dt() {
        DType::F32 => binary::CPU::<f32, _>::new(binary).apply_cpu(dst),
        DType::F16 => binary::CPU::<f16, _>::new(binary).apply_cpu(dst),
        DType::BF16 => binary::CPU::<bf16, _>::new(binary).apply_cpu(dst),
        _ => todo!(),
    }
}

pub fn apply_operation(op: LazyOp, dst: Tensor) -> Result<Tensor, OperationError> {
    match op {
        LazyOp::Binary(b) => cpu_binary(b, dst),
        LazyOp::Cast(c) => cpu_cast(c, dst),
        LazyOp::Matmul(m) => m.apply_cpu(dst),
        LazyOp::Softmax(_s) => todo!(),
        LazyOp::RoPE(_r) => todo!(),
        LazyOp::Unary(u) => u.apply_cpu(dst),
        LazyOp::Reindex(_r) => todo!(),
        LazyOp::Concat(c) => cpu_concat(c, dst),
        LazyOp::Norm(_n) => todo!(),
        LazyOp::Conv(_c) => todo!(),
        LazyOp::Select(i) => cpu_index_select(i, dst),
        LazyOp::IndexWrite(_i) => todo!(),
        LazyOp::Cache(_c) => todo!(),
        LazyOp::Const => todo!(),
        LazyOp::View(_) => todo!(),
    }
}

pub trait CPUOperation: Operation {
    fn apply_cpu(&self, dst: Tensor) -> Result<Tensor, OperationError>;
}

fn index_select<T: TensorDType>(
    index_select: IndexSelect,
    dst: Tensor,
) -> Result<Tensor, OperationError> {
    let src = index_select.src();
    let indices = index_select.indices();
    let dim = index_select.dim();

    // TODO: Add support for other indexing types
    if !matches!(indices.dt(), DType::I32) {
        return Err(InvariantError::DTypeMismatch {
            expected: DType::I32,
            actual: indices.dt(),
        }
        .into());
    }

    let mut dst_dims = src.shape().to_vec();
    let indices_dims = indices.shape().to_vec();

    let src_dim = dst_dims[dim];
    let n_ids = indices_dims[0];
    dst_dims[dim] = n_ids;

    let dst_len: usize = dst_dims.iter().product();
    let left_len: usize = dst_dims[..dim].iter().product();
    let right_len: usize = dst_dims[dim + 1..].iter().product();

    let src = src.to_vec::<T>()?;
    let indices = indices.to_vec::<i32>()?;
    let mut result = vec![T::zero(); dst_len];

    for left_i in 0..left_len {
        let start_src_idx = left_i * right_len * src_dim;
        let start_dst_idx = left_i * right_len * n_ids;
        for (i, idx) in indices.iter().enumerate().take(n_ids) {
            let src_idx = start_src_idx + *idx as usize * right_len;
            let dst_idx = start_dst_idx + i * right_len;
            result[dst_idx..dst_idx + right_len]
                .copy_from_slice(&src[src_idx..src_idx + right_len]);
        }
    }
    cpu_store_result(&dst, &result);
    Ok(dst)
}

fn qindex_select(op: IndexSelect, dst: Tensor) -> Result<Tensor, OperationError> {
    // NOTE: qindex_select is functional but not optimized at all.
    // Currently we simply dequantize the entire input tensor to f32 and then call index_select.
    // Because of borrowing rules dequantizing also requires a deep clone of the input tensor, which is less than ideal.
    // In the future we would rather directly index the raw buffer of the quantized tensor and dequantize only what is required.
    // TODO: Add support for direct indexing + partial dequantization
    let src = op.src().deep_clone();

    // NOTE: Support for other quantization types is dependent on the corresponding dequantization functions.
    let src = dequantize(src);
    let indices = op.indices().clone();
    let dim = op.dim();

    index_select::<f32>(IndexSelect::new(src, indices, dim), dst)
}

pub fn cpu_index_select(i: IndexSelect, dst: Tensor) -> Result<Tensor, OperationError> {
    match i.src().dt() {
        DType::F32 => index_select::<f32>(i, dst),
        DType::F16 => index_select::<f16>(i, dst),
        DType::BF16 => index_select::<bf16>(i, dst),
        DType::Q8_0F(_) => qindex_select(i, dst),
        dtype => Err(InvariantError::UnsupportedDType(dtype).into()),
    }
}

fn direct_cast<T: TensorDType, U: TensorDType>(
    input: &Tensor,
    dst: &Tensor,
) -> Result<(), OperationError> {
    let input = input.to_vec::<T>()?;
    let result =
        bytemuck::try_cast_slice::<T, U>(&input).map_err(|_| anyhow!("Failed direct cast"))?;
    cpu_store_result(dst, result);
    Ok(())
}

pub fn cpu_cast(cast: Cast, dst: Tensor) -> Result<Tensor, OperationError> {
    if cast.input().dt() == cast.dst_dt() {
        return Ok(cast.input().clone());
    }
    match (cast.input().dt(), cast.dst_dt()) {
        // F32 ->
        (DType::F32, DType::F16) => unary_apply_fn::<f32, f16>(cast.input(), &dst, f16::from_f32)?,
        (DType::F32, DType::BF16) => {
            unary_apply_fn::<f32, bf16>(cast.input(), &dst, bf16::from_f32)?
        }
        (DType::F32, DType::I32) => direct_cast::<f32, i32>(cast.input(), &dst)?,
        (DType::F32, DType::U32) => direct_cast::<f32, u32>(cast.input(), &dst)?,

        // F16 ->
        (DType::F16, DType::F32) => unary_apply_fn::<f16, f32>(cast.input(), &dst, f32::from)?,

        // BF16 ->
        (DType::BF16, DType::F32) => unary_apply_fn::<bf16, f32>(cast.input(), &dst, f32::from)?,

        // I32 ->
        (DType::I32, DType::F32) => direct_cast::<i32, f32>(cast.input(), &dst)?,

        // U32 ->
        (DType::U32, DType::F32) => direct_cast::<u32, f32>(cast.input(), &dst)?,

        _ => unimplemented!("Cannot cast {:?} -> {:?}", cast.input().dt(), cast.dst_dt()),
    };

    Ok(dst)
}

fn concat_inner<T: TensorDType>(
    inputs: RVec<Tensor>,
    dim: usize,
    dst: Tensor,
) -> Result<Tensor, OperationError> {
    let dst_size = dst.shape().clone().product();
    let mut result = vec![T::zero(); dst_size];

    let dst_dim_len = dst.shape()[dim];
    let block: usize = dst.shape().iter().skip(1 + dim).product();
    let dst_s = block * dst_dim_len;
    let src_o = 0;
    let mut dst_o = 0;
    for t in inputs {
        let src = t.to_vec::<T>()?;

        let t_dims = t.shape().as_slice();
        let a_dim: usize = t_dims.iter().take(dim).product();
        let b_dim = block * t_dims[dim];

        for idx in 0..a_dim {
            let dst_idx = idx * dst_s + dst_o;
            let src_idx = idx * b_dim + src_o;
            let dst = &mut result[dst_idx..dst_idx + b_dim];
            let src = &src[src_idx..src_idx + b_dim];
            dst.copy_from_slice(src)
        }
        dst_o += b_dim;
    }
    cpu_store_result(&dst, &result);
    Ok(dst)
}

pub fn cpu_concat(Concat { inputs, dim }: Concat, dst: Tensor) -> Result<Tensor, OperationError> {
    match dst.dt() {
        DType::F32 => concat_inner::<f32>(inputs, dim, dst),
        DType::F16 => concat_inner::<f16>(inputs, dim, dst),
        DType::BF16 => concat_inner::<bf16>(inputs, dim, dst),
        dtype => Err(InvariantError::UnsupportedDType(dtype).into()),
    }
}

#[inline]
fn unary_apply_fn_helper<T: TensorDType, U: TensorDType>(src: &[T], dst: &mut [U], f: fn(T) -> U) {
    assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().copied().zip(dst.iter_mut()) {
        *d = f(s);
    }
}

#[inline]
pub fn unary_apply_fn<T: TensorDType, U: TensorDType>(
    input: &Tensor,
    dst: &Tensor,
    f: fn(T) -> U,
) -> Result<(), OperationError> {
    let input = input.to_vec::<T>()?;
    let mut result = vec![U::zero(); dst.shape().numel()];
    unary_apply_fn_helper(&input, &mut result, f);
    cpu_store_result(dst, &result);
    Ok(())
}

#[inline]
fn binary_apply_fn_helper<T: TensorDType, U: TensorDType>(
    lhs: &[T],
    rhs: &[T],
    dst: &mut [U],
    f: fn(T, T) -> U,
) {
    assert_eq!(lhs.len(), dst.len());
    assert_eq!(rhs.len(), dst.len());
    for ((l, r), d) in lhs
        .iter()
        .copied()
        .zip(rhs.iter().copied())
        .zip(dst.iter_mut())
    {
        *d = f(l, r);
    }
}

#[inline]
fn binary_apply_inplace_helper<T: TensorDType>(lhs: &mut [T], rhs: &[T], f: fn(T, T) -> T) {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter_mut().zip(rhs.iter()).for_each(|(l, r)| {
        *l = f(*l, *r);
    });
}

#[inline]
pub fn binary_apply_fn<T: TensorDType, U: TensorDType>(
    lhs: &Tensor,
    rhs: &Tensor,
    dst: &Tensor,
    f: fn(T, T) -> U,
) -> Result<(), OperationError> {
    let lhs = lhs.to_vec::<T>()?;
    let rhs = rhs.to_vec::<T>()?;
    let mut result = vec![U::zero(); dst.shape().numel()];
    binary_apply_fn_helper(&lhs, &rhs, &mut result, f);
    cpu_store_result(dst, &result);
    Ok(())
}

#[inline]
pub fn binary_apply_inplace<T: TensorDType>(
    lhs: &Tensor,
    rhs: &Tensor,
    dst: &Tensor,
    f: fn(T, T) -> T,
) -> Result<(), OperationError> {
    let mut lhs = lhs.to_vec::<T>()?;
    let rhs = rhs.to_vec::<T>()?;
    binary_apply_inplace_helper(&mut lhs, &rhs, f);
    cpu_store_result(dst, &lhs);
    Ok(())
}

pub fn cpu_store_result<T: NoUninit>(dst: &Tensor, data: &[T]) {
    dst.update_storage(Storage::CPU(CPUBuffer::from_slice(data, dst.shape())));
}
