use crate::{
    Binary, BinaryOp, CPUBuffer, CPUOperation, Cast, DType, IndexSelect, InvariantError, OpGuards,
    Operation, OperationError, RVec, Storage, StorageView, Tensor, TensorDType, Unary, UnaryOp,
};
use anyhow::anyhow;
use bytemuck::NoUninit;
use core::marker::PhantomData;
use half::{bf16, f16};
use num_traits::Float;

#[derive(Debug)]
pub struct CPU<T: TensorDType, OP: Operation> {
    op: OP,
    dtype: PhantomData<T>,
}

impl<T: TensorDType, OP: Operation> CPU<T, OP> {
    pub fn new(op: OP) -> Self {
        Self {
            op,
            dtype: PhantomData,
        }
    }
}

impl<T: TensorDType, OP: Operation> OpGuards for CPU<T, OP> {
    fn check_shapes(&self) {
        self.op.check_shapes();
    }

    fn check_dtypes(&self) {
        self.op.check_dtypes();
    }
}

impl<T: TensorDType, OP: Operation> Operation for CPU<T, OP> {
    fn name(&self) -> &'static str {
        self.op.name()
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        self.op.compute_view()
    }

    fn srcs(&self) -> RVec<&Tensor> {
        self.op.srcs()
    }
}

macro_rules! impl_cpu_unary_op {
    ($method_name:ident, $op:expr) => {
        fn $method_name(input: &Tensor, dst: Tensor) -> Result<Tensor, OperationError> {
            unary_apply_fn(input, &dst, $op)?;
            Ok(dst)
        }
    };
}

macro_rules! impl_cpu_unary_wrapper {
    ($dtype:ident, $conv:expr) => {
        impl CPU<$dtype, Unary> {
            impl_cpu_unary_op!(gelu, |x: $dtype| $conv(0.5)
                * x
                * ($conv(1.0)
                    + $dtype::tanh(
                        $conv(0.797_884_6) * x * ($conv(1.0) + $conv(0.044715) * x * x)
                    )));

            impl_cpu_unary_op!(tanh, |x: $dtype| x.tanh());
            impl_cpu_unary_op!(exp, |x: $dtype| x.exp());
            impl_cpu_unary_op!(log, |x: $dtype| x.ln());
            impl_cpu_unary_op!(sin, |x: $dtype| x.sin());
            impl_cpu_unary_op!(cos, |x: $dtype| x.cos());
            impl_cpu_unary_op!(abs, |x: $dtype| x.abs());
            impl_cpu_unary_op!(sqrt, |x: $dtype| x.sqrt());
            impl_cpu_unary_op!(relu, |x: $dtype| x.max($conv(0.0)));
            impl_cpu_unary_op!(floor, |x: $dtype| x.floor());
            impl_cpu_unary_op!(ceil, |x: $dtype| x.ceil());
            impl_cpu_unary_op!(neg, |x: $dtype| -x);
            impl_cpu_unary_op!(silu, |x: $dtype| x / ($conv(1.0) + (-x).exp()));
            impl_cpu_unary_op!(sigmoid, |x: $dtype| $conv(1.0) / ($conv(1.0) + (-x).exp()));
        }
    };
}

macro_rules! impl_cpu_unary {
    ($dtype:ident) => {
        impl_cpu_unary!($dtype, |x| x);
    };
    ($dtype:ident, $conv:expr) => {
        impl_cpu_unary_wrapper!($dtype, $conv);

        impl CPUOperation for CPU<$dtype, Unary> {
            fn apply(&self, dst: Tensor) -> Result<Tensor, OperationError> {
                match self.op.op() {
                    UnaryOp::Gelu => Self::gelu(self.op.input(), dst),
                    UnaryOp::Tanh => Self::tanh(self.op.input(), dst),
                    UnaryOp::Exp => Self::exp(self.op.input(), dst),
                    UnaryOp::Log => Self::log(self.op.input(), dst),
                    UnaryOp::Sin => Self::sin(self.op.input(), dst),
                    UnaryOp::Cos => Self::cos(self.op.input(), dst),
                    UnaryOp::Abs => Self::abs(self.op.input(), dst),
                    UnaryOp::Sqrt => Self::sqrt(self.op.input(), dst),
                    UnaryOp::Relu => Self::relu(self.op.input(), dst),
                    UnaryOp::Floor => Self::floor(self.op.input(), dst),
                    UnaryOp::Ceil => Self::ceil(self.op.input(), dst),
                    UnaryOp::Neg => Self::neg(self.op.input(), dst),
                    UnaryOp::Silu => Self::silu(self.op.input(), dst),
                    UnaryOp::Sigmoid => Self::sigmoid(self.op.input(), dst),
                }
            }
        }
    };
}

impl_cpu_unary!(f32);
impl_cpu_unary!(f16, f16::from_f32);
impl_cpu_unary!(bf16, bf16::from_f32);

pub fn cpu_unary(unary: Unary, dst: Tensor) -> Result<Tensor, OperationError> {
    match dst.dt() {
        DType::F32 => CPU::<f32, _>::new(unary).apply(dst),
        DType::F16 => CPU::<f16, _>::new(unary).apply(dst),
        DType::BF16 => CPU::<bf16, _>::new(unary).apply(dst),
        _ => todo!(),
    }
}

macro_rules! impl_cpu_binary_op {
    ($method_name:ident, $dtype:ident, $op:expr) => {
        fn $method_name(lhs: &Tensor, rhs: &Tensor, dst: Tensor) -> Result<Tensor, OperationError> {
            binary_apply_inplace::<$dtype>(lhs, rhs, &dst, $op)?;
            Ok(dst)
        }
    };
}

macro_rules! impl_cpu_binary {
    ($dtype:ident) => {
        impl CPU<$dtype, Binary> {
            impl_cpu_binary_op!(add, $dtype, |lhs, rhs| lhs + rhs);
            impl_cpu_binary_op!(sub, $dtype, |lhs, rhs| lhs - rhs);
            impl_cpu_binary_op!(mul, $dtype, |lhs, rhs| lhs * rhs);
            impl_cpu_binary_op!(div, $dtype, |lhs, rhs| lhs / rhs);
        }

        impl CPUOperation for CPU<$dtype, Binary> {
            fn apply(&self, dst: Tensor) -> Result<Tensor, OperationError> {
                match self.op.op() {
                    BinaryOp::Add => Self::add(self.op.lhs(), self.op.rhs(), dst),
                    BinaryOp::Sub => Self::sub(self.op.lhs(), self.op.rhs(), dst),
                    BinaryOp::Mul => Self::mul(self.op.lhs(), self.op.rhs(), dst),
                    BinaryOp::Div => Self::div(self.op.lhs(), self.op.rhs(), dst),
                }
            }
        }
    };
}

impl_cpu_binary!(f32);
impl_cpu_binary!(f16);
impl_cpu_binary!(bf16);

pub fn cpu_binary(binary: Binary, dst: Tensor) -> Result<Tensor, OperationError> {
    match dst.dt() {
        DType::F32 => CPU::<f32, _>::new(binary).apply(dst),
        DType::F16 => CPU::<f16, _>::new(binary).apply(dst),
        DType::BF16 => CPU::<bf16, _>::new(binary).apply(dst),
        _ => todo!(),
    }
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
        for i in 0..n_ids {
            let src_idx = start_src_idx + indices[i] as usize * right_len;
            let dst_idx = start_dst_idx + i * right_len;
            result[dst_idx..dst_idx + right_len]
                .copy_from_slice(&src[src_idx..src_idx + right_len]);
        }
    }
    cpu_store_result(&dst, &result);
    Ok(dst)
}

pub fn cpu_index_select(i: IndexSelect, dst: Tensor) -> Result<Tensor, OperationError> {
    match dst.dt() {
        DType::F32 => index_select::<f32>(i, dst),
        DType::F16 => index_select::<f16>(i, dst),
        DType::BF16 => index_select::<bf16>(i, dst),
        _ => todo!(),
    }
}

fn direct_cast<T: TensorDType, U: TensorDType>(
    input: &Tensor,
    dst: &Tensor,
) -> Result<(), OperationError> {
    let input = input.to_vec::<T>()?;
    let result =
        bytemuck::try_cast_slice::<T, U>(&input).map_err(|_| anyhow!("Failed direct cast"))?;
    cpu_store_result(dst, &result);
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
