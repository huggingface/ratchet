use crate::{
    binary_apply_inplace, Binary, BinaryOp, CPUOperation, DType, OpGuards, Operation,
    OperationError, RVec, StorageView, Tensor, TensorDType,
};
use core::marker::PhantomData;
use half::{bf16, f16};

pub struct BinaryOps<T: TensorDType> {
    dtype: PhantomData<T>,
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
        impl BinaryOps<$dtype> {
            impl_cpu_binary_op!(add, $dtype, |lhs, rhs| lhs + rhs);
            impl_cpu_binary_op!(sub, $dtype, |lhs, rhs| lhs - rhs);
            impl_cpu_binary_op!(mul, $dtype, |lhs, rhs| lhs * rhs);
            impl_cpu_binary_op!(div, $dtype, |lhs, rhs| lhs / rhs);

            pub fn apply(op: &Binary, dst: Tensor) -> Result<Tensor, OperationError> {
                match op.op() {
                    BinaryOp::Add => Self::add(op.lhs(), op.rhs(), dst),
                    BinaryOp::Sub => Self::sub(op.lhs(), op.rhs(), dst),
                    BinaryOp::Mul => Self::mul(op.lhs(), op.rhs(), dst),
                    BinaryOp::Div => Self::div(op.lhs(), op.rhs(), dst),
                }
            }
        }
    };
}

impl CPUOperation for Binary {
    fn apply_cpu(&self, dst: Tensor) -> Result<Tensor, OperationError> {
        match dst.dt() {
            DType::F32 => BinaryOps::<f32>::apply(self, dst),
            DType::F16 => BinaryOps::<f16>::apply(self, dst),
            DType::BF16 => BinaryOps::<bf16>::apply(self, dst),
            _ => todo!(),
        }
    }
}

impl_cpu_binary!(f32);
impl_cpu_binary!(f16);
impl_cpu_binary!(bf16);
