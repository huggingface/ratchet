use crate::{
    binary_apply_inplace, Binary, BinaryOp, DType, OpGuards, Operation, OperationError, RVec,
    StorageView, Tensor, TensorDType,
};
use core::marker::PhantomData;
use half::{bf16, f16};

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
            fn apply_cpu(&self, dst: Tensor) -> Result<Tensor, OperationError> {
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
        DType::F32 => CPU::<f32, _>::new(binary).apply_cpu(dst),
        DType::F16 => CPU::<f16, _>::new(binary).apply_cpu(dst),
        DType::BF16 => CPU::<bf16, _>::new(binary).apply_cpu(dst),
        _ => todo!(),
    }
}
