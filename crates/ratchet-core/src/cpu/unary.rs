use crate::{
    unary_apply_fn, CPUOperation, DType, OperationError, Tensor, TensorDType, Unary, UnaryOp,
};
use core::marker::PhantomData;
use half::{bf16, f16};
use num_traits::Float;

struct UnaryOps<T: TensorDType> {
    dtype: PhantomData<T>,
}
macro_rules! impl_unary_ops {
    ($dtype:ident, $conv:expr) => {
        impl UnaryOps<$dtype> {
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

            fn apply(op: &Unary, dst: Tensor) -> Result<Tensor, OperationError> {
                match op.op() {
                    UnaryOp::Gelu => Self::gelu(op.input(), dst),
                    UnaryOp::Tanh => Self::tanh(op.input(), dst),
                    UnaryOp::Exp => Self::exp(op.input(), dst),
                    UnaryOp::Log => Self::log(op.input(), dst),
                    UnaryOp::Sin => Self::sin(op.input(), dst),
                    UnaryOp::Cos => Self::cos(op.input(), dst),
                    UnaryOp::Abs => Self::abs(op.input(), dst),
                    UnaryOp::Sqrt => Self::sqrt(op.input(), dst),
                    UnaryOp::Relu => Self::relu(op.input(), dst),
                    UnaryOp::Floor => Self::floor(op.input(), dst),
                    UnaryOp::Ceil => Self::ceil(op.input(), dst),
                    UnaryOp::Neg => Self::neg(op.input(), dst),
                    UnaryOp::Silu => Self::silu(op.input(), dst),
                    UnaryOp::Sigmoid => Self::sigmoid(op.input(), dst),
                }
            }
        }
    };
}

macro_rules! impl_cpu_unary_op {
    ($method_name:ident, $op:expr) => {
        fn $method_name(input: &Tensor, dst: Tensor) -> Result<Tensor, OperationError> {
            unary_apply_fn(input, &dst, $op)?;
            Ok(dst)
        }
    };
}

impl CPUOperation for Unary {
    fn apply_cpu(&self, dst: Tensor) -> Result<Tensor, OperationError> {
        match dst.dt() {
            DType::F32 => UnaryOps::<f32>::apply(self, dst),
            DType::F16 => UnaryOps::<f16>::apply(self, dst),
            DType::BF16 => UnaryOps::<bf16>::apply(self, dst),
            _ => todo!(),
        }
    }
}

macro_rules! impl_cpu_unary {
    ($dtype:ident) => {
        impl_cpu_unary!($dtype, |x| x);
    };
    ($dtype:ident, $conv:expr) => {
        impl_unary_ops!($dtype, $conv);
    };
}

impl_cpu_unary!(f32);
impl_cpu_unary!(f16, f16::from_f32);
impl_cpu_unary!(bf16, bf16::from_f32);
