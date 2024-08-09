use ratchet::{Tensor, UnaryOp};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use strum::IntoEnumIterator;
use tch::{Device, Tensor as TchTensor};

#[derive(Serialize, Deserialize)]
struct TestCase {
    op: String,
    inputs: Vec<Tensor>,
    outputs: Vec<Tensor>,
    atol: f64,
    rtol: f64,
}

fn apply_unary_op(op: &UnaryOp, input: &TchTensor) -> Tensor {
    match op {
        UnaryOp::Gelu => input.gelu("tanh"),
        UnaryOp::Tanh => input.tanh(),
        UnaryOp::Exp => input.exp(),
        UnaryOp::Log => input.log(),
        UnaryOp::Sin => input.sin(),
        UnaryOp::Cos => input.cos(),
        UnaryOp::Abs => input.abs(),
        UnaryOp::Sqrt => input.sqrt(),
        UnaryOp::Relu => input.relu(),
        UnaryOp::Floor => input.floor(),
        UnaryOp::Ceil => input.ceil(),
        UnaryOp::Neg => -input,
        UnaryOp::Silu => input.silu(),
        UnaryOp::Sigmoid => input.sigmoid(),
    }
}

fn tensor_to_vec(t: &TchTensor) -> Vec<f32> {
    let size = t.size1().unwrap();
    (0..size).map(|i| t.double_value(&[i]) as f32).collect()
}

fn generate_test_case(op: &UnaryOp, tol: f64) -> TestCase {
    let input = TchTensor::randn(&[10], (tch::Kind::Float, Device::Cpu));
    let output = apply_unary_op(op, &input);

    TestCase {
        op: format!("{:?}", op),
        inputs: vec![input],
        outputs: vec![output],
        atol: tol,
        rtol: tol,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for op in UnaryOp::iter() {
        let test_case = generate_test_case(&op, 1e-3);
        let json = serde_json::to_string_pretty(&test_case)?;

        let filename = format!("{:?}_test.json", op);
        let mut file = File::create(filename)?;
        file.write_all(json.as_bytes())?;
    }

    println!("Test cases generated successfully!");
    Ok(())
}
