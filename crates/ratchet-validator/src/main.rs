use ratchet::UnaryOp;
use serde::Serialize;
use strum::IntoEnumIterator;
use tch::Tensor;

#[derive(Serialize)]
struct TestCase {
    name: String,
    op: String,
    inputs: Vec<f32>,
    outputs: Vec<f32>,
}

fn generate_input_pair<S: torch_sys::IntList>(shape: S) -> (Tensor, Vec<f32>) {
    let input = Tensor::f_randn(shape, (tch::Kind::Float, tch::Device::Cpu)).unwrap();
    let mut copy = Tensor::empty_like(&input);
    copy.copy_(&input);
    let inputv: Vec<f32> = Vec::try_from(copy.reshape(-1)).unwrap();
    (input, inputv)
}

fn generate_unary<S: torch_sys::IntList>(r_op: UnaryOp, shape: S) -> TestCase {
    let (input, inputv) = generate_input_pair(shape);

    let result = match r_op {
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
        UnaryOp::Neg => input.neg(),
        UnaryOp::Silu => input.silu(),
        UnaryOp::Sigmoid => input.sigmoid(),
    };

    let output: Vec<f32> = Vec::try_from(result.reshape(-1)).unwrap();
    let n = r_op.kernel_name().to_string();
    TestCase {
        name: n.clone(),
        op: n.clone(),
        inputs: inputv,
        outputs: output,
    }
}

//1. Write a macro to do all boilerplate at least for unary ops
//2. Cos, Abs etc

// If you have random inputs
// Then CI may randomly fail when you generate an input that finds a bug
fn main() {
    let unary_cases = UnaryOp::iter()
        .map(|op| {
            let test_case = generate_unary(op, &[128, 128]);
            serde_json::to_string(&test_case).unwrap()
        })
        .collect::<Vec<_>>();
}

//Run the test suite, and generate the JSON locally
//
//    {
//       "cases": [
//            {
//                name: "abs-f16",
//                op: "abs",
//                "inputs": [.......],
//                "outputs": [.......]
//            }
//        ]
//    }
//golden_truth.json
