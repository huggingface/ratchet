use serde::Serialize;
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

// If you have random inputs
// Then CI may randomly fail when you generate an input that finds a bug
fn main() {
    let (input, inputv) = generate_input_pair(&[128, 128]);
    let output: Vec<f32> = Vec::try_from(input.abs().reshape(-1)).unwrap();
    let test_case = TestCase {
        name: "abs-f32".to_string(),
        op: "abs".to_string(),
        inputs: inputv,
        outputs: output,
    };
    let json = serde_json::to_string(&test_case).unwrap();
    println!("{}", json);
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
