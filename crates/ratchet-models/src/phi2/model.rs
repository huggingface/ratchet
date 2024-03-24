#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use numpy::PyArrayDyn;
    use pyo3::{types::PyModule, Python};
    use ratchet::Tensor;

    fn ground_truth() -> anyhow::Result<Vec<Tensor>> {
        let prg = r#"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def ground():
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

    inputs = tokenizer('''def print_prime(n):
    """
    Print all primes between 1 and n
    """''', return_tensors="pt", return_attention_mask=False)

    outputs = model.generate(**inputs, max_length=200, return_dict_in_generate=True, output_logits=True)
    logits = list(outputs["logits"])
    return [l.detach().numpy() for l in logits]
"#;
        Python::with_gil(|py| {
            let prg = PyModule::from_code(py, &prg, "x.py", "x")?;
            let py_result: Vec<&PyArrayDyn<f32>> = prg.getattr("ground")?.call0()?.extract()?;
            Ok(py_result.into_iter().map(Tensor::from).collect::<_>())
        })
    }

    #[test]
    fn phi2() {
        let _ = ground_truth().unwrap();
    }
}
