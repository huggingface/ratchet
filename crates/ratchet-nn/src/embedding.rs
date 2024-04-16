use crate::Module;
use ratchet::{shape, Tensor};

/// # Embedding
///
/// Standard `torch.nn.Embedding` module.
/// However, we also support the `transposed` flag, which means that your vocab tensor
/// is transposed.
///
/// This is useful in the following case:
/// 1. You have quantized your vocabulary tensor & Ratchet does not support fast quantized
///    transposed matrix multiplication (like the one you use to obtain your logits).
///    Therefore, you can pretranspose your vocab, to avoid the transposed matmul and
///    set `transposed` to `true`.
#[derive(Debug, derive_new::new)]
pub struct Embedding {
    pub weight: Tensor,
    pub transposed: bool,
}

impl Module for Embedding {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let mut output_shape = input.shape().clone();
        let weight_rank = self.weight.rank();
        let weight_dim = if self.transposed { 0 } else { weight_rank - 1 };
        output_shape.push(self.weight.shape()[weight_dim]);

        let flat_shape = shape![input.shape().numel()];
        let flat = input.view(flat_shape)?;
        let selection_dim = if self.transposed { 1 } else { 0 };
        let indexed = self.weight.clone().index_select(flat, selection_dim)?;
        let result = if self.transposed {
            indexed.permute(&[1, 0])?
        } else {
            indexed
        };
        result.view(output_shape)
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use hf_hub::api::sync::Api;
    use proptest::arbitrary::Arbitrary;
    use proptest::strategy::{BoxedStrategy, Just, Strategy};
    use ratchet::{Quantization, Quantizer};
    use ratchet_loader::gguf::gguf::Content;
    use test_strategy::proptest;
    use tokenizers::Tokenizer;

    use ratchet::test_util::run_py_prg;
    use ratchet::{rvec, shape, Device, DeviceRequest, Shape, Tensor};

    use crate::{Embedding, Module};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    impl Arbitrary for EmbeddingProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            {
                let args = vec![1..512usize, 1..16usize];
                args.prop_map(Into::<Shape>::into).boxed()
            }
            .prop_flat_map(|vocab_shape| (Just(vocab_shape), 1..64usize))
            .prop_map(|(vocab_shape, num_indices)| {
                let indices =
                    Tensor::randint(0, vocab_shape[0] as i32, shape![num_indices], Device::CPU);
                EmbeddingProblem {
                    vocab_shape,
                    indices,
                }
            })
            .boxed()
        }
    }

    fn ground_truth(weight: &Tensor, indices: &Tensor, transposed: bool) -> anyhow::Result<Tensor> {
        let arg = if transposed {
            "torch.from_numpy(weight).t().contiguous()"
        } else {
            "torch.from_numpy(weight)"
        };

        let prg = format!(
            r#"
import torch
def embedding(weight, indices):
    embedding = torch.nn.Embedding.from_pretrained({})
    return embedding(torch.from_numpy(indices)).numpy()
"#,
            arg
        );
        run_py_prg(prg.to_string(), &[weight, indices], &[])
    }

    fn run_embedding_trial(problem: EmbeddingProblem) {
        let device = GPU_DEVICE.with(|d| d.clone());
        println!("Embedding problem: {:?}", problem);
        let EmbeddingProblem {
            vocab_shape,
            indices,
        } = problem;
        let weight = Tensor::randn::<f32>(vocab_shape, Device::CPU);

        let transposed = false;
        let ground_truth = ground_truth(&weight, &indices, transposed).unwrap();
        println!("GROUND TRUTH: {:?}", ground_truth);

        let weight = weight.to(&device).unwrap();
        let indices = indices.to(&device).unwrap();

        let embedding = Embedding::new(weight, transposed);
        let result = embedding.schedule(indices).unwrap().resolve().unwrap();
        let x = result.to(&Device::CPU).unwrap();
        println!("OURS: {:?}", x);
        ground_truth.all_close(&x, 1e-6, 1e-6).unwrap();
    }

    #[derive(Debug, Clone)]
    struct EmbeddingProblem {
        vocab_shape: Shape,
        indices: Tensor,
    }

    #[test]
    fn debug_embedding() {
        let prob = EmbeddingProblem {
            vocab_shape: shape![10000, 384],
            indices: Tensor::from_data([400i32, 9001i32, 5555i32], shape![1, 3], Device::CPU),
        };
        run_embedding_trial(prob);
    }

    #[proptest(cases = 16)]
    fn test_embedding(prob: EmbeddingProblem) {
        run_embedding_trial(prob);
    }

    #[test]
    fn dbg_phi_embedding() -> anyhow::Result<()> {
        let api = Api::new().unwrap();
        let model_repo = api.model("FL33TW00D-HF/phi2".to_string());
        let model_path = model_repo.get("phi2-q8_0.gguf").unwrap();
        //let model_path = model_repo.get("phi2-f16.gguf").unwrap();

        println!("MODEL PATH: {}", model_path.display());

        let mut reader = std::io::BufReader::new(std::fs::File::open(model_path)?);
        let device = Device::request_device(DeviceRequest::GPU)?;
        let content = Content::read(&mut reader)?;

        let token_weight = content.tensor(&mut reader, "token_embd.weight", &Device::CPU)?;

        let quantizer = Quantizer::new(Quantization::SInt8);
        let dequant = quantizer.sint8_dequantize(token_weight);
        println!("DEQUANT: {:?}", dequant.to_ndarray_view::<f32>());

        let token_weight = content.tensor(&mut reader, "token_embd.weight", &Device::CPU)?;

        //println!("TOKEN WEIGHT: {:?}", token_weight.to_ndarray_view::<f32>());
        let embedding = Embedding::new(token_weight.to(&device)?, false);

        let tokenizer_repo = api.model("microsoft/phi-2".to_string());
        let tokenizer_path = tokenizer_repo.get("tokenizer.json").unwrap();
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

        let prompt = "def print_prime(n):";
        print!("{}", prompt);
        let encoding = tokenizer.encode(prompt, true).unwrap();
        let mut tokens = encoding
            .get_ids()
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();

        let input = Tensor::from_data(tokens.clone(), shape![1, tokens.len()], device.clone());
        let result = embedding.schedule(input).unwrap().resolve().unwrap();
        let cpu_result = result.to(&Device::CPU).unwrap();
        println!("CPU RESULT: {:?}", cpu_result.to_ndarray_view::<f32>());
        Ok(())
    }
}
