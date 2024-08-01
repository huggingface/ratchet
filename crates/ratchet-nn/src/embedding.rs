use crate::Module;
use ratchet::{shape, Tensor};

/// # Embedding
///
/// Standard `torch.nn.Embedding` module.
#[derive(Debug, derive_new::new)]
pub struct Embedding {
    pub weight: Tensor,
}

impl Module for Embedding {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let mut output_shape = input.shape().clone();
        let weight_rank = self.weight.rank();
        let weight_dim = weight_rank - 1;
        output_shape.push(self.weight.shape()[weight_dim]);

        let flat_shape = shape![input.shape().numel()];
        let flat = input.view(flat_shape)?;
        let indexed = self.weight.clone().index_select(flat, 0)?;
        indexed.view(output_shape)
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use hf_hub::api::sync::Api;
    use proptest::arbitrary::Arbitrary;
    use proptest::strategy::{BoxedStrategy, Just, Strategy};
    use ratchet::{Quantization, Quantizer};
    use ratchet_loader::gguf::gguf::Header;
    use test_strategy::proptest;
    use tokenizers::Tokenizer;

    use ratchet::test_util::run_py_prg;
    use ratchet::{rvec, shape, Device, DeviceRequest, Shape, Tensor};

    use crate::{Embedding, Module};

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

    fn ground_truth(weight: &Tensor, indices: &Tensor) -> anyhow::Result<Tensor> {
        let arg = "torch.from_numpy(weight)";

        let prg = format!(
            r#"
import torch
def embedding(weight, indices):
    embedding = torch.nn.Embedding.from_pretrained({})
    return embedding(torch.from_numpy(indices)).numpy()
"#,
            arg
        );
        run_py_prg(prg.to_string(), &[weight, indices], &[], weight.dt())
    }

    fn run_embedding_trial(problem: EmbeddingProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        println!("Embedding problem: {:?}", problem);
        let EmbeddingProblem {
            vocab_shape,
            indices,
        } = problem;
        let weight = Tensor::randn::<f32>(vocab_shape, Device::CPU);

        let ground_truth = ground_truth(&weight, &indices).unwrap();

        let weight = weight.to(&device).unwrap();
        let indices = indices.to(&device).unwrap();

        let embedding = Embedding::new(weight);
        let result = embedding.schedule(indices).unwrap().resolve().unwrap();
        let x = result.to(&Device::CPU).unwrap();
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
}
