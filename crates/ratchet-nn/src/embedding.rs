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

    fn forward(&self, input: Self::Input) -> anyhow::Result<Tensor> {
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

#[cfg(test)]
mod tests {
    use proptest::arbitrary::Arbitrary;
    use proptest::strategy::{BoxedStrategy, Just, Strategy};
    use test_strategy::proptest;

    use ratchet::test_util::run_py_prg;
    use ratchet::{rvec, shape, Device, DeviceRequest, Quantization, Quantizer, Shape, Tensor};

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
        let mut weight = Tensor::randn::<f32>(vocab_shape, Device::CPU);

        let ground_truth = ground_truth(&weight, &indices, false).unwrap();
        println!("GROUND TRUTH: {:?}", ground_truth);

        let weight = weight.to(&device).unwrap();
        let indices = indices.to(&device).unwrap();

        let embedding = Embedding::new(weight, true);
        let result = embedding.forward(indices).unwrap().resolve().unwrap();
        let x = result.to(&Device::CPU).unwrap();
        println!("OURS: {:?}", x);
        ground_truth.all_close(&x, 1e-1, 1e-1).unwrap();
    }

    #[derive(Debug, Clone)]
    struct EmbeddingProblem {
        vocab_shape: Shape,
        indices: Tensor,
    }

    #[test]
    fn debug_embedding() {
        //Transposed
        let prob = EmbeddingProblem {
            vocab_shape: shape![384, 4000],
            indices: Tensor::from_data([3i32, 4i32, 1000i32], shape![1, 3], Device::CPU),
        };
        run_embedding_trial(prob);
    }

    #[proptest(cases = 16)]
    fn test_embedding(prob: EmbeddingProblem) {
        run_embedding_trial(prob);
    }
}
