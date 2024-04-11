use ratchet::Tensor;

use crate::Module;

///    """Implements the rotary positional encoding.
///
///    The traditional implementation rotates consecutive pairs of elements in the
///    feature dimension while the default implementation rotates pairs with
///    stride half the feature dimensions for efficiency.
///
///    For more details see `RoFormer: Enhanced Transformer with Rotary Position
///    Embedding <https://arxiv.org/abs/2104.09864>`_.
///
///    Args:
///        dims (int): The feature dimensions to be rotated. If the input feature
///            is larger than dims then the rest is left unchanged.
///        traditional (bool, optional): If set to ``True`` choose the traditional
///            implementation which is slightly less efficient. Default: ``False``.
///        base (float, optional): The base used to compute angular frequency for
///            each dimension in the positional encodings. Default: ``10000``.
///        scale (float, optional): The scale used to scale the positions. Default: ``1.0``.
///    """
#[derive(Clone, Debug, derive_new::new)]
pub struct RotaryEmbedding {
    dim: usize,
    traditional: bool,
    base: f32,
    scale: f32,
}

pub struct RotaryInput {
    pub input: Tensor,
    pub offset: usize,
}

impl Module for RotaryEmbedding {
    type Input = RotaryInput;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let RotaryInput { input, offset } = input;
        input.rope(self.dim, self.base, offset)
    }
}
