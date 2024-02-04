use ratchet_nn::Linear;

pub struct MultiHeadAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads: usize,
}
