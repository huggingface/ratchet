use half::f16;
use num::traits::real::Real;
use ratchet::{rvec, shape, Tensor};
use ratchet_nn::{KVEntry, Linear, Module};

#[derive(Debug)]
pub struct MultiHeadAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads: usize,
    dk: Tensor,
}

impl MultiHeadAttention {
    pub fn new(q: Linear, k: Linear, v: Linear, o: Linear, n_heads: usize) -> MultiHeadAttention {
        let n_state = q.w.shape()[1];
        let dk = match q.w.dt() {
            ratchet::DType::F16 => {
                let dk = f16::from_f32((n_state / n_heads) as f32);
                Tensor::from_data(
                    [dk.powf(f16::from_f32(-0.25))],
                    shape![1],
                    q.w.device().clone(),
                )
            }
            ratchet::DType::F32 => {
                let dk = (n_state / n_heads) as f32;
                Tensor::from_data([dk.powf(-0.25)], shape![1], q.w.device().clone())
            }
            _ => unimplemented!(),
        };
        MultiHeadAttention {
            q,
            k,
            v,
            o,
            n_heads,
            dk,
        }
    }
}

#[derive(Debug, derive_new::new)]
pub struct MHAInputs {
    x: Tensor,
    xa: Option<Tensor>,
    mask: Option<Tensor>,
    cache: Option<KVEntry>,
    is_causal: bool,
}

impl Module for MultiHeadAttention {
    type Input = MHAInputs;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let MHAInputs {
            x,
            xa,
            mask,
            cache,
            is_causal,
        } = input;
        let is_xattn = xa.is_some();

        let q = self.q.schedule(x.clone())?;

        let to_project = xa.unwrap_or(x);
        let k = self.k.schedule(to_project.clone())?;
        let v = self.v.schedule(to_project)?;

        let (k, v) = if let Some(kv) = cache {
            let prev_entries = kv.entries;
            let k_cache = kv.k_cache.cache(k, 1, prev_entries)?;
            let v_cache = kv.v_cache.cache(v, 1, prev_entries)?;
            (k_cache, v_cache)
        } else {
            (k, v)
        };

        self.qkv_attention(q, k, v, mask, is_xattn, is_causal)
    }
}

impl MultiHeadAttention {
    fn qkv_attention(
        &self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Option<Tensor>,
        x_attn: bool,
        is_causal: bool,
    ) -> anyhow::Result<Tensor> {
        let [bs, n_ctx, n_state]: [usize; 3] = q.shape().try_into()?;
        let [k0, k1, _]: [usize; 3] = k.shape().try_into()?;
        let [v0, v1, _]: [usize; 3] = v.shape().try_into()?;
        let q_dt = q.dt();

        let hdim = n_state / self.n_heads;

        let qs = shape![bs, n_ctx, self.n_heads, hdim];
        let ks = shape![k0, k1, self.n_heads, hdim];
        let vs = shape![v0, v1, self.n_heads, hdim];

        let q = q.view(qs)?.permute(&[0, 2, 1, 3])?.mul(self.dk.clone())?;
        let k = k.view(ks)?.permute(&[0, 2, 3, 1])?.mul(self.dk.clone())?;
        let v = v.view(vs)?.permute(&[0, 2, 1, 3])?;

        if x_attn {
            //TODO: static caching
        }

        let mut qk = q.matmul(k, false, false)?;

        if let Some(m) = mask {
            let prepared_mask = if is_causal {
                m.slice(&[0..n_ctx, 0..n_ctx])?
            } else {
                m.clone()
            };
            qk = qk.add(prepared_mask)?;
        }
        qk = qk.full()?;

        let w = qk.softmax(3)?.cast(q_dt)?;

        let wv = w
            .matmul(v, false, false)?
            .permute(&[0, 2, 1, 3])?
            .view(shape![bs, n_ctx, n_state])?;

        self.o.schedule(wv)
    }
}
