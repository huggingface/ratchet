use ratchet::{Device, Shape, Tensor};

#[derive(Clone, Debug)]
pub struct KVEntry {
    pub k_cache: Tensor,
    pub v_cache: Tensor,
}

impl KVEntry {
    pub fn allocate(shape: &Shape, device: &Device) -> Self {
        KVEntry {
            k_cache: Tensor::zeros::<f32>(shape, device),
            v_cache: Tensor::zeros::<f32>(shape, device),
        }
    }
}

#[derive(Clone, Debug)]
pub struct KVCache(Vec<KVEntry>);

impl std::ops::Index<usize> for KVCache {
    type Output = KVEntry;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl KVCache {
    pub fn new(n_layers: i32, device: &Device) -> Self {
        let mut entries = Vec::with_capacity(n_layers as _);
        for _ in 0..n_layers {
            entries.push(KVEntry::allocate(&Shape::default(), device));
        }
        KVCache(entries)
    }
}
