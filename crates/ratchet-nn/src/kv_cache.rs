use ratchet::{Device, Shape, Tensor, TensorDType};

#[derive(Clone, Debug)]
pub struct KVEntry {
    pub k_cache: Tensor,
    pub v_cache: Tensor,
    pub entries: usize,
}

impl KVEntry {
    pub fn allocate<T: TensorDType>(shape: &Shape, device: &Device) -> Self {
        KVEntry {
            k_cache: Tensor::zeros::<T>(shape, device),
            v_cache: Tensor::zeros::<T>(shape, device),
            entries: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct KVCache(pub Vec<KVEntry>);

impl std::ops::Index<usize> for KVCache {
    type Output = KVEntry;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl KVCache {
    pub fn new<T: TensorDType>(n_layers: i32, shape: Shape, device: &Device) -> Self {
        let mut entries = Vec::with_capacity(n_layers as _);
        for _ in 0..n_layers {
            entries.push(KVEntry::allocate::<T>(&shape, device));
        }
        KVCache(entries)
    }

    pub fn update(&mut self, offset: usize) {
        for entry in &mut self.0 {
            entry.entries += offset;
        }
    }

    pub fn entries(&self, layer: usize) -> usize {
        self.0[layer].entries
    }

    pub fn reset(&mut self) {
        for entry in &mut self.0 {
            entry.entries = 0;
        }
    }
}
