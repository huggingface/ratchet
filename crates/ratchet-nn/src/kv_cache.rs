use ratchet::{Device, Shape, Tensor};

#[derive(Clone, Debug)]
pub struct KVEntry {
    pub k: Tensor,
    pub v: Tensor,
}

impl KVEntry {
    pub fn allocate(shape: &Shape, device: &Device) -> Self {
        let k = Tensor::zeros::<f32>(shape, device);
        let v = Tensor::zeros::<f32>(shape, device);
        KVEntry { k, v }
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

impl Default for KVCache {
    fn default() -> Self {
        KVCache(Vec::with_capacity(8))
    }
}

impl KVCache {}
