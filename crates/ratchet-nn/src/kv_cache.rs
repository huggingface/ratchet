use ratchet::{prelude::shape, rvec, Device, Shape, Tensor};

use crate::MutableModule;

#[derive(Clone, Debug)]
pub struct KVEntry {
    pub k_cache: Tensor,
    pub v_cache: Tensor,
    pub entries: usize,
}

impl KVEntry {
    pub fn allocate(shape: &Shape, device: &Device) -> Self {
        KVEntry {
            k_cache: Tensor::zeros::<f32>(shape, device),
            v_cache: Tensor::zeros::<f32>(shape, device),
            entries: 0,
        }
    }
}

impl KVEntry {
    pub fn forward(&mut self, input: KVEntryInput) -> anyhow::Result<()> {
        let KVEntryInput { x, k } = input;

        let bs = x.shape()[0];
        let n_ctx = x.shape()[x.rank() - 2];
        let n_state = x.shape()[x.rank() - 1];

        let prev_entries = self.entries;
        let new_entries = prev_entries + n_ctx;
        if k {
            let mut k_cache = std::mem::take(&mut self.k_cache);
            println!("STRONG COUNT: {}", k_cache.strong_count());
            k_cache = k_cache
                .index_write(x, rvec![0, prev_entries, 0])?
                .view(shape![bs, new_entries, n_state])?;
            let _ = std::mem::replace(&mut self.k_cache, k_cache);
        } else {
            let mut v_cache = std::mem::take(&mut self.v_cache);
            v_cache = v_cache
                .index_write(x, rvec![0, prev_entries, 0])?
                .view(shape![bs, new_entries, n_state])?;
            let _ = std::mem::replace(&mut self.v_cache, v_cache);
        }
        Ok(())
    }
}

pub struct KVEntryInput {
    pub x: Tensor,
    pub k: bool,
}

#[derive(Clone, Debug)]
pub struct KVCache(Vec<KVEntry>);

impl std::ops::Index<usize> for KVCache {
    type Output = KVEntry;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for KVCache {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl KVCache {
    pub fn new(n_layers: i32, shape: Shape, device: &Device) -> Self {
        let mut entries = Vec::with_capacity(n_layers as _);
        for _ in 0..n_layers {
            entries.push(KVEntry::allocate(&shape, device));
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
