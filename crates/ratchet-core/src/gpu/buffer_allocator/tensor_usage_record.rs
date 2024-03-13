use crate::TensorId;
use rustc_hash::FxHashMap;
use std::cmp::Reverse;

/// Records the interval for which a tensor is used
/// produce & last_consumer as indices into the topologically sorted execution order
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorUsageRecord {
    pub id: Option<TensorId>,
    pub producer: Option<usize>,
    pub last_consumer: usize,
    #[cfg(debug_assertions)]
    pub last_consumer_id: TensorId,
    pub size: usize,
}

impl std::ops::Index<usize> for TensorUsageRecords {
    type Output = TensorUsageRecord;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for TensorUsageRecords {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

#[derive(Debug, Clone)]
pub struct TensorUsageRecords(pub Vec<TensorUsageRecord>);

impl From<FxHashMap<TensorId, TensorUsageRecord>> for TensorUsageRecords {
    fn from(mut map: FxHashMap<TensorId, TensorUsageRecord>) -> Self {
        let mut records = map.drain().map(|(_, v)| v).collect::<Vec<_>>();
        records.sort_unstable_by_key(|r| Reverse(r.size));
        TensorUsageRecords(records)
    }
}
