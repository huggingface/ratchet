use std::{cmp::Reverse, ops::Range};

use rustc_hash::FxHashMap;

use crate::{rvec, RVec, TensorId};

/// Records the interval for which a tensor is used
/// produce & last_consumer as indices into the topologically sorted execution order
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorUsageRecord {
    pub id: Option<TensorId>,
    pub producer: Option<usize>,
    pub last_consumer: usize,
    pub last_consumer_id: TensorId,
    pub size: usize,
}

impl TensorUsageRecord {
    pub fn op_range(&self) -> Range<usize> {
        self.producer.unwrap_or(self.last_consumer)..self.last_consumer
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

/// The set of all tensor usage records within which an operation lies.
#[derive(Debug, Clone)]
pub struct OpProfile(RVec<TensorUsageRecord>);

impl Default for OpProfile {
    fn default() -> Self {
        OpProfile(rvec![])
    }
}

impl OpProfile {
    pub fn push(&mut self, record: TensorUsageRecord) {
        self.0.push(record);
    }
}

impl std::ops::Index<usize> for OpProfile {
    type Output = TensorUsageRecord;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
