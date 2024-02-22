use std::cmp::Ordering;
use std::ops::Range;

use crate::TensorId;

/// Records the interval for which a tensor is used
/// produce & last_consumer as indices into the topologically sorted execution order
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorUsageRecord {
    pub producer: Option<usize>,
    pub last_consumer: usize,
    pub size: usize,
}

impl TensorUsageRecord {
    pub fn op_range(&self) -> Range<usize> {
        self.producer.unwrap_or(self.last_consumer)..self.last_consumer
    }
}

impl PartialOrd for TensorUsageRecord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.size.cmp(&other.size))
    }
}

/// The set of all tensor usage records within which an operation lies.
pub struct OpProfile(Vec<TensorUsageRecord>);

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
