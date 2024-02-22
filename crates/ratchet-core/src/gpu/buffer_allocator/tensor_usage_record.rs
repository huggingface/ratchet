use std::cmp::Ordering;

/// Records the interval for which a tensor is used
/// produce & last_consumer as indices into the topologically sorted execution order
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorUsageRecord {
    pub producer: Option<usize>,
    pub last_consumer: usize,
    pub size: usize,
}

impl PartialOrd for TensorUsageRecord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.size.cmp(&other.size))
    }
}
