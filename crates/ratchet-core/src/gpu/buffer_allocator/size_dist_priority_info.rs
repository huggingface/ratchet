use std::{cmp::Ordering, collections::HashMap};

use crate::TensorId;

use super::BufferId;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SizeDistPriorityInfo {
    pub position: usize,
    pub tensor_size: usize,
    pub dist: HashMap<BufferId, usize>,
    pub best_dist: Option<usize>,
    pub best_buffer: Option<BufferId>,
    pub record_index: usize, //index into tensor_usage_records
    pub tensor_usage_id: TensorId,
}

impl PartialOrd for SizeDistPriorityInfo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl SizeDistPriorityInfo {
    pub fn recalc_best_dist(&mut self) {
        let mut best_dist = 0;
        let mut best_buffer = None;
        for (buffer_id, dist) in self.dist.iter() {
            if *dist < best_dist {
                best_dist = *dist;
                best_buffer = Some(buffer_id.clone());
            }
        }
        self.best_dist = Some(best_dist);
        self.best_buffer = best_buffer;
    }
}

impl Ord for SizeDistPriorityInfo {
    fn cmp(&self, other: &Self) -> Ordering {
        let condition = self.position < other.position
            || (self.position == other.position
                && (self.best_dist < other.best_dist
                    || (self.best_dist == other.best_dist
                        && self.tensor_size > other.tensor_size)));
        if condition {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }
}
