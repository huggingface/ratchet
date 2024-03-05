use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use wgpu::BufferUsages;

use crate::{
    gpu::{
        BufferDescriptor, BufferPool, BufferUsagesExt, CpuUniform, GpuBufferHandle,
        PooledGPUBuffer, SizeDistPriorityInfo, TensorUsageRecords, WgpuDevice, UNIFORM_ALIGN,
    },
    DeviceError, Tensor, TensorId,
};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use super::{BufferId, BufferRequest, OpProfile, TensorUsageRecord};

#[derive(Clone, Debug, thiserror::Error)]
pub enum AllocatorError {
    #[error("Buffer not found")]
    BufferNotFound,
}

pub struct BufferAllocator {
    pool: RwLock<BufferPool>,
}

impl BufferAllocator {
    pub fn new() -> Self {
        Self {
            pool: BufferPool::new().into(),
        }
    }

    pub fn begin_pass(&self, pass_index: u64) {
        self.pool.write().begin_pass(pass_index);
    }

    pub fn get(&self, handle: GpuBufferHandle) -> PooledGPUBuffer {
        self.pool.read().get(handle).unwrap()
    }

    pub fn create_buffer(&self, desc: &BufferDescriptor, device: &WgpuDevice) -> PooledGPUBuffer {
        self.pool.write().get_or_create(desc, device)
    }

    pub fn create_buffer_init(
        &self,
        desc: &BufferDescriptor,
        contents: &[u8],
        device: &WgpuDevice,
    ) -> PooledGPUBuffer {
        let buf = self.pool.write().get_or_create(desc, device);
        device.queue().write_buffer(&buf.inner, 0, contents);
        device.queue().submit(None);
        device.poll(wgpu::Maintain::Wait);
        buf
    }

    pub fn create_uniform_init(&self, uniform: CpuUniform, device: &WgpuDevice) -> PooledGPUBuffer {
        let mut uniform = uniform.into_inner();
        uniform.resize(
            uniform.len() + UNIFORM_ALIGN - uniform.len() % UNIFORM_ALIGN,
            0u8,
        );
        let desc = BufferDescriptor::new(
            uniform.len() as _,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            false,
        );

        let resource = self.pool.write().get_or_create(&desc, device);
        device
            .queue()
            .write_buffer(&resource.inner, 0, uniform.as_slice());
        resource
    }

    /// # Graph memory allocation
    ///
    /// Greedy algorithm, that takes the first buffer larger than the request
    /// In future, since we know the entire graph and sizes, we can
    /// do better.
    fn graph_allocate(
        &self,
        descriptor: BufferDescriptor,
        free: &mut Vec<PooledGPUBuffer>,
        device: &WgpuDevice,
    ) -> PooledGPUBuffer {
        let required_size = descriptor.size as _;
        let mut closest_index = None;
        let mut closest_size_diff: Option<usize> = None;
        for (idx, buffer) in free.iter().enumerate() {
            let current_size = buffer.descriptor.size as _;
            if current_size >= required_size {
                let size_diff = usize::abs_diff(current_size, required_size);

                if closest_size_diff.map_or(true, |diff| size_diff < diff) {
                    closest_index = Some(idx);
                    closest_size_diff = Some(size_diff);
                }
            }
        }

        if std::env::var("RATCHET_DEBUG").is_ok() {
            return self.create_buffer(&descriptor, device);
        }

        match closest_index {
            Some(idx) => free.remove(idx),
            None => self.create_buffer(&descriptor, device),
        }
    }

    /// # Inplace operations
    ///
    /// If an operation supports inplace, we need to "lease" the buffer
    /// from the actual source (i.e the first non-inplace operation)
    ///
    /// On what conditions do we terminate the upward traversal?
    /// 1. We reach an operation that does not support inplace
    /// 2. We reach an operation that has more than one consumer
    /// 3. We reach an operation that has more than one source (this condition is wrong)
    fn determine_tensor_source(source: &Tensor) -> &Tensor {
        let mut true_source = source;
        loop {
            let cant_inplace = !true_source.op().supports_inplace();
            let multiple_consumers = Arc::strong_count(&true_source.inner) > 1;
            log::debug!("Conditions: {:?} {:?}", cant_inplace, multiple_consumers);
            if cant_inplace || multiple_consumers {
                break;
            }

            true_source = true_source.op().srcs()[0]; //TODO: this shouldn't be 0, operations
                                                      //should define their inplace source
        }
        log::debug!("Traversed to true source: {:?}", true_source.id());
        true_source
    }

    //To calculate the tensor usage records, we do the following:
    //1. Traverse topologically sorted graph in reverse order
    //2. When we encounter the last consumer of a tensor, we start recording the interval.
    //3. When we encounter the producer of a tensor, we stop recording the interval.
    fn calculate_usage_records(
        execution_order: &[&Tensor],
    ) -> FxHashMap<TensorId, TensorUsageRecord> {
        let mut records =
            FxHashMap::with_capacity_and_hasher(execution_order.len(), Default::default());
        let topo_len = execution_order.len() - 1;
        for (iter, t) in execution_order.iter().rev().enumerate() {
            if t.resolved() {
                continue;
            }
            for source in t.op().srcs() {
                if source.resolved() {
                    continue;
                }
                let true_source = Self::determine_tensor_source(source);
                records
                    .entry(true_source.id())
                    .or_insert_with(|| TensorUsageRecord {
                        id: None,
                        producer: None,
                        #[cfg(debug_assertions)]
                        producer_op: None,
                        last_consumer: topo_len - iter,
                        #[cfg(debug_assertions)]
                        last_consumer_id: t.id(),
                        size: true_source.num_bytes(),
                    });
            }

            if let Some(record) = records.get_mut(&t.id()) {
                record.id = Some(t.id());
                record.producer = Some(topo_len - iter);
                #[cfg(debug_assertions)]
                {
                    record.producer_op = Some(t.op().name().to_string());
                }
            }
        }
        records
    }

    fn calculate_op_profiles(usage_records: &TensorUsageRecords, num_ops: usize) -> Vec<OpProfile> {
        //An operation profile is the set of all tensor usage records within which an operation lies.
        let mut op_profiles: Vec<OpProfile> = vec![OpProfile::default(); num_ops];
        for record in usage_records.0.iter() {
            for o in record.op_range() {
                op_profiles[o].push(record.clone());
            }
        }

        for profile in op_profiles.iter_mut() {
            profile.sort();
        }

        op_profiles
    }

    fn calculate_positional_maximums(
        usage_records: &TensorUsageRecords,
        num_ops: usize,
    ) -> Vec<usize> {
        let profiles = Self::calculate_op_profiles(usage_records, num_ops);
        let num_positions = profiles.iter().map(|p| p.len()).max().unwrap();
        log::info!("Num positions: {}", num_positions);
        let mut positional_maximums = vec![0; num_positions];

        for profile in profiles {
            for (idx, record) in profile.iter().enumerate() {
                positional_maximums[idx] = std::cmp::max(positional_maximums[idx], record.size);
            }
        }
        log::info!("Postional maximums: {:#?}", positional_maximums);
        positional_maximums
    }

    //https://arxiv.org/pdf/2001.03288.pdf + inplace support
    //Takes in const assignments as inplace may be performed on constants
    pub fn greedy_by_size(
        &self,
        execution_order: &[&Tensor],
        device: &WgpuDevice,
    ) -> Result<FxHashMap<TensorId, PooledGPUBuffer>, DeviceError> {
        let mut assignments = FxHashMap::default();
        let record_map = Self::calculate_usage_records(execution_order);
        let records = TensorUsageRecords::from(record_map);
        let mut shared_objects: Vec<PooledGPUBuffer> = Vec::with_capacity(records.0.len());

        for record in records.0.iter() {
            if record.producer.is_none() {
                continue;
            }
            let mut best_obj = None;
            for obj in shared_objects.iter() {
                let mut suitable = true;
                for inner_r in records.0.iter() {
                    if inner_r.producer.is_none() {
                        continue;
                    }
                    let max_first =
                        std::cmp::max(record.producer.unwrap(), inner_r.producer.unwrap());
                    let min_last = std::cmp::min(record.last_consumer, inner_r.last_consumer);
                    if max_first <= min_last && assignments.get(&inner_r.id.unwrap()) == Some(obj) {
                        suitable = false;
                        break;
                    }
                }
                if suitable {
                    best_obj = Some(obj);
                }
            }
            if let Some(obj) = best_obj {
                assignments.insert(record.id.unwrap(), (*obj).clone());
            } else {
                let buf = self.create_buffer(
                    &BufferDescriptor::new(record.size as _, BufferUsages::standard(), false),
                    device,
                );
                shared_objects.push(buf.clone());
                assignments.insert(record.id.unwrap(), buf);
            }
        }

        Ok(assignments)
    }

    fn determine_best_info(
        priority_info: &[SizeDistPriorityInfo],
        assignments: &FxHashMap<TensorId, BufferRequest>,
    ) -> SizeDistPriorityInfo {
        let mut best_info = None;

        for (info_index, info) in priority_info.iter().enumerate() {
            if assignments.contains_key(&info.tensor_usage_id) {
                continue;
            }
            best_info = match best_info {
                None => Some(info),
                Some(current_best) => {
                    if info > current_best {
                        Some(info)
                    } else {
                        Some(current_best)
                    }
                }
            };
        }
        best_info.unwrap().clone()
    }

    // Assigns given tensors to shared objects, using the following greedy
    // algorithm:
    // - Input: TensorUsageRecords of all intermediate tensors.
    // Distance between two usage intervals is the absolute difference between
    // closest tasks in their intervals. If two usage intervals don't intersect,
    // than the distance between them is positive;
    //
    // - Calculate positional maximums vector, e.g. the vector of lower bounds on
    // size of each shared object;
    // - For each tensor find the rightmost positional maximum, that is greater or
    // equal, than current tensor's size (call it position);
    // - Iterate through all tensors in increasing order of their
    // SizeDistPriority
    pub fn greedy_by_size_improved(
        &self,
        execution_order: &[&Tensor],
        device: &WgpuDevice,
    ) -> Result<FxHashMap<TensorId, PooledGPUBuffer>, DeviceError> {
        let mut assignments = FxHashMap::default();
        let record_map = Self::calculate_usage_records(execution_order);
        let tensor_usage_records = TensorUsageRecords::from(record_map.clone());
        let positional_maximums =
            Self::calculate_positional_maximums(&tensor_usage_records, execution_order.len());

        let mut priority_info = tensor_usage_records.iter().enumerate().fold(
            Vec::with_capacity(tensor_usage_records.len()),
            |mut acc, (record_index, record)| {
                let position = positional_maximums
                    .iter()
                    .position(|&x| x >= record.size)
                    .expect("No suitable position found");

                acc.push(SizeDistPriorityInfo {
                    position,
                    tensor_size: record.size,
                    record_index,
                    tensor_usage_id: record.id.unwrap(),
                    dist: HashMap::new(),
                    best_dist: None,
                    best_buffer: None,
                });
                acc
            },
        );

        for (record_index, record) in tensor_usage_records.iter().enumerate() {
            let best_info = Self::determine_best_info(&priority_info, &assignments);

            let mut new_object = None;
            if best_info.best_dist.is_none() {
                let request = BufferRequest {
                    id: BufferId::new(),
                    size: best_info.tensor_size,
                };
                new_object = Some(request.clone());
                assignments.insert(best_info.tensor_usage_id, request);
            } else {
                let current_assignment = assignments.get(&best_info.tensor_usage_id).unwrap();
                let request = BufferRequest {
                    id: current_assignment.id,
                    size: std::cmp::max(best_info.tensor_size, current_assignment.size),
                };
                assignments.insert(best_info.tensor_usage_id, request);
            }

            //Update dists for all other priority infos with new information
            for (info_index, info) in priority_info.iter_mut().enumerate() {
                if assignments.contains_key(&info.tensor_usage_id) {
                    continue;
                }

                if new_object.is_none() && !info.dist.contains_key(&best_info.best_buffer.unwrap())
                {
                    continue;
                }

                let best_record = record_map.get(&best_info.tensor_usage_id).unwrap();
                let current_record = record_map.get(&info.tensor_usage_id).unwrap();

                let dist = if current_record.last_consumer < best_record.producer.unwrap() {
                    Some(best_record.producer.unwrap() - current_record.last_consumer)
                } else if best_record.last_consumer < current_record.producer.unwrap() {
                    Some(current_record.producer.unwrap() - best_record.last_consumer)
                } else {
                    None
                };

                if dist.is_none() {
                    if let Some(best_buffer_id) = best_info.best_buffer {
                        info.dist.insert(best_buffer_id, usize::MAX);
                    }
                    if let Some(current_info_best_id) = info.best_buffer {
                        if let Some(best_buffer_id) = best_info.best_buffer {
                            if current_info_best_id == best_buffer_id {
                                info.recalc_best_dist();
                            }
                        }
                    }
                } else if new_object.is_some() {
                    info.dist
                        .insert(new_object.clone().unwrap().id, dist.unwrap());
                } else {
                    let dist = dist.unwrap();
                    let best_buffer_id = best_info.best_buffer.unwrap();
                    info.dist.insert(
                        best_buffer_id,
                        std::cmp::min(dist, *info.dist.get(&best_buffer_id).unwrap_or(&usize::MAX)),
                    );
                }
                if let Some(cur_dist) = info.best_dist {
                    if let Some(calc_dist) = dist {
                        if calc_dist < cur_dist {
                            info.best_dist = Some(calc_dist);
                            info.best_buffer = Some(best_info.best_buffer.unwrap());
                        }
                    }
                }
            }
        }

        println!("ASSIGNMENTS: {:#?}", assignments);

        //Now, we have a map from TensorId to BufferRequest
        //We need to convert BufferRequest to PooledGPUBuffer

        let unique_requests: HashSet<BufferRequest> = assignments.values().cloned().collect();
        let buffer_map = unique_requests
            .into_iter()
            .map(|req| {
                let buf = self.create_buffer(
                    &BufferDescriptor::new(req.size as _, BufferUsages::standard(), false),
                    device,
                );
                (req.id, buf)
            })
            .collect::<FxHashMap<BufferId, PooledGPUBuffer>>();
        let mut intermediate_assignments: FxHashMap<TensorId, PooledGPUBuffer> =
            FxHashMap::default();
        for (tensor_id, request) in assignments {
            let buf = buffer_map.get(&request.id).unwrap();
            intermediate_assignments.insert(tensor_id, buf.clone());
        }

        Ok(intermediate_assignments)
    }

    /// # Graph memory allocation
    ///
    /// Simple greedy algorithm
    /// 1. Iterate over all tensors in reverse order (leaf -> root)
    /// 2. For each tensor, loop through it's input values.
    ///     a. Assign a buffer for each input value, if it is not already assigned
    ///     b. If the input value is an inplace operation, traverse upwards until we find
    ///        the "true" buffer source (i.e the first non-inplace operation).
    /// 3. We release our **output** buffer, because the value is no longer needed,
    ///    and earlier tensors can use it.
    pub fn allocate_cfg(
        &self,
        execution_order: &[&Tensor],
        device: &WgpuDevice,
    ) -> Result<FxHashMap<TensorId, PooledGPUBuffer>, DeviceError> {
        let mut free = Vec::new(); //TODO: switch to BTreeMap
        let mut assignments = FxHashMap::default();
        //Assignments already needs all of the constants in it.
        for t in execution_order.iter().rev() {
            if t.resolved() {
                //Consts are immediately resolved
                let storage_guard = t.storage();
                let pooled = storage_guard
                    .as_ref()
                    .ok_or(AllocatorError::BufferNotFound)?
                    .try_gpu()?
                    .inner
                    .clone();
                assignments.insert(t.id(), pooled);
            }
        }

        //The output never gets allocated in the below loop, because it is not a source.
        //We know we need an allocation for the output.
        //We traverse upwards until we find the first non-inplace operation, and use it's buffer.
        let output = execution_order.last().unwrap();
        let output_source = Self::determine_tensor_source(output);
        let output_buffer = assignments
            .get(&output_source.id())
            .cloned()
            .unwrap_or_else(|| {
                self.graph_allocate(
                    BufferDescriptor::new(
                        output_source.num_bytes() as _,
                        BufferUsages::standard(),
                        false,
                    ),
                    &mut free,
                    device,
                )
            });
        assignments.insert(output.id(), output_buffer);

        let intermediate_assignments = match std::env::var("RATCHET_MEM_ALLOC_STRAT") {
            Ok(s) => match s.as_str() {
                "GREEDY_BY_SIZE" => self.greedy_by_size(execution_order, device)?,
                _ => panic!("Invalid memory allocation strategy"),
            },
            Err(_) => {
                //default
                self.greedy_by_size_improved(execution_order, device)?
                //self.greedy_by_size(execution_order, device)?
            }
        };

        println!("INTERMEDIATE ASSIGNMENTS: {:#?}", intermediate_assignments);

        assignments.extend(intermediate_assignments);

        //Loop through and add inplace assignments O(n)
        for t in execution_order.iter() {
            if t.resolved() {
                continue;
            }
            for source in t.op().srcs() {
                let true_source = Self::determine_tensor_source(source);
                if true_source.id() != source.id() {
                    if let Some(buf) = assignments.get(&true_source.id()) {
                        assignments.insert(source.id(), buf.clone());
                    }
                }
            }
        }

        log::info!(
            "Total bytes allocated: {}kb",
            self.pool.read().total_gpu_size_in_bytes() / 1024,
        );
        log::info!(
            "Total buffers allocated: {}",
            self.pool.read().num_resources()
        );

        Ok(assignments)
    }
}
