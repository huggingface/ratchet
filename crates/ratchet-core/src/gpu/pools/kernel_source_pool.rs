use crate::{KernelKey, KernelSource, MetaOperation, OperationError, Tensor, WorkgroupSize};

use super::static_resource_pool::{StaticResourcePool, StaticResourcePoolReadLockAccessor};
use std::hash::Hash;

// ---

slotmap::new_key_type! { pub struct KernelSourceHandle; }

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct KernelSourceDesc {
    /// Unique identifier for the compute module.
    /// e.g softmax_vec4_f32_128_1_1
    pub key: KernelKey,
}

impl KernelSourceDesc {
    pub fn create_compute_module<O: MetaOperation>(
        &self,
        op: &O,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        op.build_kernel(inplace, dst, workgroup_size)
    }
}

#[derive(Default)]
pub struct KernelSourcePool {
    pool: StaticResourcePool<KernelSourceHandle, KernelSourceDesc, KernelSource>,
}

impl KernelSourcePool {
    pub fn new() -> Self {
        Self {
            pool: StaticResourcePool::default(),
        }
    }

    pub fn get_or_create<O: MetaOperation>(
        &self,
        desc: &KernelSourceDesc,
        op: &O,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> KernelSourceHandle {
        self.pool.get_or_create(desc, |desc| {
            desc.create_compute_module(op, inplace, dst, workgroup_size)
                .unwrap()
        })
    }

    /// Locks the resource pool for resolving handles.
    ///
    /// While it is locked, no new resources can be added.
    pub fn resources(
        &self,
    ) -> StaticResourcePoolReadLockAccessor<'_, KernelSourceHandle, KernelSource> {
        self.pool.resources()
    }

    pub fn num_resources(&self) -> usize {
        self.pool.num_resources()
    }
}
