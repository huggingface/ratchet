//Adapted from https://github.com/rerun-io/rerun MIT licensed.
use std::hash::Hash;

use parking_lot::{RwLock, RwLockReadGuard};
use rustc_hash::FxHashMap;
use slotmap::{Key, SlotMap};

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum PoolError {
    #[error("Requested resource isn't available because the handle is no longer valid")]
    ResourceNotAvailable,

    #[error("The passed resource handle was null")]
    NullHandle,

    #[error("The passed descriptor doesn't refer to a known resource")]
    UnknownDescriptor,
}

pub trait ResourceConstructor<Descriptor, Resource> = Fn(&Descriptor) -> Resource;
pub trait ResourceDestructor<Resource> = FnMut(&Resource);

/// Generic resource pool for all resources that are fully described upon creation, i.e. never have any variable content.
///
/// This implies, a resource is uniquely defined by its description.
/// We call these resources "static" because they never change their content over their lifetime.
///
/// Lookup is queried to determine if a resource with the given descriptor already exists.
pub(super) struct StaticResourcePool<Handle: Key, Descriptor, Resource> {
    resources: RwLock<SlotMap<Handle, Resource>>,
    lookup: RwLock<FxHashMap<Descriptor, Handle>>,
}

/// We cannot #derive(Default) as that would require Handle/Desc/Res to implement Default too.
impl<Handle: Key, Desc, Res> Default for StaticResourcePool<Handle, Desc, Res> {
    fn default() -> Self {
        Self {
            resources: Default::default(),
            lookup: Default::default(),
        }
    }
}

impl<Handle, Descriptor, Resource> StaticResourcePool<Handle, Descriptor, Resource>
where
    Handle: Key,
    Descriptor: std::fmt::Debug + Clone + Eq + Hash,
{
    fn to_pool_error<T>(get_result: Option<T>, handle: Handle) -> Result<T, PoolError> {
        get_result.ok_or_else(|| {
            if handle.is_null() {
                PoolError::NullHandle
            } else {
                PoolError::ResourceNotAvailable
            }
        })
    }

    pub fn get_or_create<C: ResourceConstructor<Descriptor, Resource>>(
        &self,
        descriptor: &Descriptor,
        constructor: C,
    ) -> Handle {
        // Ensure the lock isn't held in the creation case.
        if let Some(handle) = self.lookup.read().get(descriptor) {
            return *handle;
        }

        let resource = constructor(descriptor);
        let handle = self.resources.write().insert(resource);
        self.lookup.write().insert(descriptor.clone(), handle);

        handle
    }

    /// Locks the resource pool for resolving handles.
    ///
    /// While it is locked, no new resources can be added.
    pub fn resources(&self) -> StaticResourcePoolReadLockAccessor<'_, Handle, Resource> {
        StaticResourcePoolReadLockAccessor {
            resources: self.resources.read(),
        }
    }

    pub fn num_resources(&self) -> usize {
        self.resources.read().len()
    }
}

/// Accessor to the resource pool, either by taking a read lock or by moving out the resources.
pub trait StaticResourcePoolAccessor<Handle: Key, Res> {
    fn get(&self, handle: Handle) -> Result<&Res, PoolError>;
}

/// Accessor to the resource pool by taking a read lock.
pub struct StaticResourcePoolReadLockAccessor<'a, Handle: Key, Res> {
    resources: RwLockReadGuard<'a, SlotMap<Handle, Res>>,
}

fn to_pool_error<T>(get_result: Option<T>, handle: impl Key) -> Result<T, PoolError> {
    get_result.ok_or_else(|| {
        if handle.is_null() {
            PoolError::NullHandle
        } else {
            PoolError::ResourceNotAvailable
        }
    })
}

impl<'a, Handle: Key, Res> StaticResourcePoolAccessor<Handle, Res>
    for StaticResourcePoolReadLockAccessor<'a, Handle, Res>
{
    fn get(&self, handle: Handle) -> Result<&Res, PoolError> {
        to_pool_error(self.resources.get(handle), handle)
    }
}
