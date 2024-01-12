//Adapted from https://github.com/rerun-io/rerun MIT licensed.
use std::hash::Hash;

use rustc_hash::FxHashMap;
use slotmap::{Key as SlotHandle, SlotMap};

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

struct StoredResource<Res> {
    resource: Res,
}

/// Generic resource pool for all resources that are fully described upon creation, i.e. never have any variable content.
///
/// This implies, a resource is uniquely defined by its description.
/// We call these resources "static" because they never change their content over their lifetime.
pub(super) struct StaticResourcePool<Handle: SlotHandle, Descriptor, Resource> {
    resources: SlotMap<Handle, StoredResource<Resource>>,
    lookup: FxHashMap<Descriptor, Handle>,
}

/// We cannot #derive(Default) as that would require Handle/Desc/Res to implement Default too.
impl<Handle: SlotHandle, Desc, Res> Default for StaticResourcePool<Handle, Desc, Res> {
    fn default() -> Self {
        Self {
            resources: Default::default(),
            lookup: Default::default(),
        }
    }
}

impl<Handle, Descriptor, Resource> StaticResourcePool<Handle, Descriptor, Resource>
where
    Handle: SlotHandle,
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
        &mut self,
        descriptor: &Descriptor,
        creation_func: C,
    ) -> Handle {
        *self.lookup.entry(descriptor.clone()).or_insert_with(|| {
            let resource = creation_func(descriptor);
            self.resources.insert(StoredResource { resource })
        })
    }

    pub fn get_resource(&self, handle: Handle) -> Result<&Resource, PoolError> {
        Self::to_pool_error(
            self.resources
                .get(handle)
                .map(|resource| &resource.resource),
            handle,
        )
    }

    pub fn num_resources(&self) -> usize {
        self.resources.len()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use slotmap::Key;

    use crate::gpu::PoolError;

    use super::StaticResourcePool;

    slotmap::new_key_type! { pub struct ConcreteHandle; }

    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    pub struct ConcreteResourceDesc(u32);

    #[derive(PartialEq, Eq, Debug)]
    pub struct ConcreteResource(u32);

    type Pool = StaticResourcePool<ConcreteHandle, ConcreteResourceDesc, ConcreteResource>;

    #[test]
    fn resource_reuse() {
        let mut pool = Pool::default();

        // New resource
        let res0 = {
            let new_resource_created = Cell::new(false);

            let constructor = |d: &ConcreteResourceDesc| {
                new_resource_created.set(true);
                ConcreteResource(d.0)
            };

            let handle = pool.get_or_create(&ConcreteResourceDesc(0), constructor);
            assert!(new_resource_created.get());
            handle
        };

        // Get same resource again
        {
            let new_resource_created = Cell::new(false);
            let handle = pool.get_or_create(&ConcreteResourceDesc(0), |d| {
                new_resource_created.set(true);
                ConcreteResource(d.0)
            });
            assert!(!new_resource_created.get());
            assert_eq!(handle, res0);
        }
    }

    #[test]
    fn get_resource() {
        let mut pool = Pool::default();

        // Query with valid handle
        let handle = pool.get_or_create(&ConcreteResourceDesc(0), |d| ConcreteResource(d.0));
        assert!(pool.get_resource(handle).is_ok());
        assert_eq!(*pool.get_resource(handle).unwrap(), ConcreteResource(0));

        // Query with null handle
        assert_eq!(
            pool.get_resource(ConcreteHandle::null()),
            Err(PoolError::NullHandle)
        );

        // Query with invalid handle
        pool = Pool::default();
        assert_eq!(
            pool.get_resource(handle),
            Err(PoolError::ResourceNotAvailable)
        );
    }
}
