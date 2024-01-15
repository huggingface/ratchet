use super::*;
use crate::{gpu::WgpuDevice, RVec};
use std::sync::Arc;

slotmap::new_key_type! { pub struct GpuBindGroupHandle; }

/// A reference-counter baked bind group.
///
/// Once instances handles are dropped, the bind group will be marked for reclamation in the following pass.
/// Tracks use of dependent resources as well!
#[derive(Clone)]
pub struct GpuBindGroup {
    resource: Arc<DynamicResource<GpuBindGroupHandle, BindGroupDescriptor, wgpu::BindGroup>>,
    _owned_buffers: RVec<GPUBuffer>,
}

impl std::fmt::Debug for GpuBindGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBindGroup")
            .field("resource", &self.resource.inner)
            .finish()
    }
}

impl std::ops::Deref for GpuBindGroup {
    type Target = wgpu::BindGroup;

    fn deref(&self) -> &Self::Target {
        &self.resource.inner
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct BindGroupEntry {
    pub handle: GpuBufferHandle,
    /// Base offset of the buffer. For bindings with `dynamic == true`, this offset
    /// will be added to the dynamic offset provided in [`wgpu::RenderPass::set_bind_group`].
    ///
    /// The offset has to be aligned to [`wgpu::Limits::min_uniform_buffer_offset_alignment`]
    /// or [`wgpu::Limits::min_storage_buffer_offset_alignment`] appropriately.
    pub offset: wgpu::BufferAddress,
    /// Size of the binding, or `None` for using the rest of the buffer.
    pub size: Option<wgpu::BufferSize>,
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct BindGroupDescriptor {
    pub entries: RVec<BindGroupEntry>,
    pub layout: BindGroupLayoutHandle,
}

impl DynamicResourcesDesc for BindGroupDescriptor {
    fn resource_size_in_bytes(&self) -> u64 {
        // Size depends on gpu/driver (like with all resources).
        // We could guess something like a pointer per descriptor, but let's not pretend we know!
        0
    }

    fn allow_reuse(&self) -> bool {
        true
    }
}

/// Resource pool for bind groups.
///
/// Implementation notes:
/// Requirements regarding ownership & resource lifetime:
/// * owned [`wgpu::BindGroup`] should keep buffer/texture alive
///   (user should not need to hold buffer/texture manually)
/// * [`GpuBindGroupPool`] should *try* to re-use previously created bind groups if they happen to match
/// * musn't prevent buffer/texture re-use on next pass
///   i.e. a internally cached [`GpuBindGroupPool`]s without owner shouldn't keep textures/buffers alive
///
/// We satisfy these by retrieving the "weak" buffer/texture handles and make them part of the [`GpuBindGroup`].
/// Internally, the [`GpuBindGroupPool`] does *not* hold any strong reference to any resource,
/// i.e. it does not interfere with the ownership tracking of buffer/texture pools.
/// The question whether a bind groups happen to be re-usable becomes again a simple question of matching
/// bind group descs which itself does not contain any ref counted objects!
pub struct BindGroupPool {
    // Use a DynamicResourcePool because it gives out reference counted handles
    // which makes interacting with buffer/textures easier.
    //
    // On the flipside if someone requests the exact same bind group again as before,
    // they'll get a new one which is unnecessary. But this is *very* unlikely to ever happen.
    inner: DynamicResourcePool<GpuBindGroupHandle, BindGroupDescriptor, wgpu::BindGroup>,
}

impl BindGroupPool {
    pub fn new() -> Self {
        Self {
            inner: DynamicResourcePool::default(),
        }
    }

    /// Returns a reference-counted, currently unused bind-group.
    /// Once ownership to the handle is given up, the bind group may be reclaimed in future passs.
    /// The handle also keeps alive any dependent resources.
    pub fn allocate(&self, desc: &BindGroupDescriptor, device: &WgpuDevice) -> GpuBindGroup {
        // Retrieve strong handles to buffers and textures.
        // This way, an owner of a bind group handle keeps buffers & textures alive!.
        let owned_buffers: RVec<GPUBuffer> = {
            desc.entries
                .iter()
                .map(|e| device.get_buffer(e.handle).unwrap())
                .collect()
        };

        let resource = self.inner.allocate(desc, |desc| {
            let mut buffer_index = 0;

            let entries = desc
                .entries
                .iter()
                .enumerate()
                .map(|(index, entry)| wgpu::BindGroupEntry {
                    binding: index as _,
                    resource: {
                        let res = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &owned_buffers[buffer_index],
                            offset: entry.offset,
                            size: entry.size,
                        });
                        buffer_index += 1;
                        res
                    },
                })
                .collect::<Vec<_>>();

            let resources = device.bind_group_layout_resources();
            let bind_group_descriptor = wgpu::BindGroupDescriptor {
                label: None,
                entries: &entries,
                layout: resources.get(desc.layout).unwrap(),
            };

            device.create_bind_group(&bind_group_descriptor)
        });

        GpuBindGroup {
            resource,
            _owned_buffers: owned_buffers,
        }
    }

    pub fn begin_pass(&mut self, pass_index: u64) {
        self.inner.begin_pass(pass_index, |_res| {});
    }

    pub fn num_resources(&self) -> usize {
        self.inner.num_resources()
    }
}
