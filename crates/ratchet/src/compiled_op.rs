use crate::gpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutHandle, ComputePipelineHandle, GPUBuffer,
    GpuBindGroup, WgpuDevice, WorkgroupCount, UNIFORM_ALIGN,
};
use crate::{drvec, rvec, DRVec, RVec, Tensor};
use derive_new::new;
use wgpu::DynamicOffset;

//Compiled op represents a single kernel invocation
//We need to be more general here, and somehow encode encoder.copy_buffer_to_buffer as a COPY
//operation
#[derive(Debug, new)]
pub struct CompiledOp {
    workgroup_count: WorkgroupCount,
    pipeline: ComputePipelineHandle,
    storage_groups: RVec<GpuBindGroup>,
    offset: DynamicOffset, //offset into the metadata uniform buffer
}

impl CompiledOp {
    const MAX_BINDINGS_PER_GROUP: usize = 4;

    pub fn create_storage_bind_groups(
        srcs: &[&Tensor],
        dsts: &[&Tensor],
        bind_group_layouts: RVec<BindGroupLayoutHandle>,
        device: &WgpuDevice,
    ) -> RVec<GpuBindGroup> {
        let mut binding_counter: usize = 0;
        let mut bind_group_entries: DRVec<BindGroupEntry> = drvec![];

        for tensor in srcs.iter().chain(dsts.iter()) {
            bind_group_entries.push(BindGroupEntry {
                offset: 0,
                size: todo!(),
                handle: todo!(),
            });
            binding_counter += 1;
        }

        let mut storage_groups: RVec<GpuBindGroup> = rvec![];
        for (group_index, bind_group_layout) in bind_group_layouts.iter().enumerate() {
            let group_end = usize::min(
                (group_index + 1) * Self::MAX_BINDINGS_PER_GROUP,
                binding_counter,
            );
            let group_range = group_index * Self::MAX_BINDINGS_PER_GROUP..group_end;

            let descriptor = BindGroupDescriptor {
                entries: bind_group_entries[group_range].into(),
                layout: *bind_group_layout,
            };

            let bind_group = device.get_or_create_bind_group(&descriptor).unwrap();
            storage_groups.push(bind_group);
        }
        storage_groups
    }

    //TODO: pool this
    pub fn create_uniform_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        buf: &GPUBuffer,
    ) -> wgpu::BindGroup {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buf.inner,
                    offset: 0,
                    size: Some(std::num::NonZeroU64::new(UNIFORM_ALIGN as _).unwrap()),
                }),
            }],
        });
        bind_group
    }
}
