use crate::gpu::WgpuDevice;
use crate::CompiledOp;
use wgpu::SubmissionIndex;

pub struct Executable;

impl Executable {
    #[cfg(not(feature = "gpu-profiling"))]
    fn dispatch_operations(steps: Vec<CompiledOp>, device: &WgpuDevice) -> SubmissionIndex {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            for step in steps {
                cpass.set_pipeline(step.pipeline());
                for (index, bind_group) in step.storage_groups().iter().enumerate() {
                    cpass.set_bind_group(index as u32, bind_group, &[]);
                }
                let uniform_index = step.storage_groups().len() as u32;
                cpass.set_bind_group(uniform_index, uniform_group, &[step.offset()]);
                let [x_count, y_count, z_count] = step.workgroup_count().as_slice();
                cpass.dispatch_workgroups(x_count, y_count, z_count);
            }
        }
        device.queue().submit(Some(encoder.finish()))
    }
}
