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
                cpass.set_bind_group(uniform_index, &self.uniform_group, &[step.offset()]);
                let (x_count, y_count, z_count) = step.workload().get_counts();
                cpass.dispatch_workgroups(x_count, y_count, z_count);
            }
        }
        self.handle.queue().submit(Some(encoder.finish()))
    }

    #[cfg(feature = "gpu-profiling")]
    fn dispatch_operations(&self) -> SubmissionIndex {
        let mut encoder = self
            .handle
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut profiler = Profiler::new(self.handle.clone(), self.steps.len() as _);

        {
            for op in self.steps.iter() {
                let timestamp_writes =
                    Some(profiler.create_timestamp_queries(op.node_id(), op.key()));

                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes,
                });
                cpass.set_pipeline(op.pipeline());
                for (index, bind_group) in op.storage_groups().iter().enumerate() {
                    cpass.set_bind_group(index as u32, bind_group, &[]);
                }
                let uniform_index = op.storage_groups().len() as u32;
                cpass.set_bind_group(uniform_index, &self.uniform_group, &[op.offset()]);
                let (x_count, y_count, z_count) = op.workload().get_counts();
                cpass.dispatch_workgroups(x_count, y_count, z_count);
            }
        }

        profiler.resolve(&mut encoder);
        let index = self.handle.queue().submit(Some(encoder.finish()));
        profiler.read_timestamps(true);
        index
    }
}
