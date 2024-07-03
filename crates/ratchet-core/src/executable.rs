use crate::gpu::{GpuUniform, PoolError, StaticResourcePoolAccessor, WgpuDevice};
use crate::{CompiledOp, OpDebugger};
use derive_new::new;
use wgpu::{CommandEncoder, SubmissionIndex};

/// # Executable
///
/// A linear sequence of compiled operations, with a single uniform buffer
/// containing metadata for all operations.
#[derive(new)]
pub struct Executable {
    steps: Vec<CompiledOp>,
    gpu_uniform: GpuUniform,
}

//this error ExecutionError
#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error(transparent)]
    PipelineNotFound(#[from] PoolError),
    #[error("Failed during debugging: {0}")]
    DebuggingError(&'static str),
}

impl Executable {
    pub(crate) fn write_debug(
        encoder: &mut CommandEncoder,
        op_debugger: &OpDebugger,
    ) -> Result<(), ExecutionError> {
        let storage_guard = op_debugger.dst_tensor.inner.storage();
        let storage = storage_guard
            .as_ref()
            .ok_or(ExecutionError::DebuggingError("Storage is None"))?;

        let gpu_storage = storage
            .try_gpu()
            .map_err(|_| ExecutionError::DebuggingError("GPU storage not available"))?;

        encoder.copy_buffer_to_buffer(
            &op_debugger.debug_buffer,
            0,
            &gpu_storage.inner,
            0,
            op_debugger.dst_tensor.num_bytes() as _,
        );
        Ok(())
    }

    #[cfg(not(feature = "gpu-profiling"))]
    pub fn dispatch_operations(
        &self,
        device: &WgpuDevice,
    ) -> Result<SubmissionIndex, ExecutionError> {
        let pipeline_resources = device.pipeline_resources();
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ratchet inference pass"),
                timestamp_writes: None,
            });
            for step in self.steps.iter() {
                cpass.set_pipeline(pipeline_resources.get(step.pipeline_handle())?);

                for (group_index, bind_group) in step.storage_groups().iter().enumerate() {
                    cpass.set_bind_group(group_index as u32, bind_group, &[]);
                }

                let uniform_group_index = step.storage_groups().len() as u32;
                let uniform_group = self.gpu_uniform.bind_group();
                cpass.set_bind_group(uniform_group_index, uniform_group, &[step.offset()]);

                let [x_count, y_count, z_count] = step.workgroup_count().as_slice();
                cpass.dispatch_workgroups(x_count, y_count, z_count);
            }
        }
        Ok(device.queue().submit(Some(encoder.finish())))
    }

    #[cfg(debug_assertions)]
    pub(crate) fn dispatch_debugging(
        &self,
        device: &WgpuDevice,
    ) -> Result<SubmissionIndex, ExecutionError> {
        let pipeline_resources = device.pipeline_resources();
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            for step in self.steps.iter() {
                //Create cpass for every step
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ratchet inference pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(pipeline_resources.get(step.pipeline_handle())?);

                for (group_index, bind_group) in step.storage_groups().iter().enumerate() {
                    cpass.set_bind_group(group_index as u32, bind_group, &[]);
                }

                let uniform_group_index = step.storage_groups().len() as u32;
                let uniform_group = self.gpu_uniform.bind_group();
                cpass.set_bind_group(uniform_group_index, uniform_group, &[step.offset()]);

                let [x_count, y_count, z_count] = step.workgroup_count().as_slice();
                cpass.dispatch_workgroups(x_count, y_count, z_count);

                #[cfg(debug_assertions)]
                {
                    if let Some(debugger) = &step.debugger {
                        Self::write_debug(&mut encoder, debugger)?;
                    }
                }
            }
        }
        Ok(device.queue().submit(Some(encoder.finish())))
    }

    #[cfg(feature = "gpu-profiling")]
    pub fn dispatch_operations(
        &self,
        device: &WgpuDevice,
    ) -> Result<SubmissionIndex, ExecutionError> {
        use crate::gpu::Profiler;

        let pipeline_resources = device.pipeline_resources();
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut profiler = Profiler::new(device.clone(), self.steps.len() as _);
        {
            for step in self.steps.iter() {
                let label = format!("{}_{}", step.kernel_key, step.workgroup_count().to_string());
                let timestamp_writes = Some(profiler.create_timestamp_queries(0, label.as_str()));
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes,
                });
                cpass.set_pipeline(pipeline_resources.get(step.pipeline_handle())?);

                for (group_index, bind_group) in step.storage_groups().iter().enumerate() {
                    cpass.set_bind_group(group_index as u32, bind_group, &[]);
                }

                let uniform_group_index = step.storage_groups().len() as u32;
                let uniform_group = self.gpu_uniform.bind_group();
                cpass.set_bind_group(uniform_group_index, uniform_group, &[step.offset()]);

                let [x_count, y_count, z_count] = step.workgroup_count().as_slice();
                cpass.dispatch_workgroups(x_count, y_count, z_count);
            }
        }

        profiler.resolve(&mut encoder);
        let index = device.queue().submit(Some(encoder.finish()));
        profiler.read_timestamps(true);
        Ok(index)
    }
}
