use crate::gpu::{GpuUniform, PoolError, StaticResourcePoolAccessor, WgpuDevice};
use crate::CompiledOp;
use derive_new::new;
use wgpu::SubmissionIndex;

#[cfg(not(feature = "trace"))]
use std::marker::PhantomData;

#[cfg(feature = "trace")]
use crate::Tensor;

/// # Executable
///
/// A linear sequence of compiled operations, with a single uniform buffer
/// containing metadata for all operations.
#[derive(new)]
pub struct Executable<'t> {
    pub(crate) steps: Vec<CompiledOp>,
    pub(crate) gpu_uniform: GpuUniform,
    #[cfg(feature = "trace")]
    pub(crate) trace_list: Vec<&'t Tensor>,
    #[cfg(not(feature = "trace"))]
    _phantom: PhantomData<&'t ()>,
}

//this error ExecutionError
#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error(transparent)]
    PipelineNotFound(#[from] PoolError),
    #[error("Failed during debugging: {0}")]
    DebuggingError(&'static str),
}

impl Executable<'_> {
    #[cfg(not(feature = "gpu-profiling"))]
    pub fn dispatch(&self, device: &WgpuDevice) -> Result<SubmissionIndex, ExecutionError> {
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

    #[cfg(feature = "trace")]
    pub(crate) fn dispatch_trace(
        &self,
        device: &WgpuDevice,
    ) -> Result<SubmissionIndex, ExecutionError> {
        let pipeline_resources = device.pipeline_resources();
        assert!(self.trace_list.len() == self.steps.len());

        let mut last_index = None;
        for (step_index, step) in self.steps.iter().enumerate() {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            {
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
            }

            let result_t = self.trace_list[step_index].clone();
            let gpu_storage = result_t.storage();
            let result_buf = &gpu_storage
                .as_ref()
                .ok_or(ExecutionError::DebuggingError("Failed to get result buf."))?
                .try_gpu()
                .map_err(|_| ExecutionError::DebuggingError("Failed to get result buf."))?
                .inner;

            let trace_buffer = step
                .trace_buffer
                .as_ref()
                .ok_or(ExecutionError::DebuggingError(
                    "Failed to get debug buffer.",
                ))?;
            encoder.copy_buffer_to_buffer(result_buf, 0, trace_buffer, 0, trace_buffer.size());

            let index = device.queue().submit(Some(encoder.finish()));
            last_index = Some(index);
        }

        Ok(last_index.unwrap())
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
