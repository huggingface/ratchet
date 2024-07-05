use crate::gpu::{GpuUniform, PoolError, StaticResourcePoolAccessor, WgpuDevice};
use crate::{CompiledOp, Device, Tensor};
use derive_new::new;
#[cfg(not(debug_assertions))]
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DownloadBuffer;
use wgpu::{CommandEncoder, SubmissionIndex};

/// # Executable
///
/// A linear sequence of compiled operations, with a single uniform buffer
/// containing metadata for all operations.
#[derive(new)]
pub struct Executable<'t> {
    steps: Vec<CompiledOp>,
    gpu_uniform: GpuUniform,
    #[cfg(debug_assertions)]
    debug_list: Vec<&'t Tensor>,
    #[cfg(not(debug_assertions))]
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

    #[cfg(debug_assertions)]
    pub(crate) fn dispatch_debugging(
        &self,
        device: &WgpuDevice,
    ) -> Result<SubmissionIndex, ExecutionError> {
        use crate::{CPUBuffer, DeviceStorage};

        let pipeline_resources = device.pipeline_resources();
        assert!(self.debug_list.len() == self.steps.len());

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

            let result_t = self.debug_list[step_index].clone();
            let result_s = result_t.storage();
            let result_storage = result_s.as_ref().unwrap();
            let result_gpu = result_storage.try_gpu();
            let result_buf = &result_gpu.unwrap().inner;

            let debug_buffer = step.debug_buffer.as_ref().unwrap();
            encoder.copy_buffer_to_buffer(result_buf, 0, debug_buffer, 0, debug_buffer.size());

            let index = device.queue().submit(Some(encoder.finish()));
            last_index = Some(index);
        }

        for (step_index, step) in self.steps.iter().enumerate() {
            let dt = self.debug_list[step_index].dt();
            let debug_buffer = step.debug_buffer.as_ref().unwrap();
            let buffer_slice = debug_buffer.slice(..);
            let alignment = dt.size_of();
            let (tx, rx) = std::sync::mpsc::channel();

            wgpu::util::DownloadBuffer::read_buffer(
                device,
                device.queue(),
                &buffer_slice,
                move |buffer| {
                    tx.send(match buffer {
                        Ok(db) => Ok(CPUBuffer::from_bytes(&db, alignment)),
                        Err(error) => Err(error),
                    })
                    .expect("Failed to send result of read_buffer");
                },
            );
            device.poll(wgpu::Maintain::Wait);
            let storage = rx.recv().unwrap().unwrap();
            let kernel_key = &step.kernel_key;
            log::debug!("{:?}\n {:?}", kernel_key, storage.dump(dt, false));
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
