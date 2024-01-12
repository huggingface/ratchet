use crate::gpu::WgpuDevice;

#[derive(Clone)]
pub enum Device {
    CPU,
    GPU(WgpuDevice),
}
