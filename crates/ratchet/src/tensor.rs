use crate::gpu::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntryExt, BufferDescriptor, CpuUniform,
    StaticResourcePoolAccessor,
};
use crate::{
    ops::*, rvec, CompiledOp, DType, Device, Executable, Operation, RVec, RawCPUBuffer,
    RawGPUBuffer, RawStorage, Shape, Storage, Strides, TensorDType,
};
use crate::{BinaryOp, LazyOp};
use bytemuck::NoUninit;
use parking_lot::RwLock;
use std::sync::Arc;
use wgpu::BufferUsages;

/// Unique identifier for tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(usize);

impl TensorId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

/// A multi-dimensional array of data.
///
/// A tensor is a lazy representation of an operation. It, and the nodes required to compute it's
/// value, will not be computed until `resolve` is called.
///
/// After resolving, the Tensor is a logical view of a Storage object.
#[derive(Clone)]
pub struct Tensor {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let storage = self.storage().try_read().unwrap();
        f.debug_struct("Tensor")
            .field("id", &self.inner.id)
            .field("op", &self.inner.op)
            .field("storage", &storage.dump(self.dt(), false))
            .finish()
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.inner.id == other.inner.id
    }
}

impl std::ops::Deref for Tensor {
    type Target = Inner;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref()
    }
}

#[derive(Debug, Clone)]
pub struct Metadata {
    shape: Shape,
    dt: DType,
    strides: Strides,
}

#[derive(Debug)]
pub struct Inner {
    id: TensorId,
    op: LazyOp,
    meta: Metadata,
    device: Device,
    storage: Arc<RwLock<Storage>>,
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl AsRef<Inner> for Inner {
    fn as_ref(&self) -> &Inner {
        self
    }
}

impl Inner {
    fn new(op: LazyOp, meta: Metadata, storage: Storage, device: Device) -> Self {
        Self {
            id: TensorId::new(),
            meta,
            op,
            device,
            storage: Arc::new(RwLock::new(storage)),
        }
    }
}

impl Tensor {
    pub fn id(&self) -> TensorId {
        self.inner.id
    }

    pub fn rank(&self) -> usize {
        self.meta.shape.len()
    }

    pub fn dt(&self) -> DType {
        self.meta.dt
    }

    pub fn shape(&self) -> &Shape {
        &self.meta.shape
    }

    pub fn num_bytes(&self) -> usize {
        self.meta.shape.numel() * self.meta.dt.size_of()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn storage(&self) -> &Arc<RwLock<Storage>> {
        &self.storage
    }

    pub fn resolved(&self) -> bool {
        self.storage().try_read().unwrap().raw().is_some()
    }

    pub(crate) fn set_storage(&self, storage: Storage) {
        *self.storage().write() = storage;
    }

    pub fn srcs(&self) -> RVec<&Tensor> {
        match &self.inner.op {
            LazyOp::Const => rvec![],
            LazyOp::Binary(b) => b.srcs(),
            _ => unimplemented!(),
        }
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        //Enforce valid shape, dtype, device
        //Determine output shape
        //Binary::check_invariants(self, other);
        //Binary::shape_inference(self.shape(), other.shape());
        let op = LazyOp::Binary(Binary::new(self.clone(), other.clone(), BinaryOp::Add));
        //TODO: real shapes & strides
        let storage = Storage::new(None);
        let inner = Inner::new(op, self.meta.clone(), storage, self.device.clone());
        Tensor {
            inner: Arc::new(inner),
        }
    }

    /// Creates a data from a vector.
    ///
    /// If a non-CPU device is specified, the data will be copied to the device.
    pub fn from_vec<T: TensorDType>(data: Vec<T>, shape: Shape, device: Device) -> Tensor {
        let strides = Strides::from(&shape);
        let storage = Storage::from_slice(&data, &shape, &device);
        let meta = Metadata {
            shape,
            dt: T::dt(),
            strides,
        };
        let inner = Inner::new(LazyOp::Const, meta, storage, device);
        Tensor {
            inner: Arc::new(inner),
        }
    }

    fn execution_order(&self) -> Vec<Tensor> {
        let mut stack = vec![self.clone()];
        let mut visited = vec![];
        while let Some(tensor) = stack.pop() {
            if visited.contains(&tensor) {
                continue;
            }
            match &tensor.inner.op {
                LazyOp::Const => {}
                LazyOp::Binary(b) => {
                    let sources = b.srcs();
                    stack.push(sources[0].clone());
                    stack.push(sources[1].clone());
                }
                _ => unimplemented!(),
            }
            visited.push(tensor);
        }
        visited.reverse();
        visited
    }

    //TODO: massively refactor, just seeing if it can work for now
    pub fn resolve(&self) {
        let mut compiled_ops = vec![];
        let mut uniform = CpuUniform::new();
        let device = self.device().is_gpu().unwrap();

        //Here we need to do memory allocation
        //Root nodes should be constants or user provided inputs
        let execution_order = self.execution_order();

        let mut allocations = self
            .device()
            .is_gpu()
            .unwrap()
            .allocate_intermediates(&execution_order, &device);

        //Allocate for leaf node (ourselves)
        allocations.insert(
            self.id(),
            device
                .allocate_buffer(&BufferDescriptor {
                    size: self.num_bytes() as _,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
                .unwrap(),
        );

        for t in execution_order {
            if !t.resolved() {
                let storage = Storage::new(Some(RawStorage::from(
                    allocations.get(&t.id()).unwrap().clone(),
                )));
                t.set_storage(storage);
            }

            if let Some((pipeline_handle, wgc, offset)) = t.op.compile(&device, &mut uniform) {
                let storage_layout = t.op.storage_layout(&device);
                let storage_bind_groups = CompiledOp::create_storage_bind_groups(
                    &t.op.srcs(),
                    &rvec![&t],
                    rvec![storage_layout],
                    &device,
                );

                compiled_ops.push(CompiledOp::new(
                    wgc,
                    pipeline_handle,
                    storage_bind_groups,
                    offset,
                ))
            }
        }
        let gpu_uniform = device.create_uniform_init(uniform);
        let gpu_uniform_bind_group = CompiledOp::create_uniform_bind_group(
            &device,
            &device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry::dynamic_uniform_buffer()],
            }),
            &gpu_uniform,
        );
        let executable = Executable::new(compiled_ops, gpu_uniform, gpu_uniform_bind_group);
        let index = executable.dispatch_operations(&device);
        device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(index));

        self.into_cpu();
    }

    fn read_to_host<A: NoUninit>(shape: Shape, dt: DType, bytes: &[A]) -> Storage {
        match dt {
            DType::F32 => {
                Storage::from_slice::<f32>(bytemuck::cast_slice(bytes), &shape, &Device::CPU)
            }
            DType::I32 => {
                Storage::from_slice::<i32>(bytemuck::cast_slice(bytes), &shape, &Device::CPU)
            }
            _ => todo!(
                "Attempted to read GPU tensor to host with unsupported dtype: {:?}",
                dt
            ),
        }
    }

    async fn into_cpu_inner(&self) {
        let device = self.device().is_gpu().unwrap();
        let shape = self.shape().clone();
        let dt = self.dt();

        let gpu_storage = {
            let storage_resource = self.storage().try_read().unwrap();
            storage_resource.try_gpu().unwrap().clone()
        };
        if !gpu_storage.usage().contains(BufferUsages::COPY_SRC) {
            panic!("Attempted to read GPU tensor to host without COPY_SRC usage")
        }
        let buffer_slice = gpu_storage.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();

        wgpu::util::DownloadBuffer::read_buffer(
            device,
            device.queue(),
            &buffer_slice,
            move |buffer| {
                // Called on download completed
                tx.send(match buffer {
                    Ok(db) => Ok(Self::read_to_host(shape, dt, &db)),
                    Err(error) => Err(error),
                })
                .unwrap();
            },
        );
        device.poll(wgpu::Maintain::Wait);
        let storage = rx.receive().await.unwrap();
        self.set_storage(storage.unwrap());
        //TOOD: update device here!!!
    }

    ///Consumes the GPU tensor and returns a CPU tensor
    pub fn into_cpu(&self) {
        pollster::block_on(self.into_cpu_inner())
    }
}

#[cfg(test)]
mod tests {
    use crate::{shape, DeviceRequest};

    use super::*;

    #[test]
    fn test_cfg() {
        let device = Device::request_device(DeviceRequest::GPU);
        println!("{:#?}", device);
        let a = Tensor::from_vec(vec![1., 2., 3., 4.], shape![2, 2], device.clone());
        let b = Tensor::from_vec(vec![55.], shape![1], device);
        let c = a.add(&b);
        c.resolve();
        println!("\nA: {:#?}", a);
        println!("\nB: {:#?}", b);
        println!("\nC: {:#?}", c);
    }
}
