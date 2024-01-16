use crate::gpu::{BindGroupLayoutEntryExt, BufferDescriptor, BufferUsagesExt, CpuUniform};
use crate::{
    ops::*, rvec, CompiledOp, DType, Device, Executable, Operation, RVec, RawStorage, Shape,
    Storage, Strides, TensorDType, TensorId,
};
use crate::{BinaryOp, LazyOp};
use derive_new::new;
use parking_lot::RwLock;
use std::sync::Arc;
use wgpu::BufferUsages;

/// A multi-dimensional array of data.
///
/// A tensor is a lazy representation of an operation. It, and the nodes required to compute it's
/// value, will not be computed until `resolve` is called.
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

/// # Metadata
///
/// All of the below field can be thought of as a "view" of the underlying storage.
#[derive(new, Debug, Clone)]
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
        //TODO: real shapes & strides
        let op = LazyOp::Binary(Binary::new(self.clone(), other.clone(), BinaryOp::Add));
        Tensor {
            inner: Inner::new(op, self.meta.clone(), Storage::empty(), self.device.clone()).into(),
        }
    }

    /// Creates a new tensor from a vector of data.
    ///
    /// The Tensor is instantly resolved.
    /// If a non-CPU device is specified, the data will be copied to the device.
    pub fn from_vec<T: TensorDType>(data: Vec<T>, shape: Shape, device: Device) -> Tensor {
        let storage = Storage::from_slice(&data, &shape, &device);
        let strides = Strides::from(&shape);
        let meta = Metadata::new(shape, T::dt(), strides);
        Tensor {
            inner: Inner::new(LazyOp::Const, meta, storage, device).into(),
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
        let device = self.device().try_gpu().unwrap();

        let execution_order = self.execution_order();

        let mut allocations = device
            .allocate_intermediates(&execution_order, device)
            .unwrap();

        //Allocate for leaf node (ourselves)
        allocations.insert(
            self.id(),
            device
                .allocate_buffer(&BufferDescriptor {
                    size: self.num_bytes() as _,
                    usage: BufferUsages::standard(),
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

            if let Some((pipeline_handle, wgc, offset)) = t.op.compile(device, &mut uniform) {
                let storage_layout = t.op.storage_layout(device);
                let storage_bind_groups = CompiledOp::create_storage_bind_groups(
                    &t.op.srcs(),
                    &rvec![&t],
                    rvec![storage_layout],
                    device,
                );

                compiled_ops.push(CompiledOp::new(
                    wgc,
                    pipeline_handle,
                    storage_bind_groups,
                    offset,
                ))
            }
        }
        let executable = Executable::new(compiled_ops, uniform.into_gpu(device));
        let index = executable.dispatch_operations(device);
        device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(index));
    }

    async fn to_cpu_inner(&self) {
        let device = self.device().try_gpu().unwrap();
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
                tx.send(match buffer {
                    Ok(db) => Ok(Storage::read_to_host(shape, dt, &db)),
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

    pub fn to_cpu(&self) {
        pollster::block_on(self.to_cpu_inner())
    }
}

impl Inner {}

#[cfg(test)]
mod tests {
    use crate::{shape, DeviceRequest};

    use super::*;

    #[test]
    fn test_cfg() {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let a = Tensor::from_vec(vec![1., 2., 3., 4.], shape![2, 2], device.clone());
        let b = Tensor::from_vec(vec![55.], shape![1], device);
        let c = a.add(&b);
        c.resolve();
        println!("\nA: {:#?}", a);
        println!("\nB: {:#?}", b);
        println!("\nC: {:#?}", c);
        c.to_cpu();
        println!("\nC CPU: {:#?}", c);
    }
}
