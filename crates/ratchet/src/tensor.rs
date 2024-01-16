use crate::gpu::{BufferDescriptor, BufferUsagesExt, CpuUniform};
use crate::{
    ops::*, rvec, CompiledOp, DType, Device, Executable, Operation, RawStorage, Shape, Storable,
    Storage, Strides, TensorDType, TensorId,
};
use crate::{BinaryOp, LazyOp};
use derive_new::new;
use parking_lot::RwLock;
use std::sync::Arc;
use wgpu::BufferUsages;

// thiserror error for Tensor
#[derive(thiserror::Error, Debug)]
pub enum TensorError {
    #[error("Tensor is not resolved")]
    NotResolved,
    #[error("Tensor {0:?} is missing storage")]
    NoStorage(TensorId),
    #[error(transparent)]
    DeviceError(#[from] crate::DeviceError),
    #[error("Failed to transfer data to host")]
    TransferError,
}

/// A multi-dimensional array of data.
///
/// A tensor is a lazy representation of an operation. It, and the nodes required to compute it's
/// value, will not be computed until `resolve` is called.
#[derive(Clone)]
pub struct Tensor {
    inner: Arc<Inner>,
}

impl Tensor {
    fn new(inner: Inner) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let storage = self.storage().try_read().expect("Could not read storage");
        let storage_fmt = storage.dump(self.dt(), false);
        let (id, op) = (self.id(), self.op());
        f.debug_struct("Tensor")
            .field("id", &id)
            .field("op", &op)
            .field("storage", &storage_fmt)
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

//TODO: method to go from storage -> Inner
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

    fn from_move(&self, storage: Storage, device: Device) -> Self {
        Self {
            id: TensorId::new(),
            meta: self.meta.clone(),
            op: self.op.clone(),
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

    /// # Safety
    ///
    /// Make sure your device & storage are compatible.
    pub(crate) unsafe fn set_storage(&self, storage: Storage) {
        *self.storage().write() = storage;
    }

    pub(crate) fn op(&self) -> &LazyOp {
        &self.inner.op
    }
}

impl Tensor {
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
        Tensor::new(Inner::new(LazyOp::Const, meta, storage, device))
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
    pub fn resolve(&self) -> Result<(), TensorError> {
        let mut uniform = CpuUniform::new();
        let device = self.device().try_gpu()?;

        let execution_order = self.execution_order();
        let mut compiled_ops = Vec::with_capacity(execution_order.len());

        let mut allocations = device.allocate_cfg(&execution_order, device)?;

        //Allocate for leaf node (ourselves)
        //TODO: remove
        allocations.insert(
            self.id(),
            device.allocate_buffer(&BufferDescriptor {
                size: self.num_bytes() as _,
                usage: BufferUsages::standard(),
                mapped_at_creation: false,
            })?,
        );

        for t in execution_order {
            if !t.resolved() {
                let id = t.id();
                let gpu_buf = allocations.get(&id).ok_or(TensorError::NoStorage(id))?;
                assert!(t.device().is_gpu());
                unsafe {
                    t.set_storage(Storage::from(RawStorage::from_gpu(gpu_buf.clone(), t.dt())));
                }
            }

            if let Some((pipeline_handle, wgc, offset)) = t.op.compile(device, &mut uniform) {
                let storage_layout = t.op.storage_layout(device);
                //TODO: this is ugly
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
        Ok(())
    }

    async fn to_cpu(&self) -> Result<Tensor, TensorError> {
        let raw_gpu_buf = {
            let storage_resource = self.storage().try_read().ok_or(TensorError::NotResolved)?;
            storage_resource.try_gpu()?.clone()
        };
        let cpu_storage = Storage::from(raw_gpu_buf.to_cpu(self.device()).unwrap());
        Ok(Tensor::new(self.inner.from_move(cpu_storage, Device::CPU)))
    }

    pub fn to(&self, device: Device) -> Result<Tensor, TensorError> {
        match (self.device(), device) {
            (Device::GPU(_), Device::CPU) => pollster::block_on(self.to_cpu()),
            (Device::CPU, Device::GPU(_)) => todo!(),
            _ => Ok(self.clone()),
        }
    }
}

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
        c.resolve().unwrap();
        println!("\nA: {:#?}", a);
        println!("\nB: {:#?}", b);
        println!("\nC: {:#?}", c);
        let d = c.to(Device::CPU).unwrap();
        println!("\nD: {:#?}", d);
    }
}
