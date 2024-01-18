use crate::gpu::{CpuUniform, WgpuDevice};
use crate::{
    ops::*, CompiledOp, DType, Device, DeviceStorage, Executable, Operation, OperationError,
    RawStorage, Shape, Storage, Strides, TensorDType, TensorId,
};
use crate::{BinaryOp, LazyOp};
use derive_new::new;
use parking_lot::RwLock;
use std::sync::Arc;

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
    #[error(transparent)]
    OperationError(#[from] OperationError),
}

/// A multi-dimensional array of data.
///
/// A tensor is a lazy representation of an operation. The nodes required to compute it's
/// value and it's own value will not be computed until `resolve` is called.
#[derive(Clone)]
pub struct Tensor {
    inner: Arc<Inner>,
}

impl Tensor {
    fn new(op: LazyOp, meta: StorageView, storage: Storage, device: Device) -> Self {
        let inner = Inner::new(op, meta, storage, device);
        Self {
            inner: Arc::new(inner),
        }
    }

    fn lazy(op: LazyOp, meta: StorageView, device: Device) -> Self {
        Self::new(op, meta, Storage::empty(), device)
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

#[derive(new, Debug, Clone)]
pub struct StorageView {
    shape: Shape,
    dt: DType,
    strides: Strides,
}

#[derive(Debug)]
pub struct Inner {
    id: TensorId,
    op: LazyOp,
    device: Device,
    view: StorageView,
    storage: Arc<RwLock<Storage>>,
}

impl AsRef<Inner> for Inner {
    fn as_ref(&self) -> &Inner {
        self
    }
}

impl Inner {
    fn new(op: LazyOp, meta: StorageView, storage: Storage, device: Device) -> Self {
        Self {
            id: TensorId::new(),
            view: meta,
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

    pub fn view(&self) -> &StorageView {
        &self.view
    }

    pub fn rank(&self) -> usize {
        self.view.shape.len()
    }

    pub fn dt(&self) -> DType {
        self.view.dt
    }

    pub fn shape(&self) -> &Shape {
        &self.view.shape
    }

    pub fn num_bytes(&self) -> usize {
        self.view.shape.numel() * self.view.dt.size_of()
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
    pub fn add(&self, other: &Tensor) -> anyhow::Result<Tensor> {
        Binary::check_invariants(&[self, other])?;

        let binary = Binary::new(self.clone(), other.clone(), BinaryOp::Add);
        let new_view = binary.infer_output(&[self, other])?;
        Ok(Tensor::lazy(
            LazyOp::Binary(binary),
            new_view,
            self.device.clone(),
        ))
    }

    pub fn matmul(&self, other: &Tensor) -> anyhow::Result<Tensor> {
        Matmul::check_invariants(&[self, other])?;

        let matmul = Matmul::new(self.clone(), other.clone());
        let new_view = matmul.infer_output(&[self, other])?;
        Ok(Tensor::lazy(
            LazyOp::Matmul(matmul),
            new_view,
            self.device.clone(),
        ))
    }

    /// Creates a new tensor from a vector of data.
    ///
    /// The Tensor is instantly resolved.
    /// If a non-CPU device is specified, the data will be copied to the device.
    pub fn from_vec<T: TensorDType>(data: Vec<T>, shape: Shape, device: Device) -> Tensor {
        let storage = Storage::from_slice(&data, &shape, &device);
        let strides = Strides::from(&shape);
        let meta = StorageView::new(shape, T::dt(), strides);
        Tensor::new(LazyOp::Const, meta, storage, device)
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
                LazyOp::Matmul(m) => {
                    let sources = m.srcs();
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

    pub fn compile(&self, uniform: &mut CpuUniform, device: &WgpuDevice) -> Option<CompiledOp> {
        match self.op() {
            LazyOp::Binary(b) => Some(b.compile(self, uniform, device).unwrap()),
            LazyOp::Matmul(m) => Some(m.compile(self, uniform, device).unwrap()),
            LazyOp::Const => None,
            _ => unimplemented!(),
        }
    }

    pub fn resolve(&self) -> Result<(), TensorError> {
        let mut uniform = CpuUniform::new();
        let device = self.device().try_gpu()?;

        let execution_order = self.execution_order();
        let mut compiled_ops = Vec::with_capacity(execution_order.len());
        let allocations = device.allocate_cfg(&execution_order, device)?;

        for t in execution_order {
            if !t.resolved() {
                let id = t.id();
                let gpu_buf = allocations.get(&id).ok_or(TensorError::NoStorage(id))?;
                assert!(t.device().is_gpu());
                unsafe {
                    t.set_storage(Storage::from(RawStorage::from_gpu(gpu_buf.clone(), t.dt())));
                }
            }

            if let Some(compiled_op) = t.compile(&mut uniform, device) {
                compiled_ops.push(compiled_op);
            }
        }
        let executable = Executable::new(compiled_ops, uniform.into_gpu(device));
        let index = executable.dispatch_operations(device).unwrap();
        device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(index));
        Ok(())
    }

    async fn to_cpu(&self) -> Result<Tensor, TensorError> {
        let raw_gpu_buf = {
            let storage_resource = self.storage().try_read().ok_or(TensorError::NotResolved)?;
            storage_resource.try_gpu()?.clone()
        };
        let cpu_storage = Storage::from(raw_gpu_buf.to_cpu(self.device()).unwrap());

        Ok(Tensor::new(
            LazyOp::Const,
            self.view.clone(),
            cpu_storage,
            Device::CPU,
        ))
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
    fn test_cfg() -> anyhow::Result<()> {
        let device = Device::request_device(DeviceRequest::GPU)?;
        let a = Tensor::from_vec(vec![1., 2., 3., 4.], shape![2, 2], device.clone());
        let b = Tensor::from_vec(vec![1., 2., 3., 4.], shape![2, 2], device);
        let c = a.matmul(&b)?;
        c.resolve()?;
        println!("\nA: {:#?}", a);
        println!("\nB: {:#?}", b);
        println!("\nC: {:#?}", c);
        let d = c.to(Device::CPU)?;
        println!("\nD: {:#?}", d);
        Ok(())
    }
}
