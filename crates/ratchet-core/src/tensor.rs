use crate::gpu::{CpuUniform, WgpuDevice};
use crate::{
    ops::*, CPUBuffer, CompiledOp, DType, Device, DeviceStorage, Executable, GPUBuffer, Operation,
    OperationError, Shape, Storage, Strides, TensorDType, TensorId,
};
use crate::{BinaryOp, LazyOp};

use derive_new::new;
use parking_lot::{RwLock, RwLockReadGuard};
use std::sync::Arc;

#[cfg(feature = "rand")]
use {rand::prelude::*, rand_distr::StandardNormal};

#[cfg(feature = "pyo3")]
use {
    ndarray::{ArrayD, ArrayViewD},
    numpy::PyArrayDyn,
};

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
    fn new(op: LazyOp, meta: StorageView, storage: Option<Storage>, device: Device) -> Self {
        Self {
            inner: Arc::new(Inner::new(op, meta, storage, device)),
        }
    }

    fn lazy(op: LazyOp, meta: StorageView, device: Device) -> Self {
        Self::new(op, meta, None, device)
    }

    fn update_storage(&self, storage: Storage) {
        *self.inner.storage.write() = Some(storage);
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let storage_fmt = self.storage().as_ref().map(|s| s.dump(self.dt(), false));
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
    storage: Arc<RwLock<Option<Storage>>>,
}

impl AsRef<Inner> for Inner {
    fn as_ref(&self) -> &Inner {
        self
    }
}

impl Inner {
    fn new(op: LazyOp, meta: StorageView, storage: Option<Storage>, device: Device) -> Self {
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

    pub fn storage(&self) -> RwLockReadGuard<Option<Storage>> {
        self.inner.storage.read()
    }

    pub fn resolved(&self) -> bool {
        self.storage().is_some()
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

    #[cfg(feature = "rand")]
    pub fn randn<T: TensorDType + num_traits::Float>(shape: Shape, device: Device) -> Self {
        let mut rng = rand::thread_rng();
        //TODO: fix copy on CPU
        let data = (0..shape.numel())
            .map(|_| {
                let sample: f32 = StandardNormal.sample(&mut rng);
                T::from(sample).expect("Failed to convert sample")
            })
            .collect::<Vec<_>>();
        Self::from_data(data, shape, device)
    }

    /// Creates a new tensor from a chunk of data.
    ///
    /// The Tensor is instantly resolved.
    /// If a non-CPU device is specified, the data will be copied to the device.
    pub fn from_data<T: TensorDType, U: AsRef<[T]>>(
        data: U,
        shape: Shape,
        device: Device,
    ) -> Tensor {
        let storage = Storage::from_slice(data.as_ref(), &shape, &device);
        let strides = Strides::from(&shape);
        let meta = StorageView::new(shape, T::dt(), strides);
        Tensor::new(LazyOp::Const, meta, Some(storage), device)
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
            LazyOp::Binary(b) => b.compile(self, uniform, device).ok(),
            LazyOp::Matmul(m) => m.compile(self, uniform, device).ok(),
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
                let pooled_buffer = allocations.get(&id).ok_or(TensorError::NoStorage(id))?;
                assert!(t.device().is_gpu());

                let storage = Storage::GPU(GPUBuffer {
                    inner: pooled_buffer.clone(),
                    alignment: t.dt().size_of(),
                });
                t.update_storage(storage);
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

    fn to_cpu(&self) -> Result<Tensor, TensorError> {
        if self.device().is_cpu() || !self.resolved() {
            return Ok(self.clone());
        }
        let storage_guard = self.storage();
        let gpu_buf = storage_guard
            .as_ref()
            .ok_or(TensorError::TransferError)?
            .try_gpu()?;
        let cpu_buf = gpu_buf.to_cpu(&self.device)?;

        Ok(Tensor::new(
            LazyOp::Const,
            self.view.clone(),
            Some(Storage::CPU(cpu_buf)),
            Device::CPU,
        ))
    }

    fn to_gpu(&self, dst_device: &Device) -> Result<Tensor, TensorError> {
        if self.device().is_gpu() || !self.resolved() {
            return Ok(self.clone());
        }
        let storage_guard = self.storage();
        let cpu_buf = storage_guard
            .as_ref()
            .ok_or(TensorError::TransferError)?
            .try_cpu()?;
        let gpu_buf = cpu_buf.to_device(dst_device)?;

        let wgpu_device = dst_device.try_gpu()?;
        Ok(Tensor::new(
            LazyOp::Const,
            self.view.clone(),
            Some(Storage::GPU(gpu_buf)),
            Device::GPU(wgpu_device.clone()),
        ))
    }

    /// Transfers the tensor to the specified device.
    ///
    /// If the tensor is already on the specified device, it will be returned as-is,
    /// and the underlying storage will not be copied.
    /// If the tensor is on a different device, it will be copied to the specified device.
    pub fn to(&self, device: Device) -> Result<Tensor, TensorError> {
        match (self.device(), &device) {
            (Device::GPU(_), Device::CPU) => self.to_cpu(),
            (Device::CPU, Device::GPU(_)) => self.to_gpu(&device),
            _ => Ok(self.clone()),
        }
    }

    #[cfg(feature = "pyo3")]
    pub fn into_ndarray<T: TensorDType>(self) -> ArrayD<T> {
        assert!(self.device().is_cpu());
        let shape = self.shape().to_vec();
        if self.num_bytes() != 0 {
            let storage_guard = self.storage();
            let buffer = storage_guard.as_ref().unwrap().try_cpu().unwrap();
            let (ptr, _) = buffer.inner().into_raw_parts();
            println!("INTO NDARRAY: {:?}", ptr);
            unsafe { ArrayViewD::from_shape_ptr(shape, ptr as *const T).to_owned() }
        } else {
            ArrayViewD::from_shape(shape, &[]).unwrap().to_owned()
        }
    }

    #[cfg(feature = "pyo3")]
    pub fn to_py<'s, 'p: 's, T: TensorDType + numpy::Element>(
        &'s self,
        py: &'p pyo3::Python<'p>,
    ) -> &PyArrayDyn<T> {
        use numpy::PyArray;
        println!("TO PY: {:?}", self);
        let cloned = self.deep_clone();
        println!("CLONED: {:?}", cloned);
        PyArray::from_owned_array(*py, cloned.into_ndarray::<T>())
    }

    pub fn deep_clone(&self) -> Tensor {
        let storage_guard = self.storage();
        let storage = storage_guard.as_ref().unwrap();
        let cloned_storage = storage.deep_clone(self.device()).unwrap();
        Tensor::new(
            LazyOp::Const,
            self.view.clone(),
            Some(cloned_storage),
            self.device.clone(),
        )
    }
}

#[cfg(feature = "pyo3")]
impl<T: TensorDType> From<ArrayD<T>> for Tensor {
    fn from(it: ArrayD<T>) -> Self {
        if it.as_slice().is_some() {
            let layout = std::alloc::Layout::from_size_align(
                it.len() * std::mem::size_of::<T>(),
                std::mem::align_of::<T>(),
            )
            .unwrap();
            let shape = it.shape().to_vec().into();
            let strides = Strides::from(&shape);
            let vec = it.into_raw_vec().into_boxed_slice();
            let ptr = Box::into_raw(vec) as *mut u8;

            let cpu_buf = CPUBuffer::from_raw_parts(ptr, layout);
            let meta = StorageView::new(shape, T::dt(), strides);
            Tensor::new(
                LazyOp::Const,
                meta,
                Some(Storage::CPU(cpu_buf)),
                Device::CPU,
            )
        } else {
            panic!("Cannot convert numpy array with non-contiguous memory layout to tensor");
        }
    }
}

#[cfg(feature = "pyo3")]
impl<T: TensorDType + numpy::Element> From<&PyArrayDyn<T>> for Tensor {
    fn from(array: &PyArrayDyn<T>) -> Self {
        Self::from(array.to_owned_array())
    }
}

#[cfg(test)]
mod tests {
    use pyo3::{types::PyModule, Python};

    use crate::{shape, DeviceRequest};

    use super::*;

    #[test]
    fn test_matmul() -> anyhow::Result<()> {
        let device = Device::request_device(DeviceRequest::GPU)?;
        let a = Tensor::randn::<f32>(shape![1024, 1024], device.clone());
        let b = Tensor::randn::<f32>(shape![1024, 1024], device.clone());
        let c = a.matmul(&b)?;
        c.resolve()?;
        println!("\nA: {:#?}", a);
        println!("\nB: {:#?}", b);
        println!("\nC: {:#?}", c);
        let d = c.to(Device::CPU)?;
        println!("\nD: {:#?}", d);
        Ok(())
    }

    #[test]
    fn test_pyo3() -> anyhow::Result<()> {
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let a = Tensor::randn::<f32>(shape![1024, 1024], cpu_device.clone());
        let b = Tensor::randn::<f32>(shape![1024, 1024], cpu_device.clone());

        let ground: anyhow::Result<Tensor> = Python::with_gil(|py| {
            let prg = PyModule::from_code(
                py,
                r#"
import torch

def matmul(a, b):
    return torch.matmul(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    "#,
                "x.py",
                "x",
            )?;

            let result = prg
                .getattr("matmul")?
                .call1((a.to_py::<f32>(&py), b.to_py::<f32>(&py)))?
                .extract::<&PyArrayDyn<f32>>()?;
            Ok(Tensor::from(result))
        });
        println!("\nTORCH: {:#?}", ground);

        println!("\nA: {:#?}", a);
        println!("\nB: {:#?}", b);

        let gpu_device = Device::request_device(DeviceRequest::GPU)?;
        let a = a.to(gpu_device.clone())?;
        let b = b.to(gpu_device)?;

        let c = a.matmul(&b)?;
        c.resolve()?;

        let our_result = c.to(cpu_device)?;
        println!("\nOURS: {:#?}", our_result);

        Ok(())
    }
}
