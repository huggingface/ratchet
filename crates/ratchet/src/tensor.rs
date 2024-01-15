use crate::{
    ops::*, DType, Device, DeviceError, Enforcer, Operation, RawStorage, Shape, Storage, Strides,
    TensorDType,
};
use crate::{BinaryOp, LazyOp};
use parking_lot::RwLock;
use std::sync::Arc;

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
#[derive(Clone)]
pub struct Tensor {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("id", &self.inner.id)
            .field("op", &self.inner.op)
            .field("shape", &self.inner.shape)
            .field("strides", &self.inner.strides)
            .field("dt", &self.inner.dt)
            .field("storage", &self.inner.storage)
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

#[derive(Debug)]
pub struct Inner {
    id: TensorId,
    op: LazyOp,
    shape: Shape,
    strides: Strides,
    dt: DType,
    storage: Arc<RwLock<Storage>>,
}

impl AsRef<Inner> for Inner {
    fn as_ref(&self) -> &Inner {
        self
    }
}

impl Inner {
    fn new(op: LazyOp, dt: DType, shape: Shape, strides: Strides, storage: Storage) -> Self {
        Self {
            id: TensorId::new(),
            dt,
            shape,
            strides,
            op,
            storage: Arc::new(RwLock::new(storage)),
        }
    }
}

impl Tensor {
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn dt(&self) -> DType {
        self.dt
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        //Enforce valid shape, dtype, device
        //Determine output shape
        //Binary::check_invariants(self, other);
        //Binary::shape_inference(self.shape(), other.shape());
        let op = LazyOp::Binary(Binary::new(self.clone(), other.clone(), BinaryOp::Add));
        //TODO: real shapes & strides
        let device = self.storage.try_read().unwrap().device().clone();
        let storage = Storage::new(device, None);
        let inner = Inner::new(
            op,
            self.dt,
            self.shape.clone(),
            self.strides.clone(),
            storage,
        );
        Tensor {
            inner: Arc::new(inner),
        }
    }

    /// Creates a data from a vector.
    ///
    /// If a non-CPU device is specified, the data will be copied to the device.
    pub fn from_vec<T: TensorDType>(data: Vec<T>, shape: Shape, device: Device) -> Tensor {
        let strides = Strides::from(&shape);
        //TODO: allow creating straight on the GPU
        let storage = Storage::from_slice(&data, &shape);
        let inner = Inner::new(LazyOp::Const, T::dt(), shape, strides, storage);
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

    pub fn resolve(&self) {
        println!("Order: {:#?}", self.execution_order());
        let mut compiled_ops = vec![];
        for t in self.execution_order() {
            compiled_ops.push(t.op.compile()); //Compile on Op or tensor?
        }
        //Execute kernels

        //Return result
    }
}

#[cfg(test)]
mod tests {
    use crate::shape;

    use super::*;

    #[test]
    fn test_cfg() {
        let a = Tensor::from_vec(vec![1, 2, 3, 4], shape![2, 2], Device::CPU);
        let b = Tensor::from_vec(vec![1, 2, 3, 4], shape![2, 2], Device::CPU);
        let c = a.add(&b);
        c.resolve();
    }
}
