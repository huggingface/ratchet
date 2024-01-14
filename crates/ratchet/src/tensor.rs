use crate::{
    ops::*, Device, DeviceError, Operation, RawStorage, Shape, Storage, Strides, TensorDType,
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
#[derive(Clone, Debug)]
pub struct Tensor {
    inner: Arc<Inner>,
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
    storage: Arc<RwLock<Storage>>,
}

impl AsRef<Inner> for Inner {
    fn as_ref(&self) -> &Inner {
        self
    }
}

impl Inner {
    fn new(op: LazyOp, storage: Storage) -> Self {
        Self {
            id: TensorId::new(),
            op,
            storage: Arc::new(RwLock::new(storage)),
        }
    }
}

impl Tensor {
    pub fn add(&self, other: &Tensor) -> Tensor {
        //Enforce valid shape, dtype, device
        //Determine output shape
        //Binary::shape_inference(self.shape(), other.shape());
        let inner = Inner::new(LazyOp::Binary(Binary::new(
            self.clone(),
            other.clone(),
            BinaryOp::Add,
        )));
        Tensor {
            inner: Arc::new(inner),
        }
    }

    /// Creates a data from a vector.
    ///
    /// If a non-CPU device is specified, the data will be copied to the device.
    pub fn from_vec<T: TensorDType>(data: Vec<T>, shape: Shape, device: Device) -> Tensor {
        let mut inner = Inner::new(LazyOp::Const, device.clone());
        let dt = T::dt();
        let strides = Strides::from(&shape);

        todo!()
    }

    fn execution_order(&self) -> Vec<Tensor> {
        let mut stack = vec![self.clone()];
        let mut visited = vec![];
        while let Some(tensor) = stack.pop() {
            if visited.contains(&tensor) {
                continue;
            }
            match &tensor.inner.op {
                LazyOp::Empty => {}
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
        println!("Order: {:?}", self.execution_order());
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
    use super::*;

    #[test]
    fn test_cfg() {
        let a = Tensor::empty();
        let b = Tensor::empty();
        let c = a.add(&b);
        c.resolve();
    }
}
