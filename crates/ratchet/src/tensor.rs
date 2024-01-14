use crate::{ops::*, Device, DeviceError, Operation};
use crate::{BinaryOp, LazyOp};
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

// For now, a Tensor is simply used to construct the CFG.
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
    device: Device,
}

impl AsRef<Inner> for Inner {
    fn as_ref(&self) -> &Inner {
        self
    }
}

impl std::default::Default for Inner {
    fn default() -> Self {
        Inner {
            id: TensorId::new(),
            op: LazyOp::Empty,
            device: Device::default(),
        }
    }
}

impl Tensor {
    pub fn add(&self, other: &Tensor) -> Tensor {
        self.check_device(other).unwrap();
        Tensor {
            inner: Arc::new(Inner {
                op: LazyOp::Binary(Binary::new(self.clone(), other.clone(), BinaryOp::Add)),
                device: self.device.clone(),
                ..Default::default()
            }),
        }
    }

    pub fn empty() -> Tensor {
        Tensor {
            inner: Arc::new(Inner::default()),
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

    pub fn check_device(&self, o: &Self) -> Result<(), DeviceError> {
        if self.device == o.device {
            Ok(())
        } else {
            Err(DeviceError::DeviceMismatch(
                self.device.label(),
                o.device.label(),
            ))
        }
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
