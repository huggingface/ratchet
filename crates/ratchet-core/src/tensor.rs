use crate::gpu::{BindGroupEntry, CpuUniform, WgpuDevice};
use crate::{
    ops::*, rvec, CPUBuffer, CompiledOp, DType, Device, DeviceStorage, Executable, GPUBuffer,
    MetaOperation, Operation, OperationError, RVec, RawCPUBuffer, Shape, Storage, Strides,
    TensorDType, TensorId,
};
use crate::{BinaryOp, LazyOp};
use derive_new::new;
use parking_lot::{RwLock, RwLockReadGuard};
use std::ops::Bound;
use std::sync::Arc;

#[cfg(feature = "rand")]
use {rand::prelude::*, rand_distr::StandardNormal};

#[cfg(feature = "pyo3")]
#[cfg(not(target_arch = "wasm32"))]
use {
    ndarray::{ArrayD, ArrayViewD, Dimension},
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

/// Tensors are just an view into their underlying byte storage.
#[derive(new, Debug, Clone)]
pub struct StorageView {
    shape: Shape,
    dt: DType,
    strides: Strides,
}

impl StorageView {
    pub fn is_contiguous(&self) -> bool {
        todo!()
    }
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

    pub fn strides(&self) -> &Strides {
        &self.view.strides
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

macro_rules! impl_binary_op {
    ($method_name:ident, $op:expr) => {
        pub fn $method_name(&self, other: &Tensor) -> anyhow::Result<Tensor> {
            Binary::check_invariants(&[self, other])?;

            let binary = Binary::new(self.clone(), other.clone(), $op);
            let new_view = binary.infer_output(&[self, other])?;
            Ok(Tensor::lazy(
                LazyOp::Binary(binary),
                new_view,
                self.device.clone(),
            ))
        }
    };
}

macro_rules! impl_unary_op {
    ($method_name:ident, $op:expr) => {
        pub fn $method_name(&self) -> anyhow::Result<Tensor> {
            Unary::check_invariants(&[self])?;

            let unary = Unary::new(self.clone(), $op);
            let new_view = unary.infer_output(&[self])?;
            Ok(Tensor::lazy(
                LazyOp::Unary(unary),
                new_view,
                self.device.clone(),
            ))
        }
    };
}

impl Tensor {
    impl_binary_op!(add, BinaryOp::Add);
    impl_binary_op!(sub, BinaryOp::Sub);
    impl_binary_op!(mul, BinaryOp::Mul);
    impl_binary_op!(div, BinaryOp::Div);

    impl_unary_op!(gelu, UnaryOp::Gelu);
    impl_unary_op!(tanh, UnaryOp::Tanh);
    impl_unary_op!(exp, UnaryOp::Exp);
    impl_unary_op!(log, UnaryOp::Log);
    impl_unary_op!(sin, UnaryOp::Sin);
    impl_unary_op!(cos, UnaryOp::Cos);
    impl_unary_op!(abs, UnaryOp::Abs);
    impl_unary_op!(sqrt, UnaryOp::Sqrt);
    impl_unary_op!(relu, UnaryOp::Relu);
    impl_unary_op!(floor, UnaryOp::Floor);
    impl_unary_op!(ceil, UnaryOp::Ceil);

    /// # Slice
    ///
    /// Current slice implementation requires specification of all dimensions.
    /// Currently very user hostile, but will be improved.
    pub fn slice<D: std::ops::RangeBounds<usize>>(&self, ranges: &[D]) -> anyhow::Result<Tensor> {
        let mut resolved_ranges = rvec![];

        for (ridx, r) in ranges.iter().enumerate() {
            let start = match r.start_bound() {
                Bound::Included(&s) => s,
                Bound::Excluded(&s) => s + 1,
                Bound::Unbounded => 0,
            };
            let end = match r.end_bound() {
                Bound::Included(&e) => e + 1,
                Bound::Excluded(&e) => e,
                Bound::Unbounded => self.shape()[ridx],
            };
            resolved_ranges.push(start..end);
        }

        let slice = Slice::new(resolved_ranges);
        let out_view = slice.infer_output(&[self])?;

        let lazy_op = LazyOp::Reindex(Reindex::new(self.clone(), ReindexOp::Slice(slice)));
        Ok(Tensor::lazy(lazy_op, out_view, self.device.clone()))
    }

    pub fn permute(&self, dims: &[usize]) -> anyhow::Result<Tensor> {
        let permute = Permute::new(dims.to_vec());
        let out_view = permute.infer_output(&[self])?;

        let lazy_op = LazyOp::Reindex(Reindex::new(self.clone(), ReindexOp::Permute(permute)));
        Ok(Tensor::lazy(lazy_op, out_view, self.device.clone()))
    }

    //TODO: switch dim to isize and allow negative indexing
    pub fn softmax(&self, dim: usize) -> anyhow::Result<Tensor> {
        Softmax::check_invariants(&[self])?;

        let softmax = Softmax::new(self.clone(), dim);
        let new_view = softmax.infer_output(&[self])?;
        Ok(Tensor::lazy(
            LazyOp::Softmax(softmax),
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

    pub(crate) unsafe fn from_quantized<T: TensorDType, U: AsRef<[T]>>(
        data: U,
        shape: Shape,
        dt: DType,
        device: Device,
    ) -> Tensor {
        let storage = unsafe { Storage::from_quantized(data.as_ref(), &device) };
        let strides = Strides::from(&shape);
        let meta = StorageView::new(shape, dt, strides);
        Tensor::new(LazyOp::Const, meta, Some(storage), device)
    }

    /// # Bindings
    ///
    /// Only applicable to GPU tensors.
    /// Generates the bind group entries required to bind the tensor to a kernel.
    /// Quantized tensors may use multiple bind groups.
    /// Unquantized tensors should only use a single bind group.
    pub(crate) fn bindings(&self) -> RVec<BindGroupEntry> {
        assert!(self.device().is_gpu());
        let storage_guard = self.storage();
        let storage = storage_guard.as_ref().unwrap();
        let gpu_buf = storage.try_gpu().unwrap();
        let handle = gpu_buf.inner().handle;
        let segments = self.dt().segments(gpu_buf.inner().size() as usize);
        segments.iter().fold(rvec![], |mut entries, segment| {
            let (offset, size) = (segment.offset, segment.size);
            entries.push(BindGroupEntry {
                handle,
                offset,
                size,
            });
            entries
        })
    }

    /// Converts the tensor into a 1D vector.
    ///
    /// The 1D vector contains the data from the tensor, as it was laid out in memory.
    pub fn to_vec<T: TensorDType>(&self) -> anyhow::Result<Vec<T>> {
        assert!(self.device().is_cpu());
        let storage_guard = self.storage();
        let buffer = storage_guard.as_ref().unwrap().try_cpu()?;
        let slice = buffer.to_slice::<T>(self.shape());
        Ok(slice.to_vec())
    }

    pub(crate) fn execution_order(&self) -> Vec<Tensor> {
        let mut stack = vec![self.clone()];
        let mut visited = vec![];
        while let Some(tensor) = stack.pop() {
            if visited.contains(&tensor) {
                continue;
            }
            let srcs = match &tensor.inner.op {
                LazyOp::Const => rvec![],
                LazyOp::Binary(b) => b.srcs(),
                LazyOp::Matmul(m) => m.srcs(),
                LazyOp::Softmax(s) => s.srcs(),
                LazyOp::Unary(u) => u.srcs(),
                LazyOp::Reindex(r) => r.srcs(),
                _ => unimplemented!(),
            };
            stack.extend(srcs.into_iter().cloned());
            visited.push(tensor);
        }
        visited.reverse();
        visited
    }

    pub fn compile(
        &self,
        uniform: &mut CpuUniform,
        device: &WgpuDevice,
        can_inplace: bool,
    ) -> Option<CompiledOp> {
        match self.op() {
            LazyOp::Binary(b) => b.compile(self, uniform, device, can_inplace).ok(),
            LazyOp::Matmul(m) => m.compile(self, uniform, device, can_inplace).ok(),
            LazyOp::Softmax(s) => s.compile(self, uniform, device, can_inplace).ok(),
            LazyOp::Unary(u) => u.compile(self, uniform, device, can_inplace).ok(),
            LazyOp::Reindex(r) => r.compile(self, uniform, device, can_inplace).ok(),
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

        for (tix, t) in execution_order.iter().enumerate() {
            if !t.resolved() {
                let id = t.id();
                let pooled_buffer = allocations.get(&id).ok_or(TensorError::NoStorage(id))?;
                assert!(t.device().is_gpu());
                let storage = GPUBuffer {
                    inner: pooled_buffer.clone(),
                    alignment: t.dt().size_of(),
                };
                t.update_storage(Storage::GPU(storage));
            }

            //Can inplace && only 1 consumer
            let can_inplace = t.op().supports_inplace()
                && execution_order[tix + 1..]
                    .iter()
                    .filter(|t2| t2.op.srcs().contains(&t))
                    .count()
                    <= 1;

            if let Some(compiled_op) = t.compile(&mut uniform, device, can_inplace) {
                compiled_ops.push(compiled_op);
            }
        }
        let executable = Executable::new(compiled_ops, uniform.into_gpu(device)?);
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
    pub fn to(&self, device: &Device) -> Result<Tensor, TensorError> {
        match (self.device(), device) {
            (Device::GPU(_), Device::CPU) => self.to_cpu(),
            (Device::CPU, Device::GPU(_)) => self.to_gpu(device),
            _ => Ok(self.clone()),
        }
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

impl Tensor {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn all_close(&self, other: &Self, atol: f32, rtol: f32) -> anyhow::Result<()> {
        if self.shape() != other.shape() {
            anyhow::bail!("Shape mismatch {:?} != {:?}", self.shape(), other.shape())
        }

        let self_nd = self.to_ndarray_view::<f32>();
        let other_nd = other.to_ndarray_view::<f32>();

        let mut stats = CloseStats::new(atol, rtol);
        ndarray::indices_of(&self_nd).into_iter().for_each(|idx| {
            let (a, b) = (self_nd[&idx], other_nd[&idx]);
            stats.update(&a, &b, idx);
        });

        let idx_fmt = stats.max_abs_error_idxs.as_ref().map(|idx| idx.slice());
        if stats.fail_count > 0 {
            anyhow::bail!(
                "\x1b[1;31m{} samples not close \x1b[0m - AVGE={} MAE={} at {:?}",
                stats.fail_count,
                stats.avg_error(),
                stats.max_abs_error,
                idx_fmt
            );
        } else {
            println!(
                "\x1b[1;32mAll close \x1b[0m - AVGE={} MAE={} at {:?}",
                stats.avg_error(),
                stats.max_abs_error,
                idx_fmt
            );
            Ok(())
        }
    }
}

#[derive(Default)]
struct CloseStats {
    total_error: f32,
    max_abs_error: f32,
    #[cfg(not(target_arch = "wasm32"))]
    max_abs_error_idxs: Option<ndarray::IxDyn>,
    element_count: usize,
    fail_count: usize,
    atol: f32,
    rtol: f32,
}

impl CloseStats {
    fn new(atol: f32, rtol: f32) -> Self {
        Self {
            atol,
            rtol,
            ..Default::default()
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn update(&mut self, a: &f32, b: &f32, index: ndarray::IxDyn) {
        let abs_diff = (a - b).abs();
        self.total_error += abs_diff;
        self.element_count += 1;

        if abs_diff > self.max_abs_error {
            self.max_abs_error = abs_diff;
            self.max_abs_error_idxs = Some(index);
        }

        if !self.is_close(a, b, abs_diff) {
            self.fail_count += 1;
        }
    }

    fn avg_error(&self) -> f32 {
        self.total_error / self.element_count as f32
    }

    fn is_close(&self, a: &f32, b: &f32, abs_diff: f32) -> bool {
        (a.is_nan() && b.is_nan())
            || (a.is_infinite() && b.is_infinite() && a.signum() == b.signum())
            || abs_diff <= self.atol + self.rtol * b.abs()
    }
}

/// Conversion to and from numpy arrays
impl Tensor {
    #[cfg(feature = "pyo3")]
    #[cfg(not(target_arch = "wasm32"))]
    pub fn into_ndarray<T: TensorDType>(self) -> ArrayD<T> {
        assert!(self.device().is_cpu());
        let shape = self.shape().to_vec();
        if self.num_bytes() != 0 {
            let storage_guard = self.storage();
            let buffer = storage_guard.as_ref().unwrap().try_cpu().unwrap();
            let (ptr, _) = buffer.inner().into_raw_parts();
            unsafe { ArrayViewD::from_shape_ptr(shape, ptr as *const T).to_owned() }
        } else {
            ArrayViewD::from_shape(shape, &[]).unwrap().to_owned()
        }
    }

    #[cfg(feature = "pyo3")]
    #[cfg(not(target_arch = "wasm32"))]
    pub fn to_ndarray_view<T: TensorDType>(&self) -> ArrayViewD<T> {
        assert!(self.device().is_cpu());
        let shape = self.shape().to_vec();
        if self.num_bytes() != 0 {
            let storage_guard = self.storage();
            let buffer = storage_guard.as_ref().unwrap().try_cpu().unwrap();
            let (ptr, _) = buffer.inner().into_raw_parts();
            unsafe { ArrayViewD::from_shape_ptr(shape, ptr as *const T) }
        } else {
            ArrayViewD::from_shape(shape, &[]).unwrap()
        }
    }

    #[cfg(feature = "pyo3")]
    #[cfg(not(target_arch = "wasm32"))]
    pub fn to_py<'s, 'p: 's, T: TensorDType + numpy::Element>(
        &'s self,
        py: &'p pyo3::Python<'p>,
    ) -> &PyArrayDyn<T> {
        use numpy::PyArray;
        assert!(
            self.device().is_cpu(),
            "Cannot convert non-CPU tensor to numpy array"
        );
        PyArray::from_owned_array(*py, self.deep_clone().into_ndarray::<T>())
    }
}

#[cfg(feature = "pyo3")]
#[cfg(not(target_arch = "wasm32"))]
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

            let raw_buf = RawCPUBuffer::new(ptr, layout);
            let meta = StorageView::new(shape, T::dt(), strides);
            Tensor::new(
                LazyOp::Const,
                meta,
                Some(Storage::CPU(CPUBuffer::new(raw_buf))),
                Device::CPU,
            )
        } else {
            panic!("Cannot convert numpy array with non-contiguous memory layout to tensor");
        }
    }
}

#[cfg(feature = "pyo3")]
#[cfg(not(target_arch = "wasm32"))]
impl<T: TensorDType + numpy::Element> From<&PyArrayDyn<T>> for Tensor {
    fn from(array: &PyArrayDyn<T>) -> Self {
        Self::from(array.to_owned_array())
    }
}

#[cfg(test)]
mod tests {
    use crate::{prelude::*, DeviceRequest};

    #[derive(Debug, derive_new::new)]
    struct AttentionTest {
        input: Tensor,
        qw: Tensor,
        kw: Tensor,
        vw: Tensor,
    }

    fn sdpa_cfg(case: AttentionTest, device: Device) -> anyhow::Result<Tensor> {
        let q_proj = case.input.matmul(&case.qw)?;
        let k_proj = case.input.matmul(&case.kw)?;
        let v_proj = case.input.matmul(&case.vw)?;

        let d_k = (q_proj.shape()[2] as f32).sqrt();
        let kt = k_proj.permute(&[0, 2, 1])?;

        let logits = q_proj.matmul(&kt)?;
        let logits = logits.div(&Tensor::from_data(&[d_k], shape![1], device))?;
        let logits = logits.softmax(2)?;

        let out = logits.matmul(&v_proj)?;
        out.resolve()?;
        Ok(out)
    }

    #[test]
    pub fn test_sdpa() -> anyhow::Result<()> {
        let device = Device::request_device(DeviceRequest::GPU)?;
        let input = Tensor::randn::<f32>(shape![1, 128, 384], device.clone());
        let qw = Tensor::randn::<f32>(shape![384, 384], device.clone());
        let kw = Tensor::randn::<f32>(shape![384, 384], device.clone());
        let vw = Tensor::randn::<f32>(shape![384, 384], device.clone());

        let test_case = AttentionTest::new(input, qw, kw, vw);

        let out = sdpa_cfg(test_case, device.clone())?;
        let out_cpu = out.to(&Device::CPU)?;
        println!("{:?}", out_cpu);
        Ok(())
    }
}
