#![cfg(feature = "trace")]
use crate::{DType, Tensor};

/// # Trace
///
/// All intermediate products & result of a computation.
pub struct Trace(Vec<Tensor>);

impl Trace {
    pub fn new(t: Vec<Tensor>) -> Self {
        Self(t)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Tensor> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Tensor> {
        self.0.iter_mut()
    }

    /// # Serialization
    ///
    /// We may want to serialize a trace to disk to determine platform discrepancies.
    ///
    /// This method does the following:
    /// 1. Creates a trace directory with a UUID, time, and device details
    /// 2. Serializes each tensor in the trace to disk, with the name being the tensor ID
    pub fn serialize(&self) -> Result<(), anyhow::Error> {
        log::warn!("Serializing trace to disk");
        use half::f16;

        let trace_dir = format!("trace-{}", uuid::Uuid::new_v4().to_string(),);
        std::fs::create_dir(&trace_dir).map_err(|e| anyhow::anyhow!(e))?;
        for t in self.iter() {
            let id = t.id();
            let path = format!("{}/ratchet-{}.npy", trace_dir, id);
            let _ = match t.dt() {
                DType::F16 => t.write_npy::<f16, _>(&path),
                DType::F32 => t.write_npy::<f32, _>(&path),
                _ => unimplemented!(),
            };
        }
        Ok(())
    }
}

impl Iterator for Trace {
    type Item = Tensor;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop()
    }
}
