#![cfg(feature = "trace")]
use crate::{DType, Device, Tensor};

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
    pub fn serialize(&self, device: &Device) -> Result<(), anyhow::Error> {
        log::warn!("Serializing trace to disk");
        let device_identifier = device.device_identifier();

        let trace_dir = format!(
            "trace-{}-{}",
            uuid::Uuid::new_v4().to_string(),
            device_identifier
        );
        std::fs::create_dir(&trace_dir).map_err(|e| anyhow::anyhow!(e))?;

        let mut metadata = Vec::new();
        for t in self.iter() {
            let name = format!("ratchet-{}.npy", t.id());
            let path = format!("{}/{}", trace_dir, name);
            let _ = match t.dt() {
                DType::F32 => t.write_npy::<f32, _>(&path),
                _ => unimplemented!(),
            };
            metadata.push(name);
        }

        let metadata_path = format!("{}/metadata.json", trace_dir);
        std::fs::write(metadata_path, serde_json::to_string(&metadata)?)?;
        Ok(())
    }

    pub fn compare(&self, other: &Self, atol: f32, rtol: f32) -> Result<(), anyhow::Error> {
        assert_eq!(self.0.len(), other.0.len());
        log::warn!("Comparing traces");
        for (a, b) in self.iter().zip(other.iter()) {
            log::warn!("A: {:?}", a);
            log::warn!("B: {:?}", b);
            a.all_close(b, atol, rtol)?;
        }
        Ok(())
    }

    pub fn first(&self) -> Option<&Tensor> {
        self.0.first()
    }

    pub fn pop(&mut self) -> Option<Tensor> {
        self.0.pop()
    }
}

impl Iterator for Trace {
    type Item = Tensor;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop()
    }
}

#[cfg(target_arch = "wasm32")]
impl Trace {
    pub fn deserialize(dir: include_dir::Dir) -> Result<Self, anyhow::Error> {
        let metadata_file = dir
            .get_file("metadata.json")
            .ok_or_else(|| anyhow::anyhow!("Metadata file not found"))?;
        let metadata: Vec<String> = serde_json::from_slice(metadata_file.contents())?;

        let mut tensors = Vec::with_capacity(metadata.len());

        for filename in metadata {
            let file = dir
                .get_file(&filename)
                .ok_or_else(|| anyhow::anyhow!("File not found: {}", filename))?;
            let tensor = Tensor::from_npy_bytes::<f32>(file.contents(), &Device::CPU)?;
            tensors.push(tensor);
        }

        Ok(Self::new(tensors))
    }
}
