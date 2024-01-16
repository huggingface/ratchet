use encase::DynamicUniformBuffer;

///We use a single uniform buffer for all operations to hold their parameters.
///Every operation writes its metadata into this buffer, and an offset is returned.
///This offset is used when binding the buffer.
pub struct CpuUniform(DynamicUniformBuffer<Vec<u8>>);

///Uniforms must be 256-byte aligned, encase handles this for us.
pub const UNIFORM_ALIGN: usize = 256;

impl Default for CpuUniform {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuUniform {
    pub fn new() -> Self {
        Self(DynamicUniformBuffer::new(Vec::new()))
    }

    pub fn into_inner(self) -> Vec<u8> {
        self.0.into_inner()
    }
}

impl std::ops::Deref for CpuUniform {
    type Target = DynamicUniformBuffer<Vec<u8>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for CpuUniform {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
