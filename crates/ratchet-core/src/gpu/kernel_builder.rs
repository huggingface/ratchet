pub struct WgslKernel(String);

pub struct WgslKernelBuilder {
    pub indent: usize,
    pub kernel: String,
}
