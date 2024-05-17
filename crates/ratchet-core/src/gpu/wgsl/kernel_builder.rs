/// A builder for generating a kernel in WGSL.

#[derive(Debug)]
pub struct WgslFragment(String);

impl WgslFragment {
    pub fn new(capacity: usize) -> Self {
        Self(String::with_capacity(capacity))
    }

    pub fn write(&mut self, s: &str) {
        self.0.push_str(s);
    }

    pub fn write_fragment(&mut self, fragment: WgslFragment) {
        self.0.push_str(&fragment.0);
    }
}

pub trait RenderFragment {
    fn render(&self) -> WgslFragment;
}

pub struct WgslKernel(String);

pub struct WgslKernelBuilder {
    pub indent: usize,
    pub kernel: String,
}

impl WgslKernelBuilder {
    pub fn new() -> Self {
        Self {
            indent: 0,
            kernel: String::new(),
        }
    }

    pub fn indent(&mut self) {
        self.indent += 1;
    }

    pub fn dedent(&mut self) {
        self.indent -= 1;
    }

    pub fn write_fragment(&mut self, fragment: WgslFragment) {
        self.kernel.push_str("\t".repeat(self.indent).as_str());
        self.kernel.push_str(&fragment.0);
    }
}
