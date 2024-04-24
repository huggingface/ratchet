use std::{fs::File, io::Write};

use strum::IntoEnumIterator;
use tera::Context;

use crate::{Generate, KernelElement, KernelRenderer, WgslDType};

#[derive(Debug, Clone, strum_macros::EnumIter)]
pub enum NormOp {
    LayerNorm,
    RMSNorm,
}

impl std::fmt::Display for NormOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            NormOp::LayerNorm => "layernorm",
            NormOp::RMSNorm => "rmsnorm",
        };
        write!(f, "{}", s)
    }
}

impl Generate for NormOp {
    fn generate(renderer: &mut KernelRenderer) -> anyhow::Result<()> {
        for op in NormOp::iter() {
            for ke in KernelElement::iter() {
                let path = renderer.templates_path.join("norm.wgsl");
                renderer.tera.add_template_file(path, Some("norm"))?;

                let mut context = Context::new();
                context.insert("elem", &ke.as_wgsl(WgslDType::F32));
                context.insert("elem_size", &ke.as_size());
                context.insert("RMS_NORM", &matches!(op, NormOp::RMSNorm));
                let reduction_len = match ke {
                    KernelElement::Scalar => "metadata.N",
                    KernelElement::Vec2 => "metadata.ND2",
                    KernelElement::Vec4 => "metadata.ND4",
                };
                context.insert("reduction_len", reduction_len);
                let rendered = renderer.tera.render("norm", &context)?;

                let kernel_fname = format!("{}_{}.wgsl", op, ke);
                let mut file = File::create(renderer.dest_path.join(kernel_fname))?;
                file.write_all(rendered.as_bytes())?;
            }
        }
        Ok(())
    }
}
