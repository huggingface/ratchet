use std::{fs::File, io::Write};

use strum::IntoEnumIterator;
use tera::Context;

use crate::{Generate, KernelElement, KernelRenderer, WgslDType};

#[derive(Debug, Clone, strum_macros::EnumIter)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinaryOp {
    pub fn mapping(&self) -> (&'static str, &'static str) {
        match self {
            BinaryOp::Add => ("add", "+"),
            BinaryOp::Sub => ("sub", "-"),
            BinaryOp::Mul => ("mul", "*"),
            BinaryOp::Div => ("div", "/"),
        }
    }
}

impl Generate for BinaryOp {
    fn generate(renderer: &mut KernelRenderer) -> anyhow::Result<()> {
        let pairs = BinaryOp::iter().fold(Vec::new(), |mut acc, op| {
            acc.push(op.mapping());
            acc
        });
        let path = renderer.templates_path.join("binary.wgsl");
        renderer.tera.add_template_file(path, Some("binary"))?;
        for (op_name, op) in &pairs {
            for ke in KernelElement::iter() {
                let mut context = Context::new();
                context.insert("op", op);
                context.insert("elem", &ke.as_wgsl(WgslDType::F32));
                context.insert("elem_size", &ke.as_size());
                let rendered = renderer.tera.render("binary", &context)?;

                let kernel_fname = format!("{}_{}.wgsl", op_name, ke);
                let mut file = File::create(renderer.dest_path.join(kernel_fname))?;
                file.write_all(rendered.as_bytes())?;
            }
        }
        Ok(())
    }
}
