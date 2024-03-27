use crate::{Generate, KernelElement, KernelRenderer};
use std::{fs::File, io::Write};
use tera::Context;

pub struct ConcatOp;

impl Generate for ConcatOp {
    fn generate(renderer: &mut KernelRenderer) -> anyhow::Result<()> {
        let path = renderer.templates_path.join("concat.wgsl");
        renderer.tera.add_template_file(path, Some("concat"))?;

        for i in 2..=8 {
            let mut context = Context::new();
            context.insert("num_inputs", &i);
            let rendered = renderer.tera.render("concat", &context)?;

            let kernel_fname = format!("{}{}_{}.wgsl", "concat", i, KernelElement::Scalar);
            let mut file = File::create(renderer.dest_path.join(kernel_fname))?;
            file.write_all(rendered.as_bytes())?;
        }
        Ok(())
    }
}
