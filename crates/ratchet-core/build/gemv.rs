#![allow(non_snake_case)]
use crate::{Generate, KernelElement, KernelRenderer, WgslDType};
use std::{fs::File, io::Write};
use tera::Context;

pub struct Gemv;

impl Gemv {
    pub fn generate_kernel(renderer: &mut KernelRenderer) -> anyhow::Result<()> {
        let WORKGROUP_X = [16, 32];
        let WORKGROUP_Y = [4, 8];
        let FIT = [false, true];
        let BIAS = [false, true];
        let ke = KernelElement::Scalar;

        let path = renderer.templates_path.join("sgemv.wgsl");
        renderer.tera.add_template_file(path, Some("sgemv"))?;
        for wgx in WORKGROUP_X.iter() {
            for wgy in WORKGROUP_Y.iter() {
                for fit in FIT.iter() {
                    for bias in BIAS.iter() {
                        let mut context = Context::new();
                        context.insert("ELEM_TYPE", &ke.as_wgsl(WgslDType::F32));
                        context.insert("ELEM_SIZE", &ke.as_size());
                        context.insert("FIT", &fit);
                        context.insert("BIAS", &bias);
                        context.insert("workgroup_size_x", &wgx);
                        context.insert("workgroup_size_y", &wgy);
                        context.insert("workgroup_size_z", &1);

                        let rendered = renderer.tera.render("sgemv", &context)?;

                        let kernel_fname =
                            format!("sgemv_{}_{}_{}_{}_{}.wgsl", bias, wgx, wgy, fit, ke);
                        let mut file = File::create(renderer.dest_path.join(kernel_fname))?;
                        file.write_all(rendered.as_bytes())?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl Generate for Gemv {
    fn generate(renderer: &mut KernelRenderer) -> anyhow::Result<()> {
        Self::generate_kernel(renderer)?;
        Ok(())
    }
}
