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
        let QUANT = [false, true];
        let mut ke = KernelElement::Scalar;

        let path = renderer.templates_path.join("gemv.wgsl");
        renderer.tera.add_template_file(path, Some("gemv"))?;
        for quant in QUANT.iter() {
            if *quant {
                ke = KernelElement::Vec4;
            }

            for wgx in WORKGROUP_X.iter() {
                for wgy in WORKGROUP_Y.iter() {
                    for fit in FIT.iter() {
                        for bias in BIAS.iter() {
                            let mut context = Context::new();
                            context.insert("ELEM_TYPE", &ke.as_wgsl(WgslDType::F32));
                            context.insert("ELEM_SIZE", &ke.as_size());
                            context.insert("QUANT", &quant);
                            context.insert("FIT", &fit);
                            context.insert("BIAS", &bias);
                            context.insert("workgroup_size_x", &wgx);
                            context.insert("workgroup_size_y", &wgy);
                            context.insert("workgroup_size_z", &1);

                            let kernel_stem = if *quant { "qgemv" } else { "sgemv" };

                            let rendered = renderer.tera.render("gemv", &context)?;

                            let kernel_fname = format!(
                                "{}_{}_{}_{}_{}_{}.wgsl",
                                kernel_stem, bias, wgx, wgy, fit, ke
                            );
                            let mut file = File::create(renderer.dest_path.join(kernel_fname))?;
                            file.write_all(rendered.as_bytes())?;
                        }
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
