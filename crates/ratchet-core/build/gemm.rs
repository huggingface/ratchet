#![allow(non_snake_case)]
use crate::{Generate, KernelElement, KernelRenderer, WgslDType};
use std::{fs::File, io::Write};
use tera::Context;

pub struct Gemm;

impl Gemm {
    pub fn generate_vectorized(renderer: &mut KernelRenderer) -> anyhow::Result<()> {
        let FIT_A_OUTER = [false, true];
        let FIT_B_OUTER = [false, true];
        let FIT_INNER = [false, true];
        let QUANTIZED_B = [false, true];
        let ke = KernelElement::Vec4;

        let path = renderer.templates_path.join("gemm_vectorized.wgsl");
        renderer.tera.add_template_file(path, Some("gemm"))?;
        for quantized in QUANTIZED_B.iter() {
            for a_fit in FIT_A_OUTER.iter() {
                for b_fit in FIT_B_OUTER.iter() {
                    for inner_fit in FIT_INNER.iter() {
                        let mut context = Context::new();
                        context.insert("QUANTIZED_B", &quantized);
                        context.insert("FIT_A_OUTER", &a_fit);
                        context.insert("FIT_B_OUTER", &b_fit);
                        context.insert("FIT_INNER", &inner_fit);
                        context.insert("TILE_DIM", &32);
                        context.insert("ROW_PER_THREAD", &4);
                        context.insert("ELEM_TYPE", &ke.as_wgsl(WgslDType::F32));
                        context.insert("ELEM_SIZE", &ke.as_size());

                        let rendered = renderer.tera.render("gemm", &context)?;

                        let kernel_stem = if *quantized { "qgemm" } else { "sgemm" };

                        let kernel_fname = format!(
                            "{}_{}_{}_{}_{}.wgsl",
                            kernel_stem, a_fit, b_fit, inner_fit, ke
                        );
                        let mut file = File::create(renderer.dest_path.join(kernel_fname))?;
                        file.write_all(rendered.as_bytes())?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn generate_scalar(renderer: &mut KernelRenderer) -> anyhow::Result<()> {
        let FIT_A_OUTER = [false, true];
        let FIT_B_OUTER = [false, true];
        let FIT_INNER = [false, true];
        let TRANS_A = [false, true];
        let TRANS_B = [false, true];
        let ke = KernelElement::Scalar;

        let path = renderer.templates_path.join("gemm_scalar.wgsl");
        renderer.tera.add_template_file(path, Some("gemm"))?;
        for trans_a in TRANS_A.iter() {
            for trans_b in TRANS_B.iter() {
                for a_fit in FIT_A_OUTER.iter() {
                    for b_fit in FIT_B_OUTER.iter() {
                        for inner_fit in FIT_INNER.iter() {
                            let mut context = Context::new();
                            context.insert("TRANS_A", &trans_a);
                            context.insert("TRANS_B", &trans_b);
                            context.insert("FIT_A_OUTER", &a_fit);
                            context.insert("FIT_B_OUTER", &b_fit);
                            context.insert("FIT_INNER", &inner_fit);
                            context.insert("TILE_DIM", &32);
                            context.insert("ROW_PER_THREAD", &4);
                            context.insert("ELEM_TYPE", &ke.as_wgsl(WgslDType::F32));
                            context.insert("ELEM_SIZE", &ke.as_size());

                            let rendered = renderer.tera.render("gemm", &context)?;

                            let kernel_fname = format!(
                                "sgemm_{}_{}_{}_{}_{}_{}.wgsl",
                                a_fit, b_fit, inner_fit, trans_a, trans_b, ke
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

impl Generate for Gemm {
    fn generate(renderer: &mut KernelRenderer) -> anyhow::Result<()> {
        Self::generate_vectorized(renderer)?;
        Self::generate_scalar(renderer)?;
        Ok(())
    }
}
