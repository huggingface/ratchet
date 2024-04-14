#![allow(non_snake_case)]
use crate::{Generate, KernelElement, KernelRenderer, WgslDType};
use std::{fs::File, io::Write};
use tera::Context;

pub struct Gemm;

impl Gemm {
    const ROW_PER_THREAD: u32 = 4;
    const TILE_DIM: u32 = 32;

    pub fn generate_vectorized(renderer: &mut KernelRenderer) -> anyhow::Result<()> {
        let FIT_A_OUTER = [false, true];
        let FIT_B_OUTER = [false, true];
        let FIT_INNER = [false, true];
        let QUANT = [false, true];
        let BIAS = [false, true];
        let ke = KernelElement::Vec4;

        let path = renderer.templates_path.join("gemm_vectorized.wgsl");
        renderer.tera.add_template_file(path, Some("gemm"))?;
        for bias in BIAS.iter() {
            for quant in QUANT.iter() {
                for a_fit in FIT_A_OUTER.iter() {
                    for b_fit in FIT_B_OUTER.iter() {
                        for inner_fit in FIT_INNER.iter() {
                            let mut context = Context::new();
                            context.insert("BIAS", &bias);
                            context.insert("QUANT", &quant);
                            context.insert("FIT_A_OUTER", &a_fit);
                            context.insert("FIT_B_OUTER", &b_fit);
                            context.insert("FIT_INNER", &inner_fit);
                            context.insert("TILE_DIM", &Self::TILE_DIM);
                            context.insert("ROW_PER_THREAD", &Self::ROW_PER_THREAD);
                            context.insert("ELEM_TYPE", &ke.as_wgsl(WgslDType::F32));
                            context.insert("ELEM_SIZE", &ke.as_size());

                            let rendered = renderer.tera.render("gemm", &context)?;

                            let kernel_stem = if *quant { "qgemm" } else { "sgemm" };

                            let kernel_fname = format!(
                                "{}_{}_{}_{}_{}_{}.wgsl",
                                kernel_stem, bias, a_fit, b_fit, inner_fit, ke
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

    pub fn generate_scalar(renderer: &mut KernelRenderer) -> anyhow::Result<()> {
        let FIT_A_OUTER = [false, true];
        let FIT_B_OUTER = [false, true];
        let FIT_INNER = [false, true];
        let TRANS_A = [false, true];
        let TRANS_B = [false, true];
        let TRANS_OUT = [false, true];
        let BIAS = [false, true];
        let ke = KernelElement::Scalar;

        let path = renderer.templates_path.join("gemm_scalar.wgsl");
        renderer.tera.add_template_file(path, Some("gemm"))?;
        for bias in BIAS.iter() {
            for trans_a in TRANS_A.iter() {
                for trans_b in TRANS_B.iter() {
                    for trans_out in TRANS_OUT.iter() {
                        for a_fit in FIT_A_OUTER.iter() {
                            for b_fit in FIT_B_OUTER.iter() {
                                for inner_fit in FIT_INNER.iter() {
                                    let mut context = Context::new();
                                    context.insert("BIAS", &bias);
                                    context.insert("TRANS_A", &trans_a);
                                    context.insert("TRANS_B", &trans_b);
                                    context.insert("TRANS_OUT", &trans_out);
                                    context.insert("FIT_A_OUTER", &a_fit);
                                    context.insert("FIT_B_OUTER", &b_fit);
                                    context.insert("FIT_INNER", &inner_fit);
                                    context.insert("TILE_DIM", &Self::TILE_DIM);
                                    context.insert("ROW_PER_THREAD", &Self::ROW_PER_THREAD);
                                    context.insert("ELEM_TYPE", &ke.as_wgsl(WgslDType::F32));
                                    context.insert("ELEM_SIZE", &ke.as_size());

                                    let rendered = renderer.tera.render("gemm", &context)?;

                                    let kernel_fname = format!(
                                        "sgemm_{}_{}_{}_{}_{}_{}_{}_{}.wgsl",
                                        bias,
                                        a_fit,
                                        b_fit,
                                        inner_fit,
                                        trans_a,
                                        trans_b,
                                        trans_out,
                                        ke
                                    );
                                    let mut file =
                                        File::create(renderer.dest_path.join(kernel_fname))?;
                                    file.write_all(rendered.as_bytes())?;
                                }
                            }
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
