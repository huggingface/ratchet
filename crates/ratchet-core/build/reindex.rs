use std::{fs::File, io::Write};

use strum::IntoEnumIterator;
use tera::Context;

use crate::{Generate, KernelElement, KernelRenderer, WgslDType};

#[derive(Debug, Clone, strum_macros::EnumIter)]
pub enum ReindexOp {
    Permute,
    Slice,
    Broadcast,
}

impl std::fmt::Display for ReindexOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ReindexOp::Permute => "permute",
            ReindexOp::Slice => "slice",
            ReindexOp::Broadcast => "broadcast",
        };
        write!(f, "{}", s)
    }
}

impl ReindexOp {
    pub fn func_body(&self) -> String {
        match self {
            ReindexOp::Permute => r#"
    var src_index = vec4<u32>(0u);
    src_index[metadata.perm[0]] = dst_index[0]; 
    src_index[metadata.perm[1]] = dst_index[1];
    src_index[metadata.perm[2]] = dst_index[2];
    src_index[metadata.perm[3]] = dst_index[3];"#
                .to_string(),
            ReindexOp::Slice => r#"
    var src_index = dst_index;"#
                .to_string(),
            ReindexOp::Broadcast => format!(
                r#"
    // Broadcasting is valid if dims are equal, or if one of the dims is 1
    var src_index = select(dst_index, vec4<u32>(0u), metadata.src_shape == vec4<u32>(1u));
    "#
            ),
        }
    }
}

impl Generate for ReindexOp {
    fn generate(renderer: &mut KernelRenderer) -> anyhow::Result<()> {
        let path = renderer.templates_path.join("reindex.wgsl");
        renderer.tera.add_template_file(path, Some("reindex"))?;

        for op in ReindexOp::iter() {
            for ke in KernelElement::iter() {
                if matches!(ke, KernelElement::Vec4 | KernelElement::Vec2)
                    && !matches!(op, ReindexOp::Broadcast)
                {
                    continue;
                }

                let mut context = Context::new();
                context.insert("elem", &ke.as_wgsl(WgslDType::F32));
                context.insert("elem_size", &ke.as_size());
                context.insert("func_body", &op.func_body());

                let rendered = renderer.tera.render("reindex", &context)?;

                let kernel_fname = format!("{}_{}.wgsl", op, ke);
                let mut file = File::create(renderer.dest_path.join(kernel_fname))?;
                file.write_all(rendered.as_bytes())?;
            }
        }
        Ok(())
    }
}
