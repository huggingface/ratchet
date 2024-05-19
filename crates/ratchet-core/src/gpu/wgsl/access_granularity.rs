use crate::{wgs, BuiltIn, WgslKernel, WgslKernelBuilder};

use super::dtype::WgslDType;

/// WGSL types which are used to access buffers.
pub trait Accessor<T: WgslDType, const W: usize> {
    fn render() -> String;
}

#[derive(Default)]
pub struct Vec4<T> {
    _phantom: std::marker::PhantomData<T>,
}

#[derive(Default)]
pub struct Vec3<T> {
    _phantom: std::marker::PhantomData<T>,
}

#[derive(Default)]
pub struct Vec2<T> {
    _phantom: std::marker::PhantomData<T>,
}

#[derive(Default)]
pub struct Scalar<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: WgslDType> Accessor<T, 4> for Vec4<T> {
    fn render() -> String {
        format!("vec4<{}>", T::render_dt())
    }
}

impl<T: WgslDType> Accessor<T, 3> for Vec3<T> {
    fn render() -> String {
        format!("vec3<{}>", T::render_dt())
    }
}

impl<T: WgslDType> Accessor<T, 2> for Vec2<T> {
    fn render() -> String {
        format!("vec2<{}>", T::render_dt())
    }
}

impl<T: WgslDType> Accessor<T, 1> for Scalar<T> {
    fn render() -> String {
        format!("{}", T::render_dt())
    }
}

pub fn render_softmax<A: Accessor<T, N>, T: WgslDType, const N: usize>() -> WgslKernel {
    let mut kernel_builder = WgslKernelBuilder::new();
    let accessor = A::render();
    let reduce_len = match N {
        1 => "metadata.N",
        2 => "metadata.ND2",
        4 => "metadata.ND4",
        _ => panic!("Invalid dimension"),
    };

    kernel_builder.write_main(
        wgs![128, 1, 1],
        &[
            BuiltIn::GlobalInvocationId,
            BuiltIn::LocalInvocationId,
            BuiltIn::WorkgroupId,
        ],
    );

    let indexing = format!(
        r#"
    let batch_stride = workgroup_id.y * metadata.M * {reduce_len}; 
    let row_start = batch_stride + workgroup_id.x * {reduce_len}; 
    let index = local_invocation_id.x;
    "#
    );

    kernel_builder.write_fragment(indexing.into());

    let mut reduce_max = format!(
        r#"
smem[index] = {accessor}(minFloat);
for (var i: u32 = index; i < {reduce_len}; i += BLOCK_SIZE) {{
    smem[index] = max(smem[index], X0[row_start + i]); 
}}
workgroupBarrier();
"#
    );

    for i in (0..=6).rev().map(|x| 2u32.pow(x)) {
        reduce_max.push_str(&format!(
            r#"
block_max(index, {i}u);"#,
        ));
    }

    reduce_max.push_str(
        r#"
if index == 0u {{
    "#,
    );

    match N {
        1 => reduce_max.push_str("maximum = smem[0];"),
        2 => reduce_max.push_str("maximum = max(smem[0].x, smem[0].y);"),
        4 => reduce_max
            .push_str("maximum = max(smem[0].x, max(smem[0].y, max(smem[0].z, smem[0].w)));"),
        _ => panic!("Invalid dimension"),
    }

    reduce_max.push_str(
        r#"
}}
workgroupBarrier();
"#,
    );

    kernel_builder.write_fragment(reduce_max.into());

    let mut reduce_sum = format!(
        r#"
smem[index] = {accessor}(0.0);
for (var i: u32 = index; i < {reduce_len}; i += BLOCK_SIZE) {{
    smem[index] += exp(X[row_start + i] - maximum);
}}
workgroupBarrier();
"#
    );

    //Need to iterate from 64, 32, 16,
    for i in (0..=6).rev().map(|x| 2u32.pow(x)) {
        reduce_sum.push_str(&format!(
            r#"
block_sum(index, {i}u);"#,
        ));
    }

    reduce_sum.push_str(&format!(
        r#"
    if index == 0u {{
        sum = dot(smem[0], {accessor}(1.0)); 
    }}
    workgroupBarrier();
"#,
    ));

    kernel_builder.indent();
    kernel_builder.write_fragment(reduce_sum.into());
    kernel_builder.dedent();

    let softmax = format!(
        r#"
    for(var i: u32 = index; i < {reduce_len}; i += BLOCK_SIZE) {{
        var val = X[row_start + i];
        X[row_start + i] = exp(val - maximum) / sum;
    }}
}}
"#,
    );

    kernel_builder.write_fragment(softmax.into());
    kernel_builder.render()
}

#[cfg(test)]
mod tests {
    use half::f16;

    use crate::{Vec2, Vec4};

    #[test]
    fn test_render_softmax() {
        let v4 = super::render_softmax::<Vec4<f32>, _, 4>();
        println!("{}", v4);
        let v2 = super::render_softmax::<Vec2<f16>, _, 2>();
        println!("{}", v2);
    }
}
