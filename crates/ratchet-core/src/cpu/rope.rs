use crate::{
    cpu::{cpu_store_result, gemm::gemm},
    shape, DType, OperationError, RoPE, Shape, Strides, Tensor, TensorDType, TensorError, Unary,
};
use half::{bf16, f16};
use num_traits::Float;

pub fn cpu_rope(op: RoPE, dst: Tensor) -> Result<Tensor, OperationError> {
    match op.input().dt() {
        DType::F32 => {
            let dim = op.dim();
            let base = op.base();
            let offset = op.offset();
            let src = op.input().to_vec::<f32>()?;
            let result = rope(&src, op.input().shape(), dim, base, offset);
            cpu_store_result(&dst, &result)
        }
        _ => todo!(),
    }

    Ok(dst)
}

fn calculate_sincos(dim: usize, seq_len: usize, base: f32, offset: usize) -> (Vec<f32>, Vec<f32>) {
    let half_dim = dim / 2;

    let p_len = seq_len + offset;

    let positions = (offset..p_len).map(|x| x as f32).collect::<Vec<f32>>();

    let log_base = base.ln();
    let inv_freqs = (0..half_dim)
        .map(|i| -(i as f32))
        .map(|i| i * log_base / half_dim as f32)
        .map(f32::exp)
        .collect::<Vec<f32>>();

    println!("positions: {:?}", positions);
    println!("inv_freqs: {:?}", inv_freqs);

    let p_shape = shape!(p_len, 1);
    let p_strides = Strides::from(&p_shape);
    let i_shape = shape!(1, half_dim);
    let i_strides = Strides::from(&i_shape);
    let dst_strides = Strides::from(&shape!(p_len, half_dim));
    let theta = gemm(
        &positions,
        &p_shape,
        &p_strides,
        &inv_freqs,
        &i_shape,
        &i_strides,
        &dst_strides,
        1,
        seq_len,
        half_dim,
        1,
    )
    .unwrap();

    println!("theta: {:?}", theta);

    let (sin_theta, cos_theta) = theta.iter().map(|i| i.sin_cos()).unzip();

    (sin_theta, cos_theta)
}

#[inline]
fn split_by_offset(data: &[f32], offset: usize) -> (Vec<f32>, Vec<f32>) {
    let mut x1 = Vec::with_capacity(data.len() / 2);
    let mut x2 = Vec::with_capacity(data.len() / 2);

    let mut start = 0;
    let mut stop = offset;
    while stop < data.len() {
        let mut chunk = data[start..stop].to_vec();
        x1.append(&mut chunk);
        start += offset;
        stop += offset;

        let mut chunk = data[start..stop].to_vec();
        x2.append(&mut chunk);
        start += offset;
        stop += offset;
    }
    (x1.to_vec(), x2.to_vec())
}

fn rope(src: &[f32], shape: &Shape, dim: usize, base: f32, offset: usize) -> Vec<f32> {
    println!("Ratchet RoPE");
    let [b, h, t, d] = shape.try_into().unwrap();
    let el_count = b * h * t * d;

    let (sin, cos) = calculate_sincos(dim, t, base, offset);

    let mut dst = Vec::with_capacity(el_count);

    println!("cos len: {}", cos.len());
    println!("sin len: {}", sin.len());
    println!("src len: {}", src.len());
    println!("dst len: {}", dst.len());

    let split = sin.len();
    let (x1, x2) = split_by_offset(src, split);

    let (x1_cos, x2_cos): (Vec<f32>, Vec<f32>) = cos
        .iter()
        .zip(x1.iter())
        .zip(x2.iter())
        .map(|((c, x1), x2)| (c * x1, c * x2))
        .unzip();

    let (x1_sin, x2_sin): (Vec<f32>, Vec<f32>) = sin
        .iter()
        .zip(x1.iter())
        .zip(x2.iter())
        .map(|((s, x1), x2)| (s * x1, s * x2))
        .unzip();

    println!("x1: {:?}", x1);
    println!("x2: {:?}", x2);
    println!("sin: {:?}", sin);
    println!("cos: {:?}", cos);

    println!("x1_sin: {:?}", x1_sin);
    println!("x1_cos: {:?}", x1_cos);
    println!("x2_sin: {:?}", x2_sin);
    println!("x2_cos: {:?}", x2_cos);

    x1_cos.iter().zip(x2_sin).for_each(|(x1_cos, x2_sin)| {
        dst.push(x1_cos - x2_sin);
    });

    x1_sin.iter().zip(x2_cos).for_each(|(x1_sin, x2_cos)| {
        dst.push(x1_sin + x2_cos);
    });

    println!("dst: {:?}", dst);

    dst
}
