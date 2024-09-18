use std::num::NonZero;

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

    let p_shape = shape!(seq_len, 1);
    let p_strides = Strides::from(&p_shape);
    let i_shape = shape!(1, half_dim);
    let i_strides = Strides::from(&i_shape);
    let dst_strides = Strides::from(&shape!(seq_len, half_dim));
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

    let (sin_theta, cos_theta) = theta.iter().map(|i| i.sin_cos()).unzip();

    (sin_theta, cos_theta)
}

#[inline]
fn chunk_by_offset(data: &[f32], offset: usize, skip: usize) -> (Vec<f32>, Vec<f32>) {
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

        start += skip;
        stop += skip;
    }
    (x1.to_vec(), x2.to_vec())
}

#[inline]
fn merge(data: &[f32], offset: usize, skip: usize) -> Vec<f32> {
    let n = data.len();
    let mid = n / 2;
    let mut interleaved = Vec::with_capacity(n);

    let mut start = 0;
    let mut stop = offset;
    while stop + mid <= n {
        let mut chunk = data[start..stop].to_vec();
        interleaved.append(&mut chunk);

        let mut chunk = data[start + mid..stop + mid].to_vec();
        interleaved.append(&mut chunk);

        start += offset;
        stop += offset;

        start += skip;
        stop += skip;
    }
    interleaved
}

fn rope(src: &[f32], shape: &Shape, dim: usize, base: f32, offset: usize) -> Vec<f32> {
    println!("Ratchet RoPE");
    let [batches, num_heads, seq_len, head_dim] = shape.try_into().unwrap();
    let el_count = batches * num_heads * seq_len * head_dim;

    let half_dim = dim / 2;
    let (sin, cos) = calculate_sincos(dim, seq_len, base, offset);
    let mut intermediate = Vec::with_capacity(el_count);

    let chunk_offset = half_dim;
    let skip = 0;

    println!("chunk_offset: {}", chunk_offset);
    let (x1, x2) = chunk_by_offset(src, chunk_offset, skip);

    println!("cos len: {}", cos.len());
    println!("sin len: {}", sin.len());
    println!("src len: {}", src.len());
    println!("x1 len: {}", x1.len());
    println!("x2 len: {}", x2.len());

    let (x1_cos, x1_sin): (Vec<f32>, Vec<f32>) = x1
        .iter()
        .enumerate()
        .map(|(i, x)| (x * cos[i % cos.len()], x * sin[i % sin.len()]))
        .unzip();

    let (x2_cos, x2_sin): (Vec<f32>, Vec<f32>) = x2
        .iter()
        .enumerate()
        .map(|(i, x)| (x * cos[i % cos.len()], x * sin[i % sin.len()]))
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
        intermediate.push(x1_cos - x2_sin);
    });

    x1_sin.iter().zip(x2_cos).for_each(|(x1_sin, x2_cos)| {
        intermediate.push(x1_sin + x2_cos);
    });

    println!("intermediate: {:?}", intermediate);
    println!("intermediate len: {}", intermediate.len());

    let out_shape = shape!(batches, num_heads, seq_len, head_dim);
    println!("out_shape: {:?}", out_shape);

    let skip = head_dim.abs_diff(dim);
    let mut dst = merge(&intermediate, chunk_offset, skip);

    if dim < head_dim {
        dst.append(&mut src[dim..].to_vec())
    }
    println!("dst: {:?}", dst);
    println!("dst len: {}", dst.len());
    dst
}
