use crate::{
    concat,
    cpu::{cpu_store_result, gemm::gemm, reindex::slice},
    shape, DType, OperationError, RoPE, Shape, Strides, Tensor,
};
use anyhow::anyhow;

pub fn cpu_rope(op: RoPE, dst: Tensor) -> Result<Tensor, OperationError> {
    match op.input().dt() {
        DType::F32 => {
            let dim = op.dim();
            let base = op.base();
            let offset = op.offset();
            let src = op.input().to_vec::<f32>()?;
            let result = rope(src, op.input().shape(), dim, base, offset)?;
            cpu_store_result(&dst, &result)
        }
        _ => todo!(),
    }

    Ok(dst)
}

fn compute_theta(
    dim: usize,
    seq_len: usize,
    base: f32,
    offset: usize,
) -> Result<Vec<f32>, OperationError> {
    let half_dim = dim / 2;

    let positions = (offset..seq_len + offset)
        .map(|x| x as f32)
        .collect::<Vec<f32>>();

    let inv_freqs = (0..half_dim)
        .map(|i| -(i as f32))
        .map(|i| i * base.ln() / half_dim as f32)
        .map(f32::exp)
        .collect::<Vec<f32>>();

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
    )?;

    Ok(theta)
}

fn rope(
    src: Vec<f32>,
    shape: &Shape,
    dim: usize,
    base: f32,
    offset: usize,
) -> Result<Vec<f32>, OperationError> {
    let [batches, num_heads, seq_len, head_dim] = shape.try_into().unwrap();

    let half_dim = dim / 2;
    let theta = compute_theta(dim, seq_len, base, offset)?;
    let (sin, cos): (Vec<f32>, Vec<f32>) = theta.iter().map(|i| i.sin_cos()).unzip();
    let src_strides = Strides::from(shape);
    let x1 = slice(
        &src,
        &src_strides,
        &[0, 0, 0, 0],
        &[batches, num_heads, seq_len, half_dim],
    );
    let x2 = slice(
        &src,
        &src_strides,
        &[0, 0, 0, half_dim],
        &[batches, num_heads, seq_len, dim],
    );

    //`multiply` as an operation that deals with broadcasting
    let x1_cos = x1
        .iter()
        .zip(cos.iter().cycle())
        .map(|(x, c)| x * c)
        .collect::<Vec<f32>>();
    let x2_sin = x2
        .iter()
        .zip(sin.iter().cycle())
        .map(|(x, s)| x * s)
        .collect::<Vec<f32>>();

    let mut r1 = x1_cos
        .iter()
        .zip(x2_sin.iter())
        .map(|(x1, x2)| x1 - x2)
        .collect::<Vec<f32>>();
    r1.extend(vec![0.0; shape.numel() - r1.len()]);

    let x1_sin = x1
        .iter()
        .zip(sin.iter().cycle())
        .map(|(x, s)| x * s)
        .collect::<Vec<f32>>();
    let x2_cos = x2
        .iter()
        .zip(cos.iter().cycle())
        .map(|(x, c)| x * c)
        .collect::<Vec<f32>>();
    let mut r2 = x1_sin
        .iter()
        .zip(x2_cos.iter())
        .map(|(x1, x2)| x1 + x2)
        .collect::<Vec<f32>>();
    r2.extend(vec![0.0; shape.numel() - r2.len()]);

    let mut to_cat = vec![
        (shape![batches, num_heads, seq_len, half_dim], r1),
        (shape![batches, num_heads, seq_len, half_dim], r2),
    ];
    if dim < shape[3] {
        let r3 = slice(
            &src,
            &src_strides,
            &[0, 0, 0, dim],
            &[batches, num_heads, seq_len, head_dim],
        );
        to_cat.push((shape![batches, num_heads, seq_len, head_dim - dim], r3));
    }

    let dst_shape = shape![batches, num_heads, seq_len, head_dim];
    let mut dst = vec![0.0f32; dst_shape.numel()];
    concat(to_cat.as_slice(), 3, &dst_shape, &mut dst)?;
    Ok(dst)
}
