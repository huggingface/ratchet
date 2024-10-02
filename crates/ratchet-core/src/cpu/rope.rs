use crate::{
    cpu::{cpu_store_result, gemm::gemm},
    shape, DType, OperationError, RoPE, Shape, StridedIterator, Strides, Tensor,
};
use anyhow::anyhow;

pub fn cpu_rope(op: RoPE, dst: Tensor) -> Result<Tensor, OperationError> {
    match op.input().dt() {
        DType::F32 => {
            let dim = op.dim();
            let base = op.base();
            let offset = op.offset();
            let src = op.input().to_vec::<f32>()?;
            let result = rope(src, op.input().shape(), dim, base, offset);
            cpu_store_result(&dst, &result)
        }
        _ => todo!(),
    }

    Ok(dst)
}

fn calculate_sincos(dim: usize, seq_len: usize, base: f32, offset: usize) -> (Vec<f32>, Vec<f32>) {
    let half_dim = dim / 2;

    let positions = (offset..seq_len + offset)
        .map(|x| x as f32)
        .collect::<Vec<f32>>();

    let log_base = base.ln();
    let inv_freqs = (0..half_dim)
        .map(|i| -(i as f32))
        .map(|i| i * log_base / half_dim as f32)
        .map(f32::exp)
        .collect::<Vec<f32>>();

    let p_shape = shape!(half_dim, 1);
    let p_strides = Strides::from(&p_shape);
    let i_shape = shape!(1, half_dim);
    let i_strides = Strides::from(&i_shape);
    let dst_strides = Strides::from(&shape!(half_dim, half_dim));
    let theta = gemm(
        &positions,
        &p_shape,
        &p_strides,
        &inv_freqs,
        &i_shape,
        &i_strides,
        &dst_strides,
        1,
        half_dim,
        half_dim,
        1,
    )
    .unwrap();

    println!("THETA: {:?}", theta);

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

fn slice(src: &[f32], start: &[usize], stop: &[usize]) -> Vec<f32> {
    let stop_numel: usize = stop.iter().product();
    let start_numel: usize = stop.iter().product();
    assert!(stop_numel >= start_numel);

    let mut dst = vec![0.0; stop_numel - start_numel];

    /*
    start: [0, 0, 0, 8]
    stop: [1, 1, 1, 16]
    for
    */

    let mut src_idx = 0;
    let mut dst_idx = 0;
    for i in 0..start.len() {
        let mut src_stride = start[i];
        let mut dst_stride = 0;
        while src_stride < stop[i] {
            dst[dst_idx] = src[src_idx];
            src_idx += src_stride;
            dst_idx += dst_stride;
            src_stride += 1;
            dst_stride += 1;
        }
    }

    dst
}

// Generic transpose function
fn transpose(
    src: Vec<f32>,
    shape: &Shape,
    dim1: usize,
    dim2: usize,
) -> Result<Vec<f32>, OperationError> {
    let rank = shape.rank();
    if dim1 == dim2 {
        return Ok(src);
    }
    if rank <= dim1 || rank <= dim2 {
        return Err(anyhow!("Invalid dimensions for transpose operation").into());
    }
    let mut dims = shape.to_vec();
    let mut strides = Strides::from(shape).to_vec();
    println!("dims: {:?}", dims);
    println!("strides: {:?}", strides);
    dims.swap(dim1, dim2);
    strides.swap(dim1, dim2);
    println!("dims: {:?}", dims);
    println!("strides: {:?}", strides);

    let shape_t = Shape::from(dims);
    let strides_t = Strides::from(strides);

    let mut result = vec![0.0; src.len()];
    let strided_iter = StridedIterator::new(&shape_t, &strides_t, 0);
    let strided_iter2 = StridedIterator::new(&shape_t, &strides_t, 0);
    let indices = strided_iter2.collect::<Vec<_>>();
    println!("indices: {:?}", indices);
    for (index, dst_index) in strided_iter.enumerate() {
        result[dst_index] = src[index];
    }

    Ok(result)
}

fn rope(src: Vec<f32>, shape: &Shape, dim: usize, base: f32, offset: usize) -> Vec<f32> {
    println!("Ratchet RoPE");
    let [batches, num_heads, seq_len, head_dim] = shape.try_into().unwrap();
    let el_count = batches * num_heads * seq_len * head_dim;

    let half_dim = dim / 2;
    let (sin, cos) = calculate_sincos(dim, seq_len, base, offset);

    let mut intermediate = Vec::with_capacity(el_count);

    let chunk_offset = half_dim;
    let skip = 0;

    let (x1, x2) = chunk_by_offset(&src, chunk_offset, skip);

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

    x1_cos.iter().zip(x2_sin).for_each(|(x1_cos, x2_sin)| {
        intermediate.push(x1_cos - x2_sin);
    });

    x1_sin.iter().zip(x2_cos).for_each(|(x1_sin, x2_cos)| {
        intermediate.push(x1_sin + x2_cos);
    });

    let out_shape = shape!(batches, num_heads, seq_len, head_dim);

    let skip = head_dim.abs_diff(dim);
    let mut dst = merge(&intermediate, half_dim, skip);

    if dim < head_dim {
        let offset = (el_count / head_dim) * dim;
        let appendix = &mut src[offset..].to_vec();
        dst.append(appendix);
    }
    dst
}

fn rope_2(src: Vec<f32>, shape: &Shape, dim: usize, base: f32, offset: usize) -> Vec<f32> {
    println!("Ratchet RoPE");
    let [batches, num_heads, seq_len, head_dim] = shape.try_into().unwrap();
    let el_count = batches * num_heads * seq_len * head_dim;

    let half_dim = dim / 2;
    let (sin, cos) = calculate_sincos(dim, seq_len, base, offset);

    println!("cos: {:?}", cos);
    println!("sin: {:?}", sin);

    let src = transpose(src, &shape, 1, 2).unwrap();
    let mut dst = vec![0.0; el_count];
    let b = batches;
    let t = num_heads;
    let h = seq_len;
    let d = head_dim;
    src.chunks(t * h * d)
        .zip(dst.chunks_mut(t * h * d))
        .for_each(|(src, dst)| {
            for i_t in 0..t {
                for i_d in 0..d / 2 {
                    let i_cs = i_t * (d / 2) + i_d;
                    for i_h in 0..h {
                        let i1 = i_t * h * d + i_h * d + i_d;
                        let i2 = i1 + d / 2;
                        dst[i1] = src[i1] * cos[i_cs] - src[i2] * sin[i_cs];
                        dst[i2] = src[i1] * sin[i_cs] + src[i2] * cos[i_cs];
                    }
                }
            }
        });

    let dst = transpose(dst, &shape, 1, 2).unwrap();

    dst
}
