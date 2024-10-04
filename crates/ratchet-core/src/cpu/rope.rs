use crate::{
    concat,
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

fn compute_theta(dim: usize, seq_len: usize, base: f32, offset: usize) -> Vec<f32> {
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
    )
    .unwrap();

    theta
}

fn slice(src: &[f32], src_strides: &Strides, start: &[usize], stop: &[usize]) -> Vec<f32> {
    assert!(start.len() == stop.len());
    assert!(start.len() == src_strides.rank());
    start.iter().zip(stop.iter()).for_each(|(s, t)| {
        assert!(s < t);
    });

    let delta: Vec<usize> = stop.iter().zip(start.iter()).map(|(s, t)| s - t).collect();
    let dst_shape: Vec<usize> = delta.clone();
    let dst_numel: usize = delta.iter().product();

    let mut dst = vec![0.0; dst_numel];

    for i in 0..dst_numel {
        let mut src_index = 0;
        let mut tmp = i;
        for d in 0..delta.len() {
            let coord = tmp / dst_shape[d + 1..].iter().product::<usize>().max(1);
            tmp %= dst_shape[d + 1..].iter().product::<usize>().max(1);
            src_index += (coord + start[d]) * src_strides[d] as usize;
        }
        dst[i] = src[src_index];
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

    let half_dim = dim / 2;
    let theta = compute_theta(dim, seq_len, base, offset);
    println!("Theta: {:?}", theta);
    let (sin, cos): (Vec<f32>, Vec<f32>) = theta.iter().map(|i| i.sin_cos()).unzip();
    println!("Cos: {:?}", cos);
    println!("Sin: {:?}", sin);

    println!("Cos length: {:?}", cos.len());
    println!("Sin length: {:?}", sin.len());

    println!("HALF DIM: {:?}", half_dim);
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
    println!("X1: {:?}", x1);
    println!("X1 length: {:?}", x1.len());
    println!("X2: {:?}", x2);
    println!("X2 length: {:?}", x2.len());

    //zip and repeat
    //`multiply` as an operation that deals with broadcasting
    let x1_cos = x1
        .iter()
        .enumerate()
        .map(|(i, x)| x * cos[i % cos.len()])
        .collect::<Vec<f32>>();
    let x2_sin = x2
        .iter()
        .enumerate()
        .map(|(i, x)| x * sin[i % sin.len()])
        .collect::<Vec<f32>>();

    let mut outs = vec![];
    let r1 = x1_cos
        .iter()
        .zip(x2_sin.iter())
        .map(|(x1, x2)| x1 - x2)
        .collect::<Vec<f32>>();
    outs.push(r1.clone());

    let x1_sin = x1
        .iter()
        .enumerate()
        .map(|(i, x)| x * sin[i % sin.len()])
        .collect::<Vec<f32>>();
    let x2_cos = x2
        .iter()
        .enumerate()
        .map(|(i, x)| x * cos[i % cos.len()])
        .collect::<Vec<f32>>();
    let r2 = x1_sin
        .iter()
        .zip(x2_cos.iter())
        .map(|(x1, x2)| x1 + x2)
        .collect::<Vec<f32>>();
    outs.push(r2.clone());

    println!("R1: {:?}", r1);
    println!("R2: {:?}", r2);

    if dim < shape[3] {
        outs.push(slice(
            &src,
            &src_strides,
            &[0, 0, 0, dim],
            &[batches, num_heads, seq_len, head_dim],
        ));
    }

    let (o0, o1, o2) = (outs[0].clone(), outs[1].clone(), outs[2].clone());

    let to_cat = [
        (&shape![num_heads, seq_len, half_dim], o0),
        (&shape![num_heads, seq_len, half_dim], o1),
        (&shape![num_heads, seq_len, head_dim - dim], o2),
    ];

    let dst_shape = shape![num_heads, seq_len, head_dim];
    let mut dst = vec![0.0f32; dst_shape.numel()];
    concat(to_cat.as_slice(), 2, &dst_shape, &mut dst).unwrap();
    println!("CONCAT: {:?}", dst);
    dst
}
