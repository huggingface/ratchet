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

    let positions = (offset..seq_len + offset)
        .map(|x| x as f32)
        .collect::<Vec<f32>>();
    let log_base = base.log2();
    let inv_freqs = (0..dim)
        .step_by(2)
        .rev()
        .map(|i| -(i as f32))
        .map(|i| i * log_base / half_dim as f32)
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

    let (sin_theta, cos_theta) = theta.iter().map(|i| i.sin_cos()).unzip();

    (sin_theta, cos_theta)
}

fn rope(src: &[f32], shape: &Shape, dim: usize, base: f32, offset: usize) -> Vec<f32> {
    let [b, t, h, d] = shape.try_into().unwrap();
    let el_count = b * h * t * d;

    let (sin, cos) = calculate_sincos(dim, el_count, base, offset);
    //let sin = &sin[offset..el_count + offset];
    //let cos = &cos[offset..el_count + offset];

    let mut dst = vec![0.0; el_count];

    println!("cos len: {}", cos.len());
    println!("sin len: {}", sin.len());
    println!("src len: {}", src.len());
    println!("dst len: {}", dst.len());

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
    dst
}
