use crate::{
    cpu::cpu_store_result, DType, OperationError, RoPE, Shape, Tensor, TensorDType, TensorError,
    Unary,
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

fn rope(src: &[f32], shape: &Shape, dim: usize, base: f32, offset: usize) -> Vec<f32> {
    let [b, t, h, d] = shape.try_into().unwrap();
    let el_count = b * h * t * d;

    let src = &src[offset..offset + el_count];

    let half_dim = dim / 2;
    let positions = (offset..el_count + offset)
        .map(|x| x as f32)
        .collect::<Vec<f32>>();

    let log_base = base.log2();
    let inv_freqs = (0..d)
        .step_by(2)
        .rev()
        .map(|i| -(i as f32))
        .map(|i| i * log_base / half_dim as f32)
        .map(|i| i.exp())
        .collect::<Vec<f32>>();

    let theta = positions
        .iter()
        .zip(inv_freqs.iter())
        .map(|(p, i)| p * i)
        .collect::<Vec<f32>>();

    let cos = theta.iter().map(|x| x.cos()).collect::<Vec<f32>>();
    let sin = theta.iter().map(|x| x.sin()).collect::<Vec<f32>>();

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

fn old_rope(src: &[f32], shape: &Shape, dim: usize, base: f32, offset: usize) -> Vec<f32> {
    let cos = src.iter().map(|x| x.cos()).collect::<Vec<f32>>();
    let sin = src.iter().map(|x| x.sin()).collect::<Vec<f32>>();

    let b = *shape.get(0).unwrap();
    let t = *shape.get(1).unwrap();
    let h = *shape.get(2).unwrap();
    let d = *shape.get(3).unwrap();

    let el_count = b * h * t * d;
    let mut dst = vec![0.0; el_count];
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
