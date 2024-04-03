//Translated from: https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/rope.metal
//Reading materials
//https://blog.eleuther.ai/rotary-embeddings/
//
// RoPE summary:
// 1. In language, we don't care about the absolute position of words(tokens). I don't care that cat is in position 500.
//    However, I do care about the relative position of words. "You shall know a word by the company it keeps (Firth, 1957)"
// 2. RoPE gives us a way to encode relative positions into our hidden states for q & k.
// 3. Key insight of rope: Use rotations to encode relative positions, unaffected by translation (absolute position).
// 4. Rotations are easiest to work with in complex space. 
// 5. Complex numbers suck for computers, so we do it in the reals.
// 6. We pair up components of our q & k vectors, to form 2D coords in the complex plain.
//    This can be done in 2 ways: 1. q = (q1, q2, q3, q4) -> q = (q1 + iq2, q3 + iq4)
//                                2. q = (q1, q2 ... qd/2, qd/2+1) -> q = (q1 + iqd/2, q2 + iqd/2+1)
// 7. We then rotate these 2D coords by a fixed angle, theta, to encode relative positions.
        
@group(0) @binding(0)
var<storage, read_write> in: array<f32>;

struct Meta {
    in_strides: vec3<u32>,
    out_strides: vec3<u32>,
    seq_len: u32,
    hd: u32,
    offset: u32,
    base: f32,
    scale: f32,
}

@group(1) @binding(0)
var<uniform> metadata: Meta;

@compute @workgroup_size(16, 2, 16)
fn main( 
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(global_invocation_id) pos: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>,
) {
  if(pos.y >= metadata.seq_len) {
    return;
  }

  let grid = vec3<u32>(groups.x * 16u, groups.y * 2u, groups.z * 16u);

  let out_index_1 = dot(pos, vec3<u32>(metadata.out_strides[2], metadata.out_strides[1], metadata.out_strides[0]));
  let out_index_2 = out_index_1 + grid.x * metadata.out_strides[2];

  let in_index_1 = dot(pos, vec3<u32>(metadata.in_strides[2], metadata.in_strides[1], metadata.in_strides[0]));
  let in_index_2 = in_index_1 + grid.x * metadata.in_strides[2];

  let L = metadata.scale * f32(pos.y + metadata.offset);
  let d = f32(pos.x) / f32(grid.x);

  let theta = L * exp2(-d * metadata.base);
  let costheta = cos(theta);
  let sintheta = sin(theta);

  let x1 = in[in_index_1];
  let x2 = in[in_index_2];

  let rx1 = x1 * costheta - x2 * sintheta;
  let rx2 = x1 * sintheta + x2 * costheta;

  in[out_index_1] = rx1;
  in[out_index_2] = rx2;
}
