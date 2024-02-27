@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<u32>;

@group(0) @binding(2)
var<storage, read> absmax: array<f32>;

@group(0) @binding(3)
var<storage, read_write> C: array<vec4<f32>>;

struct Meta {
    M: u32,
    N: u32,
    K: u32,
    MD2: u32,
    ND2: u32,
    KD2: u32,
    MD4: u32,
    ND4: u32,
    KD4: u32,
    A_OFFSET: u32,
    B_OFFSET: u32,
    C_OFFSET: u32,
}

@group(1) @binding(0)
var<uniform> metadata: Meta;

@compute @workgroup_size(8,8,1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let a_offset = global_id.z * metadata.A_OFFSET; 
    let b_offset = global_id.z * metadata.B_OFFSET; 
    let c_offset = global_id.z * metadata.C_OFFSET; 

    let cRow = global_id.x;
    let cCol = global_id.y;  
    
    let b_stride = metadata.K; //Solve 4 per iter 
    let abs_stride = metadata.K / 16u;

    if (cRow < metadata.M && cCol < metadata.ND4) {
        var tmp = vec4<f32>(0.0);
        for (var k = 0u; k < metadata.KD4; k++) {
          let a = A[a_offset + cRow * metadata.KD4 + k];

          let bidx = b_offset + (cCol * b_stride) + k;
          let absidx = (cCol * abs_stride * 4u) + (k / 4u);

          let b0 = unpack4x8snorm(B[bidx]) * absmax[absidx];
          let b1 = unpack4x8snorm(B[bidx + metadata.KD4]) * absmax[absidx + abs_stride];
          let b2 = unpack4x8snorm(B[bidx + (2u * metadata.KD4)]) * absmax[absidx + (2u * abs_stride)];
          let b3 = unpack4x8snorm(B[bidx + (3u * metadata.KD4)]) * absmax[absidx + (3u * abs_stride)];
        
          tmp = fma(vec4<f32>(a.x), vec4<f32>(b0.x, b1.x, b2.x, b3.x), tmp);
          tmp = fma(vec4<f32>(a.y), vec4<f32>(b0.y, b1.y, b2.y, b3.y), tmp);
          tmp = fma(vec4<f32>(a.z), vec4<f32>(b0.z, b1.z, b2.z, b3.z), tmp);
          tmp = fma(vec4<f32>(a.w), vec4<f32>(b0.w, b1.w, b2.w, b3.w), tmp);
        }
        C[c_offset + (cRow * metadata.ND4 + cCol)] = tmp;
    }
}

