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
    
    let absmax_stride = metadata.N / 16u;
    let b_stride = metadata.N; //Solve 4 per iter a.k.a metadata.ND4 * 4u

    if (cRow < metadata.M && cCol < metadata.ND4) {
        var tmp = vec4<f32>(0.0);
        for (var k = 0u; k < metadata.KD4; k++) {
          let a = A[a_offset + cRow * metadata.KD4 + k];
          
          let bidx = b_offset + (k * b_stride) + cCol;
          let absidx = (k * 4u) * absmax_stride + (cCol / 4u);
          let b0 = unpack4x8snorm(B[bidx]) * absmax[absidx];
          let b1 = unpack4x8snorm(B[bidx + (1u * metadata.ND4)]) * absmax[absidx + absmax_stride];
          let b2 = unpack4x8snorm(B[bidx + (2u * metadata.ND4)]) * absmax[absidx + (2u * absmax_stride)];
          let b3 = unpack4x8snorm(B[bidx + (3u * metadata.ND4)]) * absmax[absidx + (3u * absmax_stride)];

          tmp = fma(vec4<f32>(a.x), b0, tmp);
          tmp = fma(vec4<f32>(a.y), b1, tmp);
          tmp = fma(vec4<f32>(a.z), b2, tmp);
          tmp = fma(vec4<f32>(a.w), b3, tmp);
        }
        C[c_offset + (cRow * metadata.ND4 + cCol)] = tmp;
    }
}

