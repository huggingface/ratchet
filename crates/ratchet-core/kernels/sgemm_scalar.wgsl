//Naive matrix multiplication
//https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/1_naive.cuh
@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> B: array<f32>;

@group(0) @binding(2)
var<storage, read_write> C: array<f32>;

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
    if (cRow < metadata.M && cCol < metadata.N) {
        var tmp = 0f;
        for (var k = 0u; k < metadata.K; k++) {
          tmp = fma(A[a_offset + (cRow * metadata.K + k)], B[b_offset + (k * metadata.N + cCol)], tmp);
        }
        C[c_offset + (cRow * metadata.N + cCol)] = tmp; 
    }
}
