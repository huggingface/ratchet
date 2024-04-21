//https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
@group(0) @binding(0)
var<storage, read_write> X: array<f32>;

struct Meta {
    M: u32,
    N: u32,
    ND2: u32,
    ND4: u32,
}

@group(1) @binding(0)
var<uniform> metadata: Meta;

var<workgroup> smem: array<f32, 128>; // max size is 16kb
var<workgroup> maximum: f32;
var<workgroup> sum: f32;

const BLOCK_SIZE = 128u;
const minFloat: f32 = -3.402823e+38f;

fn block_sum(index: u32, stride: u32) {
    if index < stride {
        smem[index] += smem[index + stride];
    }
    workgroupBarrier();
}

fn block_max(index: u32, stride: u32) {
    if index < stride {
        smem[index] = max(smem[index], smem[index + stride]);
    }
    workgroupBarrier();
}

@compute @workgroup_size(128, 1, 1)
fn main( 
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let batch_stride = group_id.y * metadata.M * metadata.N;
    let row_start = batch_stride + group_id.x * metadata.N; 
    let index = local_id.x;

    smem[index] = minFloat;
    for (var i: u32 = index; i < metadata.N; i += BLOCK_SIZE) {
        smem[index] = max(smem[index], X[row_start + i]); 
    }
    workgroupBarrier();

    block_max(index, 64u);
    block_max(index, 32u);
    block_max(index, 16u);
    block_max(index, 8u);
    block_max(index, 4u);
    block_max(index, 2u);
    block_max(index, 1u);

    if index == 0u{
        maximum = smem[0];
    }
    workgroupBarrier();

    smem[index] = 0.0;
    for (var i: u32 = index; i < metadata.N; i += BLOCK_SIZE) {
        smem[index] += exp(X[row_start + i] - maximum);
    }
    
    workgroupBarrier();
    block_sum(index, 64u);
    block_sum(index, 32u);
    block_sum(index, 16u);
    block_sum(index, 8u);
    block_sum(index, 4u);
    block_sum(index, 2u);
    block_sum(index, 1u);

    if index == 0u {
        sum = smem[0];
    }
    workgroupBarrier();

    for(var i: u32 = index; i < metadata.N; i += BLOCK_SIZE) {
        var val = X[row_start + i];
        X[row_start + i] = exp(val - maximum) / sum;
    }
}
