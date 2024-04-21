@group(0) @binding(0)
var<storage, read_write> D: array<f32>;

@group(0) @binding(1)
var<storage, read> S: array<f32>;

struct Meta {
    dst_strides: vec4<u32>,
    src_numel: u32,
    write_start: vec4<u32>,
}

@group(1) @binding(0)
var<uniform> metadata: Meta;


//Converts 4D index into 1D offset
fn ndIndexToOffset(index: vec4<u32>, stride: vec4<u32>) -> u32 {
    var offset: u32 = 0u;
    for (var i: i32 = 0; i < 4; i++) {
        offset += index[i] * stride[i];
    }
    return offset;
}

@compute @workgroup_size(8,8,1)
fn main( 
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) group_id: vec3<u32>,
        @builtin(num_workgroups) num_groups: vec3<u32>
) {
    let x_offset = group_id.x * 64u;
    let thread_offset = (group_id.y * num_groups.x * 64u) + x_offset + local_index;
    if (thread_offset >= metadata.src_numel) {
        return;
    }
    let offset_index = ndIndexToOffset(metadata.write_start, metadata.dst_strides);
    D[offset_index + thread_offset] = S[thread_offset];
}
