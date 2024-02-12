@group(0) @binding(0)
var<storage, read> X: array<f32>;

@group(0) @binding(1)
var<storage, read> I: array<i32>;

@group(0) @binding(2)
var<storage, read_write> Y: array<f32>;

struct Meta {
    dim_len: i32
}

@group(1) @binding(0)
var<uniform> metadata: Meta;

@compute @workgroup_size(8,8,1)
fn main( 
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) group_id: vec3<u32>,
        @builtin(num_workgroups) num_groups: vec3<u32>
) {
    let src_offset = u32(I[group_id.y] * metadata.dim_len);
    let dst_offset = group_id.y * u32(metadata.dim_len);

    let thread_offset = group_id.x * 64u + local_index;
    if (thread_offset >= u32(metadata.dim_len)) {
        return;
    }

    Y[dst_offset + thread_offset] = X[src_offset + thread_offset];
}
