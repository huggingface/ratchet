@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> B: array<f32>;

@group(0) @binding(2)
var<storage, read_write> Y: array<f32>;

struct Meta {
    numel: u32,
}

@group(1) @binding(0)
var<uniform> metadata: Meta;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(num_workgroups) num_groups: vec3<u32>
) {
    let x_offset = group_id.x * 64u;
    let index = (group_id.y * num_groups.x * 64u) + x_offset + local_index;
    if (index >= metadata.numel / 1u) {
        return;
    }
    Y[index] = A[index] * B[index];
}
