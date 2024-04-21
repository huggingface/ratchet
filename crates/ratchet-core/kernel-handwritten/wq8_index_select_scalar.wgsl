@group(0) @binding(0)
var<storage, read> X: array<u32>;

@group(0) @binding(1)
var<storage, read> A: array<f32>; 

@group(0) @binding(2)
var<storage, read> I: array<i32>;

@group(0) @binding(3)
var<storage, read_write> Y: array<vec4<f32>>;

struct Meta {
    dst_numel: u32,
    right_numel: u32,
    ids_numel: u32,
    src_dim_numel: u32,
}

@group(1) @binding(0)
var<uniform> metadata: Meta;

fn unpack4x8snorm_gguf(x: u32) -> vec4<f32> {
    return unpack4x8snorm(x) * 127f;
}


@compute @workgroup_size(8,8,1)
fn main( 
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) group_id: vec3<u32>,
        @builtin(num_workgroups) num_groups: vec3<u32>
) {
    let tid = (group_id.x * 64u + local_index);
    let right_numel = metadata.right_numel/ 4u;
    let src_dim_numel = metadata.src_dim_numel/ 4u;

    if (tid >= metadata.dst_numel / 4u) {
        return;
    }

    let id_i = (tid / right_numel) % metadata.ids_numel;
    let input_i = min(u32(I[id_i]), (src_dim_numel * 4u) - 1u);
    let right_rank_i = tid % right_numel;
    let left_rank_i = tid / (right_numel * metadata.ids_numel);

    let src_i = left_rank_i * src_dim_numel * right_numel + input_i * right_numel + right_rank_i;
    Y[tid] = unpack4x8snorm_gguf(X[src_i]) * A[src_i / 8u];
}
