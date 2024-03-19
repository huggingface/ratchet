//Coarse is like the other kernel for wq8, but you go against the grain.
@group(0) @binding(0)
var<storage, read> X: array<u32>;

@group(0) @binding(1)
var<storage, read> A: array<f32>; 

@group(0) @binding(2)
var<storage, read> I: array<i32>;

@group(0) @binding(3)
var<storage, read_write> Y: array<f32>;

struct Meta {
    dst_numel: u32,
    right_numel: u32,
    ids_numel: u32,
    src_dim_numel: u32,
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
    let tid = (group_id.x * 64u + local_index);
    let right_numel = metadata.right_numel;
    let src_dim_numel = metadata.src_dim_numel;

    if (tid >= metadata.dst_numel) {
        return;
    }

    let id_i = (tid / right_numel) % metadata.ids_numel;
    let input_i = u32(I[id_i]);
    let right_rank_i = tid % right_numel;
    let left_rank_i = tid / (right_numel * metadata.ids_numel);

    let src_i = (left_rank_i * src_dim_numel * right_numel + input_i * right_numel + right_rank_i) / 4u;
    Y[tid] = (unpack4x8snorm(X[src_i]) * A[src_i / 4u])[input_i % 4u];
}
