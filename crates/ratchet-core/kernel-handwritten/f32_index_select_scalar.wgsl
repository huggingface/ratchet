@group(0) @binding(0)
var<storage, read> X: array<f32>;

@group(0) @binding(1)
var<storage, read> I: array<i32>;

@group(0) @binding(2)
var<storage, read_write> Y: array<f32>;

//"right_numel" is the amount of elements we need to move per id provided
//"left" gives us the offset required to move to the start of the next "right block"
//left & right fucking suck in multi dim.
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
    let tid = group_id.x * 64u + local_index;
    if (tid >= metadata.dst_numel) {
        return;
    }
    let id_i = (tid / metadata.right_numel) % metadata.ids_numel;
    let input_i = min(u32(I[id_i]), metadata.src_dim_numel - 1u);
    let right_rank_index = tid % metadata.right_numel;
    let left_rank_index = tid / (metadata.right_numel * metadata.ids_numel);

    let left_offset = left_rank_index * metadata.src_dim_numel * metadata.right_numel;
    let right_offset = input_i * metadata.right_numel + right_rank_index;
    Y[tid] = X[left_offset + right_offset];
}
