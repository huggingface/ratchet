@group(0) @binding(0)
var<storage, read> X: array<f32>;

@group(0) @binding(1)
var<storage, read_write> Y: array<vec4<f32>>;

struct Meta {
    src_shape: vec4<u32>,
    dst_shape: vec4<u32>,
    src_stride: vec4<u32>,
    dst_stride: vec4<u32>,
    src_numel: u32,
    dst_numel: u32,
    perm: vec4<u32>,
    src_offsets: vec4<u32>,
}

@group(1) @binding(0)
var<uniform> metadata: Meta;

//Converts 1D offset into 4D index
fn offsetToNdIndex(offset: u32, stride: vec4<u32>) -> vec4<u32> {
    var index: vec4<u32> = vec4<u32>(0u, 0u, 0u, 0u);
    var remaining = offset;

    var idx = 0u;
    idx = remaining / stride[0];
        index[0] = idx;
        remaining -= idx * stride[0];idx = remaining / stride[1];
        index[1] = idx;
        remaining -= idx * stride[1];idx = remaining / stride[2];
        index[2] = idx;
        remaining -= idx * stride[2];
    index.w = remaining;
    return index;
}

var<workgroup> broadcasted: vec4<f32>;

@compute @workgroup_size(8,8,1)
fn main( 
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) group_id: vec3<u32>,
        @builtin(num_workgroups) num_groups: vec3<u32>
) {
    //Dispatch 1 thread per output element
    //dst_offset is index into the output buffer (1D)
    let x_offset = group_id.x * 64u;
    var dst_offset = (group_id.y * num_groups.x * 64u) + x_offset + local_index;
    if (dst_offset >= metadata.dst_numel) {
        return;
    }

    if (local_id.x == 0u) {
        let val = X[0];
        broadcasted = vec4<f32>(val);
    }
    workgroupBarrier();

    //Convert 1D offset into 4D index
    let dst_index = offsetToNdIndex(dst_offset, metadata.dst_stride);
    
    Y[dst_offset] = broadcasted; 
}
