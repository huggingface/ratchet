{% for i in range(end=num_inputs) -%}
@group(0) @binding({{i}})
var<storage, read> X{{i}}: array<f32>;
{% endfor -%}

@group(0) @binding({{num_inputs}})
var<storage, read_write> Y: array<f32>;

struct Meta {
    {% for i in range(end=num_inputs) -%}
        x{{i}}_stride: vec4<u32>,
    {% endfor %}
    dst_stride: vec4<u32>,
    dst_numel: u32,
    {% for i in range(end=num_inputs) -%}
        cum{{i}}: u32,
    {% endfor -%}
    dim: u32,
}

@group(1) @binding(0)
var<uniform> metadata: Meta;

//Converts 1D offset into 4D index
fn offsetToNdIndex(offset: u32, stride: vec4<u32>) -> vec4<u32> {
    var index: vec4<u32> = vec4<u32>(0u, 0u, 0u, 0u);
    var remaining = offset;

    for (var i: i32 = 0; i < 3; i++) {
        let idx = remaining / stride[i];
        index[i] = idx;
        remaining -= idx * stride[i];
    }
    index.w = remaining;
    return index;
}

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
    //Dispatch 1 thread per output element
    //dst_offset is index into the output buffer (1D)
    let x_offset = group_id.x * 64u;
    let dst_offset = (group_id.y * num_groups.x * 64u) + x_offset + local_index;
    if (dst_offset >= metadata.dst_numel) {
        return;
    }
    //Convert 1D offset into 4D index
    var dst_index = offsetToNdIndex(dst_offset, metadata.dst_stride);

    let dim = metadata.dim;
    {% for i in range(end=num_inputs) -%}
        if (dst_index[dim] < metadata.cum{{i}}) {
            {% if i > 0 %}
                dst_index[dim] -= metadata.cum{{i-1}};
            {% endif -%}
            let src_offset = ndIndexToOffset(dst_index, metadata.x{{i}}_stride);
            Y[dst_offset] = X{{i}}[src_offset];
            return;
        }
    {% endfor -%}
}
