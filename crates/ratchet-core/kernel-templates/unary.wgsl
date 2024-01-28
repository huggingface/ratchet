@group(0) @binding(0)
var<storage, read> X: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read_write> Y: array<vec4<f32>>;

struct Meta {
    numel: u32,
}

@group(1) @binding(0)
var<uniform> metadata: Meta;

const NORM_CONST: vec4<f32> = vec4<f32>(0.5f, 0.5f, 0.5f, 0.5f);
const SQRT_2_OVER_PI: vec4<f32> = vec4<f32>(0.7978845608028654f, 0.7978845608028654f, 0.7978845608028654f, 0.7978845608028654f);
const SCALED_SQRT_2_OVER_PI: vec4<f32> = vec4<f32>(0.035677408136300125f, 0.035677408136300125f, 0.035677408136300125f, 0.035677408136300125f);

fn safe_tanh(x: vec4<f32>) -> vec4<f32> {
    var mut_x = x;
    for (var i: i32 = 0; i < 4; i++) {
        if (abs(mut_x[i]) >= 10.0f) {
            mut_x[i] = sign(mut_x[i]);
        }else {
            mut_x[i] = tanh(mut_x[i]);
        }
    }
    return mut_x;
}

fn fast_gelu_kernel(index: u32) {
  if (index < metadata.numel / 4u) {
    let in = X[index];
    let cdf = NORM_CONST + NORM_CONST * safe_tanh(in * (SCALED_SQRT_2_OVER_PI * (in * in) + SQRT_2_OVER_PI));
    Y[index] = in * cdf;
  }
}

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main( 
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let index = group_id.x * {{ workgroup_size_x * workgroup_size_y }}u + local_index;
    fast_gelu_kernel(index);
}

