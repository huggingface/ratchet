var<private> localId: vec3<u32>;
var<private> globalId: vec3<u32>;
var<private> workgroupId: vec3<u32>;

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(1) @binding(0) var<uniform> metadata: Meta;

struct Meta {
    aShape: vec3<i32>,
    aStrides: vec3<i32>,
    bShape: vec3<i32>,
    bStrides: vec3<i32>,
    outShape: vec3<i32>,
    outShapeStrides: vec3<i32>,
    dimAOuter: i32,
    dimBOuter: i32,
    dimInner: i32,
}

var<workgroup> work: array<f32, {{workgroup_size_x * workgroup_size_y}}>;

@compute @workgroup_size({{workgroup_size_x}},{{workgroup_size_y}},{{workgroup_size_z}})
fn main(@builtin(local_invocation_id) localId : vec3<u32>,
        @builtin(global_invocation_id) globalId : vec3<u32>,
        @builtin(workgroup_id) workgroupId : vec3<u32>) {
    let row = i32(globalId.x);

    {% if not FIT %}
        if (row >= metadata.aShape.y) {
            return;
        }
    {% endif %}

    let batch = i32(globalId.z);
    let batchA = batch % metadata.aShape[0];
    let batchB = batch % metadata.bShape[0];

    let aOffset = metadata.aStrides.x * batchA;
    let bOffset = metadata.bStrides.x * batchB;
    let outOffset = metadata.outShapeStrides.x * batch;

    var sum = 0.0;
    let aIndex = aOffset + row * metadata.aStrides.y;

    for (var k = i32(globalId.y); k < metadata.dimInner; k+={{workgroup_size_y}}) {
        sum = fma(A[aIndex + k], X[bOffset + k], sum);
    }
    
    let rows = {{workgroup_size_x}}u;
    let cols = {{workgroup_size_y}}u;
    let ii = u32(localId.x);
    let jj = u32(localId.y);
    work[ii + rows * jj] = sum;
    workgroupBarrier();

    // Reduce sums in log2(cols) steps
    for (var s = u32(cols) / 2u; s > 0u; s >>= 1u) {
        if (jj < s) {
            work[ii + rows * jj] += work[ii + rows * (jj + s)];
        }
        workgroupBarrier();
    }

    if (jj == 0u) {
        result[outOffset + row] = work[ii];
    }
}
