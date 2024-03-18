fn getAIndexFromCoords3D(coords : vec3<i32>) -> i32 {
    return dot(coords, metadata.aStrides);
}

fn getBIndexFromCoords3D(coords : vec3<i32>) -> i32 {
    return dot(coords, metadata.bStrides);
}

fn getOutputIndexFromCoords(coords : vec3<i32>) -> i32 {
  return dot(coords, metadata.outStrides);
}
        
fn setOutputAtIndex(flatIndex : i32, value : vec4<f32>) {
    result[flatIndex] = vec4<f32>(value);
}

fn setOutputAtCoords(d0 : i32, d1 : i32, d2 : i32, value : vec4<f32>) {
    let flatIndex = getOutputIndexFromCoords(vec3<i32>(d0, d1, d2));
    setOutputAtIndex(flatIndex / 4, value);
}

fn getA(d0 : i32, d1 : i32, d2 : i32) -> vec4<f32> {
    return vec4<f32>(A[getAIndexFromCoords3D(vec3<i32>(d0,d1,d2)) / 4]);
}
   
fn getB(d0 : i32, d1 : i32, d2 : i32) -> vec4<f32> {
    return vec4<f32>(B[getBIndexFromCoords3D(vec3<i32>(d0,d1,d2)) / 4]);
}
   
{% if FIT_A_OUTER and FIT_INNER %}
fn mm_readA(batch: i32, row: i32, col: i32) -> vec4<f32> {
    var value = vec4<f32>(0.0);
    value = getA(batch, row, col);
    return value;
}
{% else %}
fn mm_readA(batch: i32, row: i32, col: i32) -> vec4<f32> {
    var value = vec4<f32>(0.0);
    if (row < metadata.aShape.y && col < metadata.aShape.z) {
        value = getA(batch, row, col);
    }
    return value;
}
{% endif %}

fn mm_readB(batch: i32, row: i32, col: i32) -> vec4<f32> {
    var value = vec4<f32>(0.0);
    value = getB(batch, row, col);
    return value;
}
  
fn mm_write(batch: i32, row: i32, col: i32, valueIn: vec4<f32>) {
{% if FIT_A_OUTER and FIT_B_OUTER %}
        var value = valueIn;
        let coords = vec3<i32>(batch, row, col);
        setOutputAtCoords(coords[0], coords[1], coords[2], value);
{% else %}
    if (row < metadata.dimAOuter && col < metadata.dimBOuter) {
        var value = valueIn;
        let coords = vec3<i32>(batch, row, col);
        setOutputAtCoords(coords[0], coords[1], coords[2], valueIn);
    }
{% endif %}
}

      
var<private> localId: vec3<u32>;
var<private> globalId: vec3<u32>;
var<private> workgroupId: vec3<u32>;

@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1) var<storage, read> B: array<vec4<f32>>;

@group(0) @binding(2) var<storage, read_write> result: array<vec4<f32>>;

struct Meta {
    aShape: vec3<i32>,
    aStrides: vec3<i32>,
    bShape: vec3<i32>,
    bStrides: vec3<i32>,
    outShape: vec3<i32>,
    outStrides: vec3<i32>,
    dimAOuter: i32,
    dimBOuter: i32,
    dimInner: i32,
}

@group(1) @binding(0)
var<uniform> metadata: Meta;

var<workgroup> mm_Asub : array<array<vec4<f32>, {{ TILE_DIM / 4 }}>, {{ TILE_DIM }}>; 
var<workgroup> mm_Bsub : array<array<vec4<f32>, {{ TILE_DIM / 4 }}>, {{ TILE_DIM }}>;
  
@compute @workgroup_size(8,8,1) 
fn main(@builtin(local_invocation_id) localId : vec3<u32>,
        @builtin(global_invocation_id) globalId : vec3<u32>,
        @builtin(workgroup_id) workgroupId : vec3<u32>) {
    let batch = i32(globalId.z);
    let batchA = batch % metadata.aShape[0];
    let batchB = batch % metadata.bShape[0];

    let localRow = i32(localId.y);
    let tileRow = localRow * {{ ROW_PER_THREAD }};
    let tileCol = i32(localId.x);

    let globalRow = i32(globalId.y) * {{ ROW_PER_THREAD }};
    let globalCol = i32(globalId.x) * 4;

    let numTiles = (metadata.dimInner - 1) / {{ TILE_DIM }} + 1;
    var kStart = 0;

    var acc: array<vec4<f32>, {{ ROW_PER_THREAD }}>;

    // Loop over shared dimension.
    let tileRowB = localRow * {{ ROW_PER_THREAD }};
    for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (var innerRow = 0; innerRow < {{ ROW_PER_THREAD }}; innerRow++) {
            let inputRow = tileRow + innerRow;
            let inputCol = tileCol;
            
            mm_Asub[inputRow][inputCol] = mm_readA(batchA, globalRow + innerRow, kStart + inputCol * 4);
        }

        // Load one tile of B into local memory.
        for (var innerRow = 0; innerRow < {{ ROW_PER_THREAD }}; innerRow++) {
            let inputRow = tileRowB + innerRow;
            let inputCol = tileCol;
            mm_Bsub[inputRow][inputCol] = mm_readB(batchB, kStart + inputRow, globalCol);
        }
        kStart = kStart + {{ TILE_DIM }};
        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k = 0; k < {{ TILE_DIM / 4 }}; k++) {
          let bidx = k * 4;
          let BCached0 = mm_Bsub[bidx][tileCol];
          let BCached1 = mm_Bsub[bidx + 1][tileCol];
          let BCached2 = mm_Bsub[bidx + 2][tileCol];
          let BCached3 = mm_Bsub[bidx + 3][tileCol];
          for (var i = 0; i < {{ ROW_PER_THREAD }}; i++) {
            let ACached = mm_Asub[tileRow + i][k];
            acc[i] = fma(BCached0, vec4<f32>(ACached[0]), acc[i]);
            acc[i] = fma(BCached1, vec4<f32>(ACached[1]), acc[i]);
            acc[i] = fma(BCached2, vec4<f32>(ACached[2]), acc[i]);
            acc[i] = fma(BCached3, vec4<f32>(ACached[3]), acc[i]);
          }
        }
        workgroupBarrier();
    }

    {% for innerRow in range(end=ROW_PER_THREAD) %}
        mm_write(batch, globalRow + {{ innerRow }}, globalCol, acc[{{ innerRow }}]);
    {% endfor %}
  }
