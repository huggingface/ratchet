fn getIndexFromCoords3D(coords : vec3<i32>, shape : vec3<i32>) -> i32 {
    return dot(coords, vec3<i32>(shape.y * shape.z, shape.z, 1));
}

fn getOutputIndexFromCoords(coords : vec3<i32>) -> i32 {
  return dot(coords, metadata.outShapeStrides);
}
        
fn setOutputAtIndex(flatIndex : i32, value : {{ ELEM_TYPE }}) {
    result[flatIndex] = {{ ELEM_TYPE }}(value);
}

fn setOutputAtCoords(d0 : i32, d1 : i32, d2 : i32, value : {{ ELEM_TYPE }}) {
    let flatIndex = getOutputIndexFromCoords(vec3<i32>(d0, d1, d2));
    setOutputAtIndex(flatIndex / {{ ELEM_SIZE }}, value);
}

fn getA(d0 : i32, d1 : i32, d2 : i32) -> {{ ELEM_TYPE }} {
    return {{ ELEM_TYPE }}(A[getIndexFromCoords3D(vec3<i32>(d0,d1,d2), metadata.aShape) / {{ ELEM_SIZE }}]);
}
   
fn getB(d0 : i32, d1 : i32, d2 : i32) -> {{ ELEM_TYPE }} {
    return {{ ELEM_TYPE }}(B[getIndexFromCoords3D(vec3<i32>(d0,d1,d2), metadata.bShape) / {{ ELEM_SIZE }}]);
}

{% if QUANTIZED_B %}
fn getAbsMax(d0 : i32, d1 : i32, d2 : i32) -> f32 {
    let abs_index = getIndexFromCoords3D(vec3<i32>(d0,d1,d2), metadata.bShape) / 16;
    return absmax[abs_index]; 
}
{% endif %}
   
fn mm_readA(batch: i32, row: i32, col: i32) -> {{ ELEM_TYPE }} {
    var value = {{ELEM_TYPE}}(0.0);
    {% if A_FIT %}
        value = getA(batch, row, col);
    {% else %}
        if (row < metadata.aShape.y && col < metadata.aShape.z) {
            value = getA(batch, row, col);
        }
    {% endif %}
    return value;
}

fn mm_readB(batch: i32, row: i32, col: i32) -> {{ ELEM_TYPE }} {
    var value = {{ ELEM_TYPE }}(0.0);
    {% if B_FIT %}
        value = getB(batch, row, col);
    {% else %}
        if (row < metadata.bShape.y && col < metadata.bShape.z) {
            value = getB(batch, row, col);
        }
    {% endif %}
    return value;
}
  
fn mm_write(batch: i32, row: i32, col: i32, valueIn: {{ ELEM_TYPE }}) {
    {% if OUT_FIT %}
            var value = valueIn;
            let coords = vec3<i32>(batch, row, col);
            setOutputAtCoords(coords[0], coords[1], coords[2], value);
    {% else %}
        if (row < metadata.outShape.y && col < metadata.outShape.z) {
            var value = valueIn;
            let coords = vec3<i32>(batch, row, col);
            setOutputAtCoords(coords[0], coords[1], coords[2], valueIn);
        }
    {% endif %}
}

      
var<private> localId: vec3<u32>;
var<private> globalId: vec3<u32>;
var<private> workgroupId: vec3<u32>;

struct Meta {
    aShape: vec3<i32>,
    bShape: vec3<i32>,
    outShape: vec3<i32>,
    outShapeStrides: vec3<i32>,
    dimInner: i32,
}

@group(0) @binding(0) var<storage, read> A: array<{{ ELEM_TYPE}}>;

{% if QUANTIZED_B %}
@group(0) @binding(1) var<storage, read> B: array<u32>;

@group(0) @binding(2) var<storage, read> absmax: f32; 

@group(0) @binding(3) var<storage, read_write> result: array<{{ ELEM_TYPE }}>;

{% else %}
@group(0) @binding(1) var<storage, read> B: array<{{ ELEM_TYPE }}>;

@group(0) @binding(2) var<storage, read_write> result: array<{{ ELEM_TYPE }}>;
{% endif %}


@group(1) @binding(0)
var<uniform> metadata: Meta;

var<workgroup> mm_Asub : array<array<{{ ELEM_TYPE }}, {{ TILE_DIM / ELEM_SIZE }}>, {{ TILE_DIM }}>; 
var<workgroup> mm_Bsub : array<array<{{ ELEM_TYPE }}, {{ TILE_DIM / ELEM_SIZE }}>, {{ TILE_DIM }}>;
  
@compute @workgroup_size(8,8,1) 
fn main(@builtin(local_invocation_id) localId : vec3<u32>,
        @builtin(global_invocation_id) globalId : vec3<u32>,
        @builtin(workgroup_id) workgroupId : vec3<u32>) {
    let localRow = i32(localId.y);
    let tileRow = localRow * {{ ROW_PER_THREAD }};
    let tileCol = i32(localId.x);

    let globalRow = i32(globalId.y) * {{ ROW_PER_THREAD }};
    let globalCol = i32(globalId.x) * {{ ROW_PER_THREAD }}; //colperthread
    let batch = i32(globalId.z);
    let batchA = batch % metadata.aShape.x; 
    let batchB = batch % metadata.bShape.x;

    let numTiles = (metadata.dimInner - 1) / {{ TILE_DIM }} + 1;
    var kStart = 0;

    var acc: array<{{ ELEM_TYPE }}, {{ ROW_PER_THREAD }}>;

    // Loop over shared dimension.
    let tileRowB = localRow * {{ ROW_PER_THREAD }};
    for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (var innerRow = 0; innerRow < {{ ROW_PER_THREAD }}; innerRow++) {
            let inputRow = tileRow + innerRow;
            let inputCol = tileCol;
            
            mm_Asub[inputRow][inputCol] = mm_readA(batchA, globalRow + innerRow, kStart + inputCol * {{ ELEM_SIZE }});
        }

        // Load one tile of B into local memory.
        for (var innerRow = 0; innerRow < {{ ROW_PER_THREAD }}; innerRow++) {
            let inputRow = tileRowB + innerRow;
            let inputCol = tileCol;
            {% if QUANTIZED_B %}
                let absmax = getAbsMax(batchB, kStart + inputRow, globalCol);
                mm_Bsub[inputRow][inputCol] = mm_readB(batchB, kStart + inputRow, globalCol) * absmax;
            {% else %}
                mm_Bsub[inputRow][inputCol] = mm_readB(batchB, kStart + inputRow, globalCol);
            {% endif %}
        }
        kStart = kStart + {{ TILE_DIM }};
        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k = 0; k < {{ TILE_DIM / ELEM_SIZE }}; k++) {
          let bidx = k * {{ ELEM_SIZE }};
          {% for elem in range(end=ELEM_SIZE) %}
            let BCached{{elem}} = mm_Bsub[bidx + {{elem}}][tileCol];
          {% endfor %}
          for (var i = 0; i < {{ ROW_PER_THREAD }}; i++) {
            let ACached = mm_Asub[tileRow + i][k];
            {% for elem in range(end=ELEM_SIZE) %}
                {% if ELEM_TYPE == "f32" %}
                    acc[i] = fma(BCached{{elem}}, ACached, acc[i]);
                {% else %}
                    acc[i] = fma(BCached{{elem}}, {{ ELEM_TYPE }}(ACached[{{elem}}]), acc[i]);
                {% endif %}
            {% endfor %}
          }
        }
        workgroupBarrier();
    }

    {% for innerRow in range(end=ROW_PER_THREAD) %}
        mm_write(batch, globalRow + {{ innerRow }}, globalCol, acc[{{ innerRow }}]);
    {% endfor %}
  }
