//TODO: Kernel is bad
//Each workgroup is responsible for a single filter.
//Each thread computes a single element of the output.
//Each thread places the 3 column wide filter over the input, and multiplies and accumulates the values
//to a SINGLE output element.
@group(0) @binding(0)
var<storage, read> X: array<f32>;

@group(0) @binding(1)
var<storage, read> W: array<f32>;

@group(0) @binding(2)
var<storage, read> B: array<f32>;

@group(0) @binding(3)
var<storage, read_write> Y: array<f32>;

struct Meta {
    padding: u32,
    stride: u32,
    Cin: u32,
    Lin: u32,
    KS: u32,
    F_numel: u32,
    Lout: u32,
    Fperthread: u32,
}

@group(1) @binding(0)
var<uniform> metadata: Meta;

var<workgroup> F: array<f32, {{ F_numel }}u>;

fn kahans_inner(input_index: u32, filter_index: u32, output_index: u32, bias_index: u32, start: u32, end: u32) {
    var inp = vec3<f32>(0f);
    var kernel = vec3<f32>(0f);
    var acc = vec3<f32>(0f); 
    var c = vec3<f32>(0f);
    for(var i = 0u; i < metadata.Cin; i++) {
        let input_start = input_index + (i * metadata.Lin) - 1u; //-1 is for padding
        //We only populate the input between the provided indices, used for padding
        for(var j = start; j <= end; j++) {
            inp[j] = X[input_start + j];
        }

        let filter_start = i * metadata.KS;
        kernel.x = F[filter_start];
        kernel.y = F[filter_start + 1u];
        kernel.z = F[filter_start + 2u];

        let r = fma(inp, kernel, -c);
        let t = acc + r;
        c = (t - acc) - r;
        acc = t;
    }        
    Y[output_index] = acc.x + acc.y + acc.z + B[bias_index];
}

fn inner(input_index: u32, filter_index: u32, output_index: u32, bias_index: u32, start: u32, end: u32) {
    var inp = vec3<f32>(0f);
    var kernel = vec3<f32>(0f);
    var acc = vec3<f32>(0f); 
    for(var i = 0u; i < metadata.Cin; i++) {
        let input_start = input_index + (i * metadata.Lin) - 1u; //-1 is for padding
        //We only populate the input between the provided indices, used for padding
        for(var j = start; j <= end; j++) {
            inp[j] = X[input_start + j];
        }

        let filter_start = i * metadata.KS;
        kernel.x = F[filter_start];
        kernel.y = F[filter_start + 1u];
        kernel.z = F[filter_start + 2u];

        acc = fma(inp, kernel, acc);
    }        
    Y[output_index] = acc.x + acc.y + acc.z + B[bias_index];
}


//Each thread may load more than 1 element into shared memory
fn load_filters_into_smem(local_id: vec3<u32>, filter_index: u32) {
    let windex = filter_index + (local_id.x * metadata.Fperthread);
    let findex = (local_id.x * metadata.Fperthread);
    for(var i=0u; i < metadata.Fperthread; i++) {
        if findex + i < metadata.F_numel {
            F[findex + i] = W[windex + i];
        }
    }
}

//TODO: support dynamic padding
@compute @workgroup_size(256,1,1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {

    let input_index = (workgroup_id.x * {{ workgroup_size_x }}u + local_id.x) * metadata.stride;
    let filter_index = (workgroup_id.y * {{ F_numel }}u); 
    
    load_filters_into_smem(local_id, filter_index);
    workgroupBarrier();

    if input_index >= metadata.Lin {
        //Break after loading because all threads may be needed for loading F
        return;
    }

    let output_index = (workgroup_id.x * {{ workgroup_size_x }}u + local_id.x) + (workgroup_id.y * metadata.Lout);
    let bias_index = workgroup_id.y;

    if input_index == metadata.Lin - metadata.padding {
        inner(input_index, filter_index, output_index, bias_index, 0u, 1u);
    } else if input_index == 0u {
        inner(input_index, filter_index, output_index, bias_index, 1u, 2u);
    } else {
        inner(input_index, filter_index, output_index, bias_index, 0u, 2u);
    }
}


