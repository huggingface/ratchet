# Ratchet

Ratchet is a web-first ML framework, designed to run cross-platform & in the browser.

## Design Decisions

> Through specialization, comes efficiency.

Ratchet is designed for 1 thing only: **Inference on WebGPU**.

This leads us to a few design decisions:
1. Ratchet is **lazy**, no computation is done until the entire computation graph is built and executed. This aligns closely with CUDAGraphs & Command buffers.
2. Ratchet supports **BOTH** static & dynamic graphs, this is key.
    - The graph is implicitly defined through tensor operations. If any of the tensors are defined with a *symbolic dimension* (i.e a dimension not known until runtime, e.g sequence_len), the graph is dynamic. When the graph is dynamic, the graph is recompiled on inference pass (because runtime information is required).
    - If no tensors contain a symbolic dimension, the graph is static. This means the graph is compiled into a single command buffer, and is repeatedly called with different input data (brrr).
    
    By exposing symbolic dimensions to the user, they can code their models with the CG in mind.
3. Memory planning is crucial. Creation and first bind of a buffer is *expensive* in WebGPU. Therefore, Ratchet uses a greedy algorithm to pool buffers for intermediate results of the CFG.

Why do this? 

Take for example Whisper from OpenAI. This is an encoder-decoder model, where the encoder is completely static (i.e everything is known at compile time), and the decoder is very dynamic (KV caching, seq_len increments every step). By allowing both paradigms, we can maximise performance.



## Quantization

Due to the buffer binding model of WebGPU, quantisation requires some careful thought in WebGPU.
First let's understand what's required when quantizing / dequantzing.

[Quantization - Neural Network Distiller](https://intellabs.github.io/distiller/algo_quantization.html)

To be brief, we need to group values into blocks, (let's say 16 values), and then we need, to get the absolute maximum value of the block.
This works quite well for performant matrix multiplication in WebGPU and other graphics based shading languages, **IF** you bind the buffers separately.
Binding them separately allows you to pack 4 quantized values in into a `vec4`, and bind the absmax separate.
Then it's just 2 loads, 1 for the 4 values of the group, and 1 for the absmax.

```wgsl
@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<u32>;

@group(0) @binding(2)
var<storage, read> absmax: array<f32>;

@group(1) @binding(0)
var<storage, read_write> C: array<vec4<f32>>;
```

What's the problem with the above approach?
With different buffer bindings you then end up with much less code reuse between a standard matmul & a quantized matrix multiply.

TBD

