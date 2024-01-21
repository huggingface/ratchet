# Ratchet

Ratchet is a web-first ML framework, designed to run cross-platform & in the browser.

## Design Decisions

> Through specialization, comes efficiency.

Ratchet is designed for 1 thing only: **Inference on WebGPU**.

This leads us to a few design decisions:
1. Ratchet is **lazy**, no computation is done until the entire computation graph is built and executed. This aligns closely with CUDAGraphs & Command buffers.
2. Ratchet supports **BOTH** static & dynamic graphs, see [Unified Graph Execution by Jittor](http://scis.scichina.com/en/2020/222103.pdf) for more details.
3. Memory planning is crucial. Creation and first bind of a buffer is *expensive* in WebGPU. Therefore, Ratchet uses a greedy algorithm to pool buffers for intermediate results of the CFG.

Take for example Whisper from OpenAI. This is an encoder-decoder model, where the encoder is completely static (i.e everything is known at compile time), and the decoder is very dynamic (KV caching, seq_len increments every step). By allowing both paradigms, we can maximise performance.

## Memory Management

Ratchets top level `Tensor` is just an `Arc` around the `Inner`. Tensors should be cheaply cloneable.
`Inner` contains a struct `Storage`, this is an enum around our 2 managed structures for CPU & GPU: `CpuStorage` & `GpuStorage`.
`CpuStorage` is an `Arc<RwLock<RawCPUBuffer>>`, and `GpuStorage` is an `Arc<RwLock<Buffer>>`.


## Quantization

Due to the buffer binding model of WebGPU, quantisation requires some careful thought in WebGPU.
First let's understand what's required when quantizing / dequantzing.

[Quantization - Neural Network Distiller](https://intellabs.github.io/distiller/algo_quantization.html)

To be brief, values are grouped into blocks (let's say 16 values). This block of values has 1 or more associated, half or full precision values. These values are used to scale the block of values. The question is, how do you manage this in memory?

### Approach 1: Separate tensors 
With your own quant scheme, you could have 2(3) separate tensors, one for weights and one for scales.
This is pretty ideal, because then in the shader you can do the buffer binding like below:

```wgsl
@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<u32>; //this is the quantized weights, wgpu only supports 32 bit values for now

@group(0) @binding(2)
var<storage, read> absmax: array<f32>;

@group(1) @binding(0)
var<storage, read_write> C: array<vec4<f32>>;
```
The above bindings are optimal for performance, and that's what we are optimizing for the most.

But if you have 2 separate tensors, what does your model loading code look like? What does your matmul API look like?

ONNX and others have a different operation altogether `QMatmul`. You'll also require 2 entirely different model implementations like so:
`https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/whisper/quantized_model.rs`
`https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/whisper/model.rs`

This to me seems quite annoying. Is there any way we can avoid it?

I think to summarize the hard requirements of this:
1. Maximal performance is the #1 priority, everything else is secondary.
2. 1 model implementation is very very desirable.
3. The API should be invisible, the user should just call `.matmul()` with Tensor B of datatype Q4_XYZ, and it should just work.

I think the fastest way to achieve that is to use a custom quantization scheme for now. We can revisit this.
