# Ratchet

Ratchet is a web-first ML framework, designed to run cross-platform & in the browser.

## Design Decisions

> Through specialization, comes efficiency.

Ratchet is designed for 1 thing only: Inference on WebGPU.

This leads us to a few design decisions:
1. Ratchet is **lazy**, no computation is done until the entire computation graph is built and executed. This aligns closely with CUDAGraphs & Command buffers.
2. Ratchet supports **BOTH** static & dynamic graphs, this is key.
    - The graph is implicitly defined through tensor operations. If any of the tensors are defined with a *symbolic dimension* (i.e a dimension not known until runtime, e.g sequence_len), the graph is dynamic. When the graph is dynamic, the graph is recompiled on inference pass (because runtime information is required).
    - If no tensors contain a symbolic dimension, the graph is static. This means the graph is compiled into a single command buffer, and is repeatedly called with different input data (brrr).
    
    By exposing symbolic dimensions to the user, they can code their models with the CG in mind.

Why do this? Take for example Whisper from OpenAI. This is an encoder-decoder model, where the encoder is completely static (i.e everything is known at compile time), and the decoder is very dynamic (KV caching, seq_len increments every step). By allowing both paradigms, we can maximise performance._full

