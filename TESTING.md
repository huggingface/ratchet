# Testing

Validating a machine learning framework against a solid ground truth is essential.

We test Ratchet in the following ways:
1. `Native` - On native, we use `proptest` in order to fuzz each of our operations against `PyTorch`.
2. `Web` - On web, we use `PyTorch` to generate ground truth (see `ratchet-ground` crate), and then embed the ground truth in a `wasm-bindgen-test` binary, which we then run against a range of headless browsers.
