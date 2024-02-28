# ratchet-client

`ratchet-client` is the bridge between the JS world & the Rust world.

Using `web-sys` APIs to access host browser capabilities.

The 2 key components are:
1. `hf-hub`: Clone of the `hf-hub` crate, but for `wasm32`.
2. `ratchet-db`: A simple DB for versioning models & tokenizers, built ontop of `indexeddb`.
