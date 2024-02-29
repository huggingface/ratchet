import { expect, describe, beforeAll, it } from "vitest";
import { ApiBuilder, Api, RepoType, default as init } from "@ratchet/ratchet-hub";
import { ModelKey, Model, default as web_init } from "@ratchet/ratchet-web";

beforeAll(async () => {
  await init(); // Init wasm
  await web_init(); // Init wasm
});

describe("The ApiBuilder", () => {
  it("should download a model from HF hub with caching", async () => {
    const api: Api = ApiBuilder.from_hf(`jantxu/ratchet-test`, RepoType.Model).build();

    console.log(`Getting model first time.`);
    const modelStream = await api.get("model.safetensors");
    const bytes = await modelStream.to_uint8();
    expect(bytes.length).toBe(8388776);

    const cached = modelStream.is_cached();
    console.log(`Cached ${cached}.`);
    expect(cached).toBe(false);

    console.log(`Getting model second time.`);
    const modelStream2 = await api.get("model.safetensors");
    const bytes2 = await modelStream2.to_uint8();
    expect(bytes2.length).toBe(8388776);

    const cached2 = modelStream2.is_cached();
    console.log(`Cached ${cached2}.`);
    expect(cached2).toBe(true);
  });
});

describe("Whisper", () => {
  it("should transcribe JFK.", async () => {
      let key = new ModelKey(`ggerganov/whisper.cpp`, `ggml-tiny.bin`);
      let model = await Model.load(key); 
      console.log(`Model loaded: ${model}`);
  });
});
