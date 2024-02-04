// sum.test.js
import { expect, test, describe, beforeAll, it } from "vitest";
import { ApiBuilder, Api, default as init } from "@ratchet/ratchet-client";

beforeAll(async () => {
  await init(); // Init wasm
});

describe("The ApiBuilder", () => {
  it("should download a model from HF hub", async () => {
    const api: Api = ApiBuilder.from_hf(`jantxu/ratchet-test`).build();

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
