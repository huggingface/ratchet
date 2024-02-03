// sum.test.js
import { expect, test, describe, beforeAll, it } from "vitest";
import { ApiBuilder, Api, default as init } from "@ratchet/ratchet-client";

beforeAll(async () => {
  await init(); // Init wasm
});

describe("The ApiBuilder", () => {
  it("should download the full model model ", async () => {
    const api: Api = ApiBuilder.from_hf(`jantxu/ratchet-test`).build();
    const modelStream = await api.get("model.safetensors");

    const bytes = await modelStream.bytes();
    expect(bytes.length).toBe(8388776);
  });
});
