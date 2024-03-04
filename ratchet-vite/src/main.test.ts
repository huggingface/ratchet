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
        const bytes = await api.get("model.safetensors");
        expect(bytes.length).toBe(8388776);
    });
});

describe("Whisper", () => {
    it("should transcribe JFK.", async () => {
        let key = new ModelKey(`ggerganov/whisper.cpp`, `ggml-tiny.bin`);
        let model = await Model.load(key);
        console.log(`Model loaded: ${model}`);
    });
});
