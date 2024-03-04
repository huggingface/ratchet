# Ratchet

### A web-first, cross-platform ML developer toolkit.

[Documentation](https://hf.co)    |     [Discord](https://discord.gg/XFe33KQTG4)

---

**Ship AI inference to your Web, Electron or Tauri apps with ease.**

## Getting Started

Check out our [HuggingFace space](https://huggingface.co/spaces/FL33TW00D-HF/ratchet-whisper) for a live demo!

```javascript
// Asynchronous loading & caching with IndexedDB
let model = await Model.load(AvailableModels.WHISPER_TINY, Quantization.Q8, (p: number) => setProgress(p))
let result = await model.run({ input });
```

## Key Features
- Lazy computation
- In-place by default
- First-class quantized support

## Supported Models
- Whisper


