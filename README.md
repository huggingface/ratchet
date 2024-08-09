<div align="center">
<img width="550px" height="200px" src="https://github.com/FL33TW00D/ratchet/raw/master/.github/ratchet.png">
<p><a href="https://huggingface.co/spaces/FL33TW00D-HF/ratchet-whisper">Demo Site</a> | <a href="https://discord.gg/XFe33KQTG4">Discord</a> | <a href="https://github.com/users/FL33TW00D/projects/3">Roadmap</a></p>
<p align="center">
A web-first, cross-platform ML developer toolkit
</p>
<br>
</div>

We are on a mission to bring fast, cross platform GPU accelerated inference on native + browser.

> [!NOTE]  
> Ratchet is currently in active development. We are working on making it more stable and adding more models.

## Getting Started

The easiest way to experience Ratchet is to check out our [Hugging Face spaces](https://huggingface.co/FL33TW00D-HF):
- [Whisper](https://huggingface.co/spaces/FL33TW00D-HF/ratchet-whisper)
- [Phi](https://huggingface.co/spaces/FL33TW00D-HF/ratchet-phi)

To dig deeper, check out the [examples](https://github.com/FL33TW00D/ratchet/tree/master/examples)

We welcome contributions from the community. If you have any ideas or suggestions, please feel free to open an issue or pull request.

### Javascript

```javascript
// Asynchronous loading & caching with IndexedDB
let model = await Model.load(AvailableModels.WHISPER_TINY, Quantization.Q8, (p: number) => setProgress(p))
let result = await model.run({ input });
```

### Rust

Rust crate & CLI coming soon...

## Philosophy

We want a toolkit for developers to make integrating performant AI functionality into existing production applications easy.
The following principles will help us accomplish this:
1. **Inference only**
2. **WebGPU/CPU only**
3. First class quantization support
4. Lazy computation
5. Inplace by default

## Supported Models
- Whisper
- Phi 2 & 3
- Moondream

## Upcoming Models
- Gemini 2 2B