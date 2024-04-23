'use client'

import { AvailableModels, Model, Quantization, default as init } from "@ratchet-ml/ratchet-web";
import { useEffect, useState } from "react";

export default function Home() {
    const [selectedModel, setSelectedModel] = useState<AvailableModels>({ Phi: "phi2" });
    const [loadedModel, setLoadedModel] = useState<AvailableModels | null>(null);
    const [model, setModel] = useState<Model | null>(null);
    const [generating, setGenerating] = useState<boolean>(false);
    const [progress, setProgress] = useState<number>(0);
    const [fetching, setFetching] = useState<boolean>(false);
    const [generatedText, setGeneratedText] = useState<string>("");
    const [prompt, setPrompt] = useState<string>("");

    useEffect(() => {
        (async () => {
            await init();
        })();
    }, []);

    async function loadModel() {
        if (fetching) {
            return;
        }
        setFetching(true);
        setModel(await Model.load(selectedModel, Quantization.Q8_0, (p: number) => setProgress(p)));
        setLoadedModel(selectedModel);
        setProgress(0);
        setFetching(false);
    }

    async function runModel() {
        if (!model || generating) {
            return;
        }

        setGenerating(true);
        setGeneratedText("");

        let cb = (s: string) => {
            setGeneratedText((prevText) => prevText + s.replace(/\n/g, "<br />"));
        };

        let input = {
            prompt: prompt,
            callback: cb,
        };

        await model.run(input);
        setGenerating(false);
    }

    return (
        <div className="max-w-3xl mx-auto p-4">
            <h1 className="text-3xl font-bold mb-4">Phi2</h1>
            <div className="mb-4">
                <button
                    className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded mr-2"
                    onClick={loadModel}
                    disabled={generating || loadedModel !== null}
                >
                    Load Model
                </button>
                <button
                    className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded"
                    onClick={runModel}
                    disabled={!model || generating}
                >
                    Run Model
                </button>
            </div>
            <div className="mb-4">
                {fetching && <p>Progress: {progress.toFixed(2)}%</p>}
            </div>
            <div className="mb-4">
                <label htmlFor="prompt" className="block mb-2 font-bold">
                    Prompt:
                </label>
                <textarea
                    id="prompt"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    disabled={generating}
                    className="w-full h-32 p-2 border border-gray-300 rounded"
                />
            </div>
            <div className="border border-gray-300 p-4 rounded">
                <h2 className="text-xl font-bold mb-2">Generated Text:</h2>
                <p dangerouslySetInnerHTML={{ __html: generatedText }}></p>
            </div>
        </div>
    );
}
