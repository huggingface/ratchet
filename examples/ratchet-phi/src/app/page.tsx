'use client'

import { AvailableModels, Model, Quantization, default as init } from "@ratchet-ml/ratchet-web";
import { useEffect, useState } from "react";
import ProgressBar from "./components/progressBar";
import WebGPUModal from "./components/WebGPUModal";
import WarningModal from "./components/warningModal";

export default function Home() {
    const [selectedModel, setSelectedModel] = useState<AvailableModels>({ Phi: "phi3" });
    const [loadedModel, setLoadedModel] = useState<AvailableModels | null>(null);
    const [model, setModel] = useState<Model | null>(null);
    const [generating, setGenerating] = useState<boolean>(false);
    const [progress, setProgress] = useState<number>(0);
    const [generatedText, setGeneratedText] = useState<string>("");
    const [prompt, setPrompt] = useState<string>("What is the meaning of life?");
    const [loadingModel, setLoadingModel] = useState<boolean>(false);
    const [isWarningOpen, setIsWarningOpen] = useState<boolean>(false);
    const [ratchetDBExists, setRatchetDBExists] = useState<boolean>(false);


    useEffect(() => {
        (async () => {
            await init();
            setRatchetDBExists((await window.indexedDB.databases()).map(db => db.name).includes("ratchet"));
        })();
    }, []);

    async function loadModel() {
        setLoadingModel(true);
        setModel(await Model.load(selectedModel, Quantization.Q8_0, (p: number) => setProgress(p)));
        setLoadedModel(selectedModel);
        setProgress(0);
        setLoadingModel(false);
    }

    async function runModel() {
        if (!model || generating) {
            return;
        }

        setGenerating(true);
        setGeneratedText("");

        let cb = (s: string) => {
            setGeneratedText((prevText) => {
                return prevText + s.replace(/\n/g, "<br />");
            });
        };

        let input = {
            prompt: prompt,
            callback: cb,
        };

        await model.run(input);
        setGenerating(false);
    }

    return (
        <div className="flex flex-col min-h-screen">
            <div className="w-full mx-auto p-8 flex-grow">
                <h1 className="text-4xl font-bold mb-8 text-blue-600">Ratchet + Phi</h1>
                {generatedText ?
                    <div className="border border-gray-300 bg-white p-6 rounded-lg shadow-md">
                        <p
                            className="text-md text-gray-800"
                            dangerouslySetInnerHTML={{ __html: generatedText }}
                        ></p>
                    </div>
                    : <></>}
            </div>

            <div className="p-8">
                <label htmlFor="prompt" className="block mb-2 font-bold text-gray-700">
                    Prompt:
                </label>
                <textarea
                    id="prompt"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    disabled={generating}
                    className="w-full h-32 p-4 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-800"
                    placeholder="Enter your prompt here..."
                />
            </div>


            <ProgressBar progress={progress} />
            <div className="px-8 flex justify-end space-x-4">
                {loadedModel ?
                    <></>
                    :
                    <button
                        className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-6 rounded-lg shadow-md transition duration-300"
                        onClick={() => {
                            if (!ratchetDBExists) {
                                setIsWarningOpen(true);
                            } else {
                                loadModel();
                            }
                        }}
                        disabled={generating || loadedModel !== null}
                    >
                        {loadingModel ? "Loading Model..." : "Load Model"}
                    </button>
                }
                <button
                    className="bg-green-500 hover:bg-green-600 text-white font-bold py-3 px-6 rounded-lg shadow-md transition duration-300"
                    onClick={runModel}
                    disabled={!model || generating}
                >
                    Run Model
                </button>
            </div>
            <footer className="w-full mt-8 py-6 bg-white border-t border-gray-300 flex justify-between mx-auto px-8">
                <a
                    href="https://huggingface.co/microsoft/phi3"
                    className="text-blue-500 hover:text-blue-600 transition duration-300"
                >
                    Model Checkpoints
                </a>
                <a
                    href="https://huggingface.co/papers/2404.14219"
                    className="text-blue-500 hover:text-blue-600 transition duration-300"
                >
                    Technical Report
                </a>
                <a
                    href="https://github.com/huggingface/ratchet"
                    className="text-blue-500 hover:text-blue-600 transition duration-300"
                >
                    Ratchet
                </a>
            </footer>
            <WarningModal isModalOpen={isWarningOpen} setIsModalOpen={setIsWarningOpen} loadModel={loadModel} />
            <WebGPUModal />
        </div>
    );
}
