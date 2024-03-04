'use client'
import { FFmpeg } from '@ffmpeg/ffmpeg'
import { toBlobURL } from '@ffmpeg/util';
import { useEffect, useRef, useState } from "react";
import { Model, DecodingOptionsBuilder, default as init, Task, AvailableModels, Quantization, Segment } from "@ratchet-ml/ratchet-web";
import ConfigModal, { ConfigOptions } from './components/configModal';
import ModelSelector from './components/modelSelector';
import ProgressBar from './components/progressBar';

export default function Home() {
    const [selectedModel, setSelectedModel] = useState<AvailableModels>(AvailableModels.WHISPER_TINY);
    const [loadedModel, setLoadedModel] = useState<AvailableModels | null>(null);
    const [model, setModel] = useState<Model | null>(null);

    const ffmpegRef = useRef(new FFmpeg());
    const [ffmpegLoaded, setFFmpegLoaded] = useState(false);
    const [blobURL, setBlobURL] = useState<string>();
    const [audio, setAudio] = useState(null);
    const [segments, setSegments] = useState<Segment[]>([]);
    const [isConfigOpen, setIsConfigOpen] = useState<boolean>(false);
    const [configOptions, setConfigOptions] = useState<ConfigOptions>({
        language: null,
        task: Task.Transcribe,
        suppress_non_speech: true,
    });
    const [generating, setGenerating] = useState<boolean>(false);
    const [progress, setProgress] = useState<number>(0);


    useEffect(() => {
        (async () => {
            await init();
        })();
    }, [])

    useEffect(() => {
        if (model) {
            console.log("Model loaded: ", model);
        }
        console.log("Selected model: ", selectedModel);
        console.log("Loaded model: ", loadedModel);
    }, [model, selectedModel, loadedModel])

    async function loadModel() {
        let toLoad = AvailableModels[selectedModel] as unknown as AvailableModels;
        setModel(await Model.load(toLoad, Quantization.Q8, (p: number) => setProgress(p)));
        setLoadedModel(selectedModel);
        setProgress(0);
    }

    const loadFFMPEG = async () => {
        console.log("Loading FFmpeg");
        const baseURL = 'https://unpkg.com/@ratchet-ml/ffmpeg-core@0.0.12/dist/umd';
        const ffmpeg = ffmpegRef.current;
        ffmpeg.on('log', ({ message }) => {
            console.log(message);
        });
        await ffmpeg.load({
            coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
            wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
        });
        console.log("Successfully loaded FFmpeg");
        setFFmpegLoaded(true);
    }

    function pcm16ToIntFloat32(pcmData: Uint8Array) {
        let int16Array = new Int16Array(pcmData);
        let float32Array = new Float32Array(int16Array.length);
        for (let i = 0; i < int16Array.length; i++) {
            float32Array[i] = int16Array[i] / 32768.0;
        }
        return float32Array;
    }

    const transcode = async (audioData: Uint8Array) => {
        if (!ffmpegLoaded) {
            await loadFFMPEG();
        }
        const ffmpeg = ffmpegRef.current;
        await ffmpeg.writeFile('input', audioData);

        const cmd = [
            "-nostdin",
            "-threads", "0",
            "-i", "input",
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-loglevel", "debug",
            "-ar", "16000",
            "output.pcm"
        ];
        console.log("Running command: ", cmd);

        await ffmpeg.exec(cmd);
        const data = (await ffmpeg.readFile('output.pcm')) as any;
        setBlobURL(URL.createObjectURL(new Blob([data.buffer], { type: 'audio/wav' })));
        return data.buffer;
    };


    async function runModel() {
        if (!model || !audio || generating) {
            return
        }
        setSegments([]);
        let floatArray = pcm16ToIntFloat32(audio);
        let builder = new DecodingOptionsBuilder();
        let options = builder
            .setLanguage(configOptions.language ? configOptions.language : "en")
            .setTask(configOptions.task)
            .setSuppressBlank(configOptions.suppress_non_speech)
            .build();
        console.log("Options: ", options);
        let callback = (segment: Segment) => {
            if (segment.last) {
                setGenerating(false);
            }
            setSegments((currentSegments) => [...currentSegments, segment]);
        };
        setGenerating(true);
        let result = await model.run({ audio: floatArray, decode_options: options, callback: callback });
        console.log("Result: ", result);
        console.log("Processing time: ", result.processing_time);
    }

    const handleAudioFile = () => async (event: any) => {
        const file = event.target.files[0];
        if (!file) {
            return;
        }
        const reader = new FileReader();
        reader.onload = async () => {
            let audioBytes = new Uint8Array(reader.result as ArrayBuffer);
            setAudio(await transcode(audioBytes));
            setBlobURL(URL.createObjectURL(file));
        };
        reader.readAsArrayBuffer(file);
    };


    return (
        <>
            <ConfigModal
                isModalOpen={isConfigOpen}
                setIsModalOpen={setIsConfigOpen}
                configOptions={configOptions}
                setConfigOptions={setConfigOptions}
            />
            <div className="flex gap-8 flex-row h-screen">
                <div className="flex-1 w-1/2 h-full flex flex-col relative z-10 overflow-hidden">
                    <div className="h-full px-4 xl:pl-32 my-4">
                        <h1 className="text-blue-700 text-4xl font-semibold mx-auto">Whisper + Ratchet</h1>
                        <div className="flex flex-col mx-auto gap-6">
                            <ModelSelector selectedModel={selectedModel} setSelectedModel={setSelectedModel} loaded={false} progress={0} />
                            {progress > 0 && progress < 100 ? <ProgressBar progress={progress} /> : <></>}
                            {loadedModel != selectedModel ? <button className="outline outline-black text-black font-semibold py-1 px-4 cursor-pointer" onClick={loadModel}>Load Model</button> :
                                <></>}
                            <div className="flex flex-row gap-4 justify-between items-center">
                                <input
                                    type="file"
                                    name="audioFile"
                                    id="audioFile"
                                    onChange={handleAudioFile()}
                                />
                                <audio controls key={blobURL}>
                                    <source key={blobURL} src={blobURL} type="audio/wav" />
                                </audio>
                            </div>

                            <div className="flex flex-row gap-4 justify-end items-center">
                                <button className="outline outline-black text-black font-semibold py-3 px-4 cursor-pointer" onClick={() => setIsConfigOpen(true)}>Options</button>
                                <button className="outline outline-black text-black font-semibold py-3 px-4 cursor-pointer" onClick={runModel}>
                                    {
                                        generating ?
                                            <div className="flex p-4">
                                                <span className="loader"></span>
                                            </div>
                                            : "Run Model"
                                    }
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="flex-1 w-1/2 h-full flex flex-col relative z-10">
                    <div className="h-full flex flex-col mx-auto px-4 xl:pr-32 overflow-scroll py-12 w-full">
                        <div className="flex flex-col h-full">
                            {segments &&
                                segments.map(
                                    (segment: Segment) => {
                                        return (
                                            <div
                                                key={segment.start}
                                                className="flex w-full py-4"
                                            >
                                                <div
                                                    className={`rounded p-4 bg-white outline outline-2 outline-black shadow-lg align-right`}
                                                >
                                                    <div className=" text-lg mb-2">
                                                        {segment.start}
                                                        {" -> "}
                                                        {segment.stop}
                                                    </div>
                                                    <div className="mb-2 text-lg text-slate-900 text-right">
                                                        {segment.text}
                                                    </div>
                                                </div>
                                            </div>
                                        );
                                    }
                                )}
                        </div>
                    </div>
                </div>

            </div>
        </>
    );
}
