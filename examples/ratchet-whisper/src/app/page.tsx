'use client'
import { FFmpeg } from '@ffmpeg/ffmpeg'
import { toBlobURL } from '@ffmpeg/util';
import { useEffect, useRef, useState } from "react";
import { Model, DecodingOptionsBuilder, default as init, Task, AvailableModels, Quantization, Segment } from "@ratchet-ml/ratchet-web";
import ConfigModal, { ConfigOptions } from './components/configModal';
import ModelSelector, { humanFileSize } from './components/modelSelector';
import ProgressBar from './components/progressBar';
import MicButton, { AudioMetadata } from './components/micButton';

export default function Home() {
    const [selectedModel, setSelectedModel] = useState<AvailableModels>(AvailableModels.WHISPER_TINY);
    const [loadedModel, setLoadedModel] = useState<AvailableModels | null>(null);
    const [model, setModel] = useState<Model | null>(null);

    const ffmpegRef = useRef(new FFmpeg());
    const [ffmpegLoaded, setFFmpegLoaded] = useState(false);
    const [blobURL, setBlobURL] = useState<string>();
    const [audioData, setAudioData] = useState<Float32Array | null>(null);
    const [audioMetadata, setAudioMetadata] = useState<AudioMetadata | null>(null);
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
        if (!model || !audioData || generating) {
            return
        }
        setSegments([]);
        let floatArray = audioData;
        console.log("Float array: ", floatArray);
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
                return;
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
            setAudioData(pcm16ToIntFloat32(await transcode(audioBytes)));
            setAudioMetadata({
                file: file,
                fromMic: false,
            });
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
            <h1 className="text-blue-700 text-4xl font-extrabold pb-6">Whisper + Ratchet</h1>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="flex flex-col gap-6 w-full">
                    <ModelSelector selectedModel={selectedModel} setSelectedModel={setSelectedModel} loaded={false} progress={0} />
                    {progress > 0 && progress < 100 ? <ProgressBar progress={progress} /> : <></>}
                    {loadedModel != selectedModel ? <button className="outline outline-black text-black font-semibold py-1 px-4 cursor-pointer" onClick={loadModel}>Load Model</button> :
                        <></>}
                    <div className="flex flex-row gap-4">
                        <div className="flex flex-col w-full">
                            <label className="text-black font-semibold">
                                Upload Audio
                            </label>
                            <label
                                className="text-lg outline outline-black  w-full font-semibold py-2.5 px-8 mx-auto cursor-pointer w-full"
                                htmlFor="audioFile"
                            >
                                <div className="flex flex-row justify-between">
                                    <span className="">
                                        {audioMetadata
                                            ? audioMetadata.file.name
                                            : `Select Audio File`}
                                    </span>
                                </div>
                            </label>
                            <input
                                type="file"
                                className="hidden"
                                name="audioFile"
                                id="audioFile"
                                onChange={handleAudioFile()}
                                accept=".wav,.aac,.m4a,.mp4,.mp3"
                            />
                        </div>
                        <MicButton
                            setBlobUrl={setBlobURL}
                            setAudioData={setAudioData}
                            setAudioMetadata={setAudioMetadata}
                        />
                    </div>
                    {blobURL && (
                        <div>
                            <label className="text-white text-xl font-semibold">
                                Your Audio
                            </label>
                            <audio
                                controls
                                key={blobURL}
                                className="mx-auto w-full"
                            >
                                <source
                                    key={blobURL}
                                    src={blobURL}
                                    type="audio/wav"
                                />
                            </audio>
                        </div>
                    )}

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
                <div className="flex flex-col relative w-full z-10 min-h-screen">
                    <div className="flex gap-4 flex-col">
                        {segments &&
                            segments.map(
                                (segment: Segment) => {
                                    return (
                                        <div
                                            key={segment.start}
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
                                    );
                                }
                            )}
                    </div>
                </div>

            </div>
        </>
    );
}
