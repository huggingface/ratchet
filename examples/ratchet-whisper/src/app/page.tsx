'use client'
import { FFmpeg } from '@ffmpeg/ffmpeg'
import { fetchFile, toBlobURL } from '@ffmpeg/util';
import { useRef, useState } from "react";
import styles from "./page.module.css";
import { ModelKey, Model, DecodingOptionsBuilder, default as init } from "@ratchet-ml/ratchet";

export default function Home() {
    const [model, setModel] = useState<any>(null);
    const [loaded, setLoaded] = useState(false);
    const ffmpegRef = useRef(new FFmpeg());
    const [blobURL, setBlobURL] = useState("");
    const [audio, setAudio] = useState(null);
    const [transcript, setTranscript] = useState(null);

    async function loadModel() {
        if (model) return;
        await init();
        let key = new ModelKey(`ggerganov/whisper.cpp`, `ggml-tiny.bin`);
        setModel(await Model.load(key));
    }

    const loadFFMPEG = async () => {
        console.log("Loading FFmpeg");
        const baseURL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd';
        //const baseURL = 'https://unpkg.com/@ratchet-ml/ffmpeg-core@0.0.7/dist/umd';
        const ffmpeg = ffmpegRef.current;
        ffmpeg.on('log', ({ message }) => {
            //console.log(message);
        });
        // toBlobURL is used to bypass CORS issue, urls with the same
        // domain can be used directly.
        await ffmpeg.load({
            coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
            wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
        });
        console.log("Successfully loaded FFmpeg");
        setLoaded(true);
    }

    function pcm16ToIntFloat32(pcmData) {
        let int16Array = new Int16Array(pcmData);
        let float32Array = new Float32Array(int16Array.length);
        for (let i = 0; i < int16Array.length; i++) {
            float32Array[i] = int16Array[i] / 32768.0;
        }
        return float32Array;
    }

    const transcode = async (audioData: Uint8Array) => {
        if (!loaded) {
            await loadFFMPEG();
        }
        const ffmpeg = ffmpegRef.current;
        await ffmpeg.writeFile('input.wav', audioData);

        // Adjusting the command according to your specifications
        const cmd = [
            "-nostdin",
            "-threads", "0",
            "-i", "input.wav",
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
        if (!model) {
            return
        }
        let floatArray = pcm16ToIntFloat32(audio);
        let options = new DecodingOptionsBuilder().build();
        console.log("Options: ", options);
        let result = await model.run({ audio: floatArray, decode_options: options });
        console.log("Processing time: ", result.processing_time);
        setTranscript(result.formatted);
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
        <main className={styles.main}>
            <h1 className={styles.title}>WHISPER TURBO IS BACK</h1>
            <input
                type="file"
                className="hidden"
                name="audioFile"
                id="audioFile"
                onChange={handleAudioFile()}
            />
            <div className={styles.buttonsContainer}>
                <button onClick={loadModel}>Load Model</button>
                <button onClick={runModel}>Run Model</button>
            </div>
            <audio controls key={blobURL}>
                <source key={blobURL} src={blobURL} type="audio/wav" />
            </audio>
            {transcript && transcript.split("\n").map((line, index) => {
                return <p key={`t-${index}`}>{line}</p>
            })}
        </main>
    );
}
