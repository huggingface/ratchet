'use client'
import { useState } from "react";
import styles from "./page.module.css";
import { ModelKey, Model, DecodingOptionsBuilder, default as init } from "@ratchet-ml/ratchet";

export default function Home() {
    const [model, setModel] = useState<any>(null);

    async function loadModel() {
        if (model) return;
        await init();
        let key = new ModelKey(`ggerganov/whisper.cpp`, `ggml-tiny.bin`);
        setModel(await Model.load(key));
    }

    async function runModel() {
        if (!model) return;
        let input = new Float32Array(16000);
        let options = new DecodingOptionsBuilder().build();
        console.log("Options: ", options);
        let result = await model.run({ audio: input, decode_options: options });
        console.log(result);
    }

    return (
        <main className={styles.main}>
            <div className={styles.description}>
                <h1 className={styles.title}>WHISPER TURBO IS BACK</h1>
                <button onClick={loadModel}>Load Model</button>
                <button onClick={runModel}>Run Model</button>
            </div>
        </main>
    );
}
