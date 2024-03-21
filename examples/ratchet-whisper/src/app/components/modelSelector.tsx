import { useState } from "react";
import { AvailableModels, Whisper } from "@ratchet-ml/ratchet-web";

interface ModelSelectorProps {
    selectedModel: AvailableModels | null;
    setSelectedModel: (model: AvailableModels) => void;
    loaded: boolean;
    progress: number;
}

const UNITS = [
    "byte",
    "kilobyte",
    "megabyte",
    "gigabyte",
];
const BYTES_PER_KB = 1000;

export function humanFileSize(sizeBytes: number | bigint): string {
    let size = Math.abs(Number(sizeBytes));

    let u = 0;
    while (size >= BYTES_PER_KB && u < UNITS.length - 1) {
        size /= BYTES_PER_KB;
        ++u;
    }

    return new Intl.NumberFormat([], {
        style: "unit",
        unit: UNITS[u],
        unitDisplay: "short",
        maximumFractionDigits: 1,
    }).format(size);
}

export function availableModelToString(model: AvailableModels): string {
    if ("Whisper" in model) {
        return model.Whisper;
    } else if ("Llama" in model) {
        return model.Llama;
    }
    return "";
}

const ModelSelector = (props: ModelSelectorProps) => {
    const { selectedModel, setSelectedModel, loaded, progress } = props;
    const [dropdownOpen, setDropdownOpen] = useState<boolean>(false);

    const whisper = ["tiny", "base", "small", "medium", "large_v2", "large_v3", "distil_large_v3"] as const;
    type WhisperIter = typeof whisper[number];

    const modelNames = [
        ...whisper,
    ];

    const displayModels = () => {
        return modelNames.map((model, idx) => (
            <li key={model as string}>
                <a
                    className={`bg-white hover:bg-slate-100 py-2 px-8 font-semibold text-xl block whitespace-no-wrap cursor-pointer ${idx === modelNames.length - 1 ? "rounded-b-md" : ""
                        }`}
                    onClick={() => {
                        const isWhisper = whisper.includes(model as WhisperIter);
                        if (isWhisper) {
                            setSelectedModel({ Whisper: model as Whisper });
                        }
                        setDropdownOpen(false);
                    }}
                >
                    {model}
                </a>
            </li>
        ));
    };

    return (
        <>
            {progress > 0 && !loaded && (
                <div className="flex flex-row justify-between">
                    <label className="text-white text-xl font-semibold text-right">
                        {progress.toFixed(2)}%
                    </label>
                </div>
            )}
            <div className="group inline-block relative w-full">
                <button
                    className="font-semibold text-xl py-2.5 px-8 w-full inline-flex items-center outline outline-black"
                    onClick={() => setDropdownOpen(!dropdownOpen)}
                >
                    <span className="mr-1">
                        {selectedModel
                            ? availableModelToString(selectedModel)
                            : "Select a model"}
                    </span>
                    <svg
                        className="fill-current h-4 w-4"
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 20 20"
                    >
                        <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
                    </svg>
                </button>
                <ul
                    className="absolute outline group-hover:block z-50 w-full shadow-lg shadow-black"
                    style={{
                        display: dropdownOpen ? "block" : "none",
                    }}
                >
                    {displayModels()}
                </ul>
            </div>
        </>
    );
};

export default ModelSelector;

