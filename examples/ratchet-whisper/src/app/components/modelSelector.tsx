import { useState } from "react";
import { AvailableModels } from "@ratchet-ml/ratchet";

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

const ModelSelector = (props: ModelSelectorProps) => {
    const { selectedModel, setSelectedModel, loaded, progress } = props;
    const [dropdownOpen, setDropdownOpen] = useState<boolean>(false);
    const modelNames = Object.values(AvailableModels).filter(value => typeof value === 'string') as string[];
    const modelValues = Object.values(AvailableModels);

    const displayModels = () => {
        return modelNames.map((model, idx) => (
            <li key={model as string}>
                <a
                    className={`bg-white hover:bg-slate-100 py-2 px-8 font-semibold text-xl block whitespace-no-wrap cursor-pointer ${idx === modelNames.length - 1 ? "rounded-b-md" : ""
                        }`}
                    onClick={() => {
                        setSelectedModel(modelValues[idx] as number);
                        setDropdownOpen(false);
                    }}
                >
                    {model as string}
                </a>
            </li>
        ));
    };

    return (
        <>
            <div className="flex flex-row justify-between">
                {progress > 0 && !loaded && (
                    <label className="text-white text-xl font-semibold text-right">
                        {progress.toFixed(2)}%
                    </label>
                )}
            </div>
            <div className="group inline-block relative w-full">
                <button
                    className="font-semibold text-xl py-2.5 px-8 w-full inline-flex items-center outline outline-black"
                    onClick={() => setDropdownOpen(!dropdownOpen)}
                >
                    <span className="mr-1">
                        {selectedModel
                            ? selectedModel
                            : "Select Model"}
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
                    className="absolute outline group-hover:block z-10 w-full"
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

