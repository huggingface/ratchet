const ProgressBar = ({ progress }: any) => {
    return (
        <>
            {progress > 0 && progress < 100 && (
                <div className="flex flex-col gap-2">
                    <div className="h-1.5 outline outline-gray-200 bg-gray-200">
                        <div
                            className="bg-emerald-600 h-1.5"
                            style={{ width: `${progress}%` }}
                        ></div>
                    </div>
                </div>
            )}
        </>
    );
};

export default ProgressBar;

