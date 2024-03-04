const ProgressBar = ({ progress }: any) => {
    return (
        <>
            {progress > 0 && progress < 100 && (
                <div className="flex flex-col gap-2">
                    <div className="h-3 outline outline bg-gray-200">
                        <div
                            className="bg-emerald-600 h-3"
                            style={{ width: `${progress}%` }}
                        ></div>
                    </div>
                </div>
            )}
        </>
    );
};

export default ProgressBar;

