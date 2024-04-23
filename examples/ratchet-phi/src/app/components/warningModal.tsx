import React from "react";
import Modal from "react-responsive-modal";

interface WarningModalProps {
    isModalOpen: boolean;
    setIsModalOpen: (value: boolean) => void;
    loadModel: () => void;
}

const WarningModal = ({ isModalOpen, setIsModalOpen, loadModel }: WarningModalProps) => {
    const handleModalClose = () => {
        setIsModalOpen(false);
    };

    const closeIcon = (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            version="1.1"
            width="50"
            height="50"
            viewBox="0 0 78 97.5"
            fill="currentColor"
        >
            <g>
                <rect x="54" y="54" width="6" height="6" />
                <rect x="36" y="36" width="6" height="6" />
                <rect x="30" y="42" width="6" height="6" />
                <rect x="24" y="48" width="6" height="6" />
                <rect x="18" y="54" width="6" height="6" />
                <rect x="42" y="30" width="6" height="6" />
                <rect x="48" y="24" width="6" height="6" />
                <rect x="54" y="18" width="6" height="6" />
                <rect x="42" y="42" width="6" height="6" />
                <rect x="48" y="48" width="6" height="6" />
                <rect x="30" y="30" width="6" height="6" />
                <rect x="18" y="18" width="6" height="6" />
                <rect x="24" y="24" width="6" height="6" />
            </g>
        </svg>
    );

    return (
        <>
            {isModalOpen ? (
                <Modal
                    classNames={{
                        modal: "outline w-1/2 md:w-1/2 xl:w-1/3 2xl:w-1/4 overflow-x-hidden text-black",
                    }}
                    open={isModalOpen}
                    onClose={handleModalClose}
                    center
                    closeIcon={closeIcon}
                >
                    <div
                        className="flex flex-col text-2xl h-full text-center"
                    >
                        <div className="mx-8 mt-8">
                            <p>
                                ⚠️  You are about to download a 2.9GB file. Click to confirm.
                            </p>
                            <button
                                className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded mt-4"
                                onClick={() => {
                                    setIsModalOpen(false);
                                    loadModel();
                                }}
                            >
                                Confirm
                            </button>
                        </div>
                    </div>
                </Modal>
            ) : (
                <></>
            )}
        </>
    );
};

export default WarningModal;

