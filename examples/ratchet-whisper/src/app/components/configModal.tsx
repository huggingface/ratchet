import React, { useState, useEffect } from "react";
import Modal from "react-responsive-modal";
import LanguageDropdown from "./languageDropdown";
import SuppressComponent from "./suppressSelector";
import TaskComponent from "./taskSelector";
import { Task } from "@ratchet-ml/ratchet-web";

interface ConfigModalProps {
    isModalOpen: boolean;
    setIsModalOpen: React.Dispatch<React.SetStateAction<boolean>>;
    configOptions: ConfigOptions;
    setConfigOptions: React.Dispatch<React.SetStateAction<ConfigOptions>>;
}

export interface ConfigOptions {
    language: string | null;
    task: Task;
    suppress_non_speech: boolean;
}

const ConfigModal = (props: ConfigModalProps) => {
    useEffect(() => {
        //@ts-ignore
        if (!navigator.gpu) {
            props.setIsModalOpen(true);
            return;
        }
    }, []);

    const handleModalClose = () => {
        props.setIsModalOpen(false);
    };

    return (
        <>
            <Modal
                classNames={{
                    modal: "!outline !outline-black h-3/4 w-3/4 md:w-3/4 xl:w-1/2 2xl:w-1/3 overflow-x-hidden",
                }}
                open={props.isModalOpen}
                onClose={handleModalClose}
                center
            >
                <div
                    className="flex flex-col text-2xl h-full text-center"
                >
                    <div className="flex flex-col p-8 gap-y-8 mx-auto w-full">
                        <LanguageDropdown configOptions={props.configOptions} setConfigOptions={props.setConfigOptions} />
                        <TaskComponent configOptions={props.configOptions} setConfigOptions={props.setConfigOptions} />
                        <SuppressComponent configOptions={props.configOptions} setConfigOptions={props.setConfigOptions} />
                    </div>
                </div>
            </Modal>
        </>
    );
};

export default ConfigModal;

