use clap::Parser;
use hf_hub::api::sync::Api;
use log::info;
use ratchet::{Device, DeviceRequest};
use ratchet_loader::GGMLCompatible;
use ratchet_models::model::Whisper;
use ratchet_models::options::DecodingOptionsBuilder;
use ratchet_models::transcribe::transcribe;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[clap(name = "ratchet", about = "A CLI for running the Whisper model")]
struct Opts {
    /// The name of the model to use
    #[clap(value_parser)]
    repo: String,

    /// The path to the audio file to transcribe
    #[clap(short, long, value_parser, value_name = "FILE")]
    audio_path: PathBuf,

    /// The path to the model file (optional, defaults to "tiny_q8.bin")
    #[clap(short, long, value_parser, default_value = "tiny_q8.bin")]
    model_file: String,
}

use std::process::Command;

fn load_sample<P: AsRef<Path>>(path: P) -> Vec<f32> {
    let path = path.as_ref();
    let output = Command::new("ffmpeg")
        .args(&[
            "-nostdin",
            "-threads",
            "0",
            "-i",
            path.to_str().unwrap(),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-loglevel",
            "error",
            "-ar",
            "16000",
            "-",
        ])
        .output()
        .expect("Failed to execute ffmpeg command");

    if !output.status.success() {
        panic!(
            "ffmpeg command failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let audio_data = output.stdout;
    let mut samples = Vec::new();

    for chunk in audio_data.chunks(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
        samples.push(sample);
    }

    samples
}

fn main() {
    let opts = Opts::parse();
    let api = Api::new().unwrap();
    let model = api.model(opts.repo);
    let model_path = model.get(opts.model_file.as_str()).unwrap();

    let samples = load_sample(&opts.audio_path);

    let options = DecodingOptionsBuilder::new().build();

    let mut reader = std::io::BufReader::new(std::fs::File::open(model_path).unwrap());
    let gg_disk = Whisper::load_ggml(&mut reader).unwrap();

    let device = Device::request_device(DeviceRequest::GPU).unwrap();
    let mut whisper = Whisper::load(&gg_disk, &mut reader, device).unwrap();

    let transcript = transcribe(&mut whisper, samples, options).unwrap();
    println!("{}", transcript.formatted.unwrap());
    info!("Processing time: {:?}", transcript.processing_time);
}
