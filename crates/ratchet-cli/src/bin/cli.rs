use clap::{value_parser, Arg, ArgMatches, Command};
use hf_hub::api::sync::Api;
use ndarray::Axis;
use ndarray_stats::QuantileExt;
use ratchet::{shape, Device, DeviceRequest, Tensor};
use ratchet_loader::gguf::gguf::{self, Header};
use ratchet_models::registry::{AvailableModels, Quantization, WhisperVariants as RegistryWhisper};
use ratchet_models::whisper::options::DecodingOptionsBuilder;
use ratchet_models::whisper::transcribe::transcribe;
use ratchet_models::{phi2::Phi2, whisper::Whisper};
use ratchet_nn::Module;
use std::io::Write;
use std::path::Path;
use std::process::Command as TermCommand;
use tokenizers::Tokenizer;

fn ffmpeg_preproc<P: AsRef<Path>>(path: P) -> Vec<f32> {
    let path = path.as_ref();
    let output = TermCommand::new("ffmpeg")
        .args([
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

pub fn start_logger() {
    let logger = fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "{}[{}][{}] {}",
                chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"),
                record.target(),
                record.level(),
                message
            ))
        })
        .level_for("tokenizers", log::LevelFilter::Off)
        .level(log::LevelFilter::Warn)
        .apply();
    match logger {
        Ok(_) => log::info!("Logging initialized."),
        Err(error) => eprintln!("Error initializing logging: {:?}", error),
    }
}

fn handle_whisper(matches: &ArgMatches, api: Api) {
    let mut whisper = if let Some(variant) = matches.get_one::<RegistryWhisper>("variant") {
        let model = AvailableModels::Whisper(variant.clone());
        let repo = api.model(model.repo_id());
        let model_path = repo.get(&model.model_id(Quantization::Q8_0)).unwrap();

        let mut reader = std::io::BufReader::new(std::fs::File::open(model_path).unwrap());
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let header = gguf::Header::read(&mut reader).unwrap();
        Whisper::load(header, &mut reader, device).unwrap()
    } else {
        panic!("Model not found");
    };

    if let Some(input) = matches.get_one::<String>("input") {
        let options = DecodingOptionsBuilder::new().build();
        let samples = ffmpeg_preproc(input);
        let transcript =
            transcribe(&mut whisper, samples, options, Some(|s| println!("{}", s))).unwrap();
        log::info!("Processing time: {:?}", transcript.processing_time);
    } else {
        panic!("Input file not found");
    };
}

fn handle_phi2(matches: &ArgMatches, api: Api) -> anyhow::Result<()> {
    let _ = env_logger::builder().is_test(true).try_init();
    let model_repo = api.model("FL33TW00D-HF/phi2".to_string());
    let model_path = model_repo.get("phi2-q8_0.gguf").unwrap();
    println!("MODEL PATH: {}", model_path.display());
    let mut reader = std::io::BufReader::new(std::fs::File::open(model_path)?);
    let device = Device::request_device(DeviceRequest::GPU)?;
    let content = Header::read(&mut reader)?;
    let mut model = Phi2::load(content, &mut reader, &device)?;

    let tokenizer =
        Tokenizer::from_file(concat!("../../", "/models/microsoft/phi-2/tokenizer.json")).unwrap();

    let prompt = if let Some(prompt) = matches.get_one::<String>("prompt") {
        prompt
    } else {
        "def print_prime(n):"
    };

    let max_tokens = matches.get_one::<usize>("max-tokens").unwrap();

    let encoding = tokenizer.encode(prompt, true).unwrap();
    let mut tokens = encoding
        .get_ids()
        .iter()
        .map(|&x| x as i32)
        .collect::<Vec<_>>();

    print!("{}", prompt);
    std::io::stdout().flush().unwrap();
    let mut all_tokens = tokens.clone();
    let mut loop_cnt = 0;
    let start_time = std::time::Instant::now();
    while tokens[tokens.len() - 1] != 50256 && loop_cnt < *max_tokens {
        let input = Tensor::from_data(tokens.clone(), shape![1, tokens.len()], device.clone());
        let result = model.schedule(input)?.resolve()?;
        let logits = result.to(&Device::CPU)?;
        model.cache_mut().update(tokens.len());

        tokens = logits
            .to_ndarray_view::<f32>()
            .map_axis(Axis(2), |row| row.argmax_skipnan().unwrap())
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        let u32_toks = tokens.iter().map(|&x| x as u32).collect::<Vec<_>>();
        print!("{}", tokenizer.decode(&u32_toks, true).unwrap());
        std::io::stdout().flush().unwrap();
        all_tokens.extend(tokens.clone());
        loop_cnt += 1;
    }
    let elapsed = start_time.elapsed();
    println!("\nElapsed time: {:?}", elapsed);
    println!(
        "tok/sec: {}",
        all_tokens.len() as f64 / elapsed.as_secs_f64()
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let matches = Command::new("ratchet")
        .about("LLM & VLM CLI")
        .version("0.1.0")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("whisper")
                .long_about(
                    "Cross-platform, GPU accelerated implementation of OpenAI's Whisper Model.",
                )
                .arg(
                    Arg::new("variant")
                        .short('v')
                        .long("variant")
                        .default_value("small")
                        .help("Whisper model variant to use.")
                        .value_parser(value_parser!(RegistryWhisper)),
                )
                .arg(
                    Arg::new("input")
                        .short('i')
                        .long("input")
                        .required(true)
                        .help("Path to the input file"),
                ),
        )
        .subcommand(
            Command::new("phi2")
                .long_about(
                    "Cross-platform, GPU accelerated implementation of Microsoft's Phi2 model.",
                )
                .arg(
                    Arg::new("prompt")
                        .short('p')
                        .long("prompt")
                        .required(true)
                        .help("Input prompt."),
                )
                .arg(
                    Arg::new("max-tokens")
                        .short('m')
                        .long("max-tokens")
                        .default_value("256")
                        .value_parser(value_parser!(usize))
                        .help("Maximum number of tokens to generate."),
                ),
        )
        .get_matches();

    let api = Api::new().unwrap();
    if let Some(matches) = matches.subcommand_matches("phi2") {
        handle_phi2(matches, api);
    } else if let Some(matches) = matches.subcommand_matches("whisper") {
        handle_whisper(matches, api);
    }

    Ok(())
}
