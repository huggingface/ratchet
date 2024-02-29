#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg_attr(target_arch = "wasm32", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub enum Language {
    String(String),
    Token(i32),
}

#[cfg_attr(target_arch = "wasm32", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub enum Prompt {
    Text(String),
    Tokens(Vec<i32>),
}

#[cfg_attr(
    target_arch = "wasm32",
    wasm_bindgen,
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy)]
pub enum Task {
    Transcribe,
    Translate,
}

impl From<Task> for i32 {
    fn from(val: Task) -> Self {
        match val {
            Task::Transcribe => 50359,
            Task::Translate => 50358,
        }
    }
}

#[allow(dead_code)]
#[cfg_attr(
    target_arch = "wasm32",
    wasm_bindgen,
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
pub struct DecodingOptions {
    pub(crate) task: Task,                         // default: "transcribe"
    pub(crate) language: Option<Language>,         // default: None
    pub(crate) temperature: f32,                   // default: 0.0
    pub(crate) sample_len: Option<u32>,            // default: None
    pub(crate) best_of: Option<u32>,               // default: None
    pub(crate) beam_size: Option<u32>,             // default: None
    pub(crate) patience: Option<f32>,              // default: None
    pub(crate) length_penalty: Option<f32>,        // default: None
    pub(crate) prompt: Option<Prompt>,             // default: None
    pub(crate) prefix: Option<String>,             // default: None
    pub(crate) suppress_tokens: Option<Vec<i32>>,  // default: Some("-1".to_string())
    pub(crate) suppress_blank: bool,               // default: true
    pub(crate) without_timestamps: bool,           // default: false
    pub(crate) max_initial_timestamp: Option<f32>, // default: Some(1.0)
    pub(crate) time_offset: Option<f64>,           // default: None
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct DecodingOptionsBuilder {
    task: Option<Task>,
    language: Option<String>,
    temperature: Option<f32>,
    sample_len: Option<u32>,
    best_of: Option<u32>,
    beam_size: Option<u32>,
    patience: Option<f32>,
    length_penalty: Option<f32>,
    prompt: Option<String>,
    prefix: Option<String>,
    suppress_tokens: Option<Vec<i32>>,
    suppress_blank: Option<bool>,
    without_timestamps: Option<bool>,
    max_initial_timestamp: Option<f32>,
    time_offset: Option<f64>,
}

impl Default for DecodingOptionsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl DecodingOptionsBuilder {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new() -> DecodingOptionsBuilder {
        DecodingOptionsBuilder {
            task: Some(Task::Transcribe),
            language: None,
            temperature: Some(0.0),
            sample_len: None,
            best_of: None,
            beam_size: None,
            patience: None,
            length_penalty: None,
            prompt: None,
            prefix: None,
            suppress_tokens: Some(vec![-1]),
            suppress_blank: Some(true),
            max_initial_timestamp: Some(1.0),
            without_timestamps: Some(false),
            time_offset: None,
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = "setTask"))]
    pub fn task(mut self, task: Task) -> Self {
        self.task = Some(task);
        self
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = "setLanguage"))]
    pub fn language(mut self, language: String) -> Self {
        self.language = Some(language);
        self
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = "setTemperature"))]
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = "setSampleLen"))]
    pub fn sample_len(mut self, sample_len: u32) -> Self {
        self.sample_len = Some(sample_len);
        self
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = "setBestOf"))]
    pub fn best_of(mut self, best_of: u32) -> Self {
        self.best_of = Some(best_of);
        self
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = "setBeamSize"))]
    pub fn beam_size(mut self, beam_size: u32) -> Self {
        self.beam_size = Some(beam_size);
        self
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = "setPatience"))]
    pub fn patience(mut self, patience: f32) -> Self {
        self.patience = Some(patience);
        self
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = "setLengthPenalty"))]
    pub fn length_penalty(mut self, length_penalty: f32) -> Self {
        self.length_penalty = Some(length_penalty);
        self
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = "setPrompt"))]
    pub fn prompt(mut self, prompt: String) -> Self {
        self.prompt = Some(prompt);
        self
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = "setPrefix"))]
    pub fn prefix(mut self, prefix: String) -> Self {
        self.prefix = Some(prefix);
        self
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = "setSuppressTokens"))]
    pub fn suppress_tokens(mut self, suppress_tokens: Vec<i32>) -> Self {
        self.suppress_tokens = Some(suppress_tokens);
        self
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = "setSuppressBlank"))]
    pub fn suppress_blank(mut self, suppress_blank: bool) -> Self {
        self.suppress_blank = Some(suppress_blank);
        self
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = "setWithoutTimestamps"))]
    pub fn without_timestamps(mut self, without_timestamps: bool) -> Self {
        self.without_timestamps = Some(without_timestamps);
        self
    }

    #[cfg_attr(
        target_arch = "wasm32",
        wasm_bindgen(js_name = "setMaxInitialTimestamp")
    )]
    pub fn max_initial_timestamp(mut self, max_initial_timestamp: f32) -> Self {
        self.max_initial_timestamp = Some(max_initial_timestamp);
        self
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = "setTimeOffset"))]
    pub fn time_offset(mut self, time_offset: f64) -> Self {
        self.time_offset = Some(time_offset);
        self
    }

    pub fn build(&self) -> DecodingOptions {
        DecodingOptions {
            task: self.task.unwrap_or(Task::Transcribe),
            language: self.language.clone().map(Language::String),
            temperature: self.temperature.unwrap_or(0.0),
            sample_len: self.sample_len,
            best_of: self.best_of,
            beam_size: self.beam_size,
            patience: self.patience,
            length_penalty: self.length_penalty,
            prompt: self.prompt.clone().map(Prompt::Text),
            prefix: self.prefix.clone(),
            suppress_tokens: self.suppress_tokens.clone(),
            suppress_blank: self.suppress_blank.unwrap_or(true),
            without_timestamps: self.without_timestamps.unwrap_or(false),
            max_initial_timestamp: self.max_initial_timestamp,
            time_offset: self.time_offset,
        }
    }
}

cfg_if::cfg_if! {
    if #[cfg(all(not(target_arch = "wasm32"), test))] {
        use pyo3::types::{IntoPyDict, PyDict};
        use pyo3::Python;
        use pyo3::types::PyString;
        use pyo3::IntoPy;
        use pyo3::PyObject;

        impl IntoPy<PyObject> for Task {
            fn into_py(self, py: Python) -> PyObject {
                let task = match self {
                    Task::Transcribe => "transcribe",
                    Task::Translate => "translate",
                };
                PyString::new(py, task).into()
            }
        }

        impl IntoPy<PyObject> for Language {
            fn into_py(self, py: Python) -> PyObject {
                let language = match self {
                    Language::String(s) => s,
                    Language::Token(t) => t.to_string(),
                };
                PyString::new(py, &language).into()
            }
        }

        impl IntoPy<PyObject> for Prompt {
            fn into_py(self, py: Python) -> PyObject {
                match self {
                    Prompt::Text(s) => PyString::new(py, &s).into(),
                    Prompt::Tokens(t) => t.into_py(py),
                }
            }
        }

        impl IntoPyDict for DecodingOptions {
            fn into_py_dict(self, py: Python) -> &pyo3::types::PyDict {
                let dict = PyDict::new(py);
                let supress_tokens_string = self.suppress_tokens.map(|v| v.iter().map(|t| t.to_string()).collect::<Vec<String>>().join(","));

                let _ = dict.set_item("task", self.task.into_py(py));
                let _ = dict.set_item("language", self.language.map_or_else(|| py.None(), |v| v.into_py(py)));
                let _ = dict.set_item("temperature", self.temperature.into_py(py));
                let _ = dict.set_item("sample_len", self.sample_len.map_or_else(|| py.None(), |v| v.into_py(py)));
                let _ = dict.set_item("best_of", self.best_of.map_or_else(|| py.None(), |v| v.into_py(py)));
                let _ = dict.set_item("beam_size", self.beam_size.map_or_else(|| py.None(), |v| v.into_py(py)));
                let _ = dict.set_item("patience", self.patience.map_or_else(|| py.None(), |v| v.into_py(py)));
                let _ = dict.set_item("length_penalty", self.length_penalty.map_or_else(|| py.None(), |v| v.into_py(py)));
                let _ = dict.set_item("prompt", self.prompt.map_or_else(|| py.None(), |v| v.into_py(py)));
                let _ = dict.set_item("prefix", self.prefix.map_or_else(|| py.None(), |v| v.into_py(py)));
                let _ = dict.set_item("suppress_tokens", supress_tokens_string.map_or_else(|| py.None(), |v| v.into_py(py)));
                let _ = dict.set_item("suppress_blank", self.suppress_blank.into_py(py));
                let _ = dict.set_item("without_timestamps", self.without_timestamps.into_py(py));
                let _ = dict.set_item("max_initial_timestamp", self.max_initial_timestamp.map_or_else(|| py.None(), |v| v.into_py(py)));

                dict
            }
        }
    }
}
