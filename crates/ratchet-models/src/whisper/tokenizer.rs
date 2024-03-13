use crate::{Language, Task};
use tokenizers::Tokenizer;

#[cfg(not(target_arch = "wasm32"))]
use hf_hub::api::sync::Api;
#[cfg(target_arch = "wasm32")]
use {ratchet_hub::ApiBuilder, ratchet_hub::RepoType, wasm_bindgen::JsError};

lazy_static::lazy_static! {
    pub static ref LANGUAGES: [&'static str; 100] = {
        [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar",
            "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu",
            "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa",
            "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn",
            "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
            "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
            "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw",
            "su", "yue",
        ]
    };
}

//Wrapper around tokenizers::Tokenizer with helpers
#[derive(Clone)]
pub struct WhisperTokenizer {
    inner: Tokenizer,
    language: i32,
    task: Task,
}

impl WhisperTokenizer {
    pub const SOT: i32 = 50258;
    pub const EOT: i32 = 50257;
    pub const LANGUAGES_BEGIN: usize = 50259;
    pub const BLANK: i32 = 220;

    //https://github.com/openai/whisper/blob/1cea4357687b676b293cb5473e1ade25f5b1cef7/whisper/tokenizer.py#L242
    pub const NON_SPEECH: [i32; 82] = [
        1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359,
        503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253,
        3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938,
        12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075,
        21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425,
        49870, 50254,
    ];

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_inner(bytes: Option<Vec<u8>>, v3: bool) -> Tokenizer {
        if let Some(bytes) = bytes {
            Tokenizer::from_bytes(bytes).unwrap()
        } else {
            Self::fetch(v3)
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(bytes: Option<Vec<u8>>, v3: bool, language: Language, task: Task) -> Self {
        let inner = Self::load_inner(bytes, v3);
        let mut tokenizer = Self {
            inner,
            language: -1,
            task,
        };
        tokenizer.set_language(language);
        tokenizer
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn fetch(v3: bool) -> Tokenizer {
        let api = Api::new().unwrap();

        //TODO: dumb hack
        let repo_name = match v3 {
            true => "openai/whisper-large-v3",
            false => "openai/whisper-tiny",
        };
        println!("LOADING TOKENIZER: {}", repo_name);

        let tokenizer_repo = api.model(repo_name.to_string());
        let tokenizer_path = tokenizer_repo.get("tokenizer.json").unwrap();
        Tokenizer::from_file(tokenizer_path).unwrap()
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn load_inner(bytes: Option<Vec<u8>>, v3: bool) -> Tokenizer {
        use wasm_bindgen::JsValue;

        if let Some(bytes) = bytes {
            Tokenizer::from_bytes(bytes).unwrap()
        } else {
            Self::fetch(v3).await.map_err(JsValue::from).unwrap()
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn load(bytes: Option<Vec<u8>>, v3: bool, language: Language, task: Task) -> Self {
        let inner = Self::load_inner(bytes, v3).await;
        let mut tokenizer = Self {
            inner,
            language: -1,
            task,
        };
        tokenizer.set_language(language);
        tokenizer
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn fetch(v3: bool) -> Result<Tokenizer, JsError> {
        let repo_name = match v3 {
            true => "openai/whisper-large-v3",
            false => "openai/whisper-tiny",
        };

        let model_repo = ApiBuilder::from_hf(repo_name, RepoType::Model).build();
        let model_bytes = model_repo.get("tokenizer.json").await?;
        Ok(Tokenizer::from_bytes(model_bytes.to_vec()).unwrap())
    }

    pub fn set_language(&mut self, language: Language) {
        let token = match language {
            Language::String(s) => {
                let lang_position = LANGUAGES.iter().position(|x| *x == s);
                if lang_position.is_none() {
                    panic!("Language {} not found", s);
                }

                WhisperTokenizer::SOT + 1 + lang_position.unwrap() as i32
            }
            Language::Token(t) => t,
        };
        self.language = token;
    }

    #[inline]
    pub fn sot_prev(&self) -> i32 {
        self.inner
            .encode("<|startofprev|>", false)
            .unwrap()
            .get_ids()[0] as i32
    }

    #[inline]
    pub fn sot_sequence(&self) -> Vec<i32> {
        vec![Self::SOT, self.language, self.task.as_token(self)]
    }

    #[inline]
    pub fn transcribe(&self) -> i32 {
        self.inner
            .encode("<|transcribe|>", false)
            .unwrap()
            .get_ids()[0] as i32
    }

    #[inline]
    pub fn translate(&self) -> i32 {
        self.inner.encode("<|translate|>", false).unwrap().get_ids()[0] as i32
    }

    #[inline]
    pub fn notimestamps(&self) -> i32 {
        self.inner
            .encode("<|notimestamps|>", false)
            .unwrap()
            .get_ids()[0] as i32
    }

    #[inline]
    pub fn timestamp_begin(&self) -> i32 {
        self.inner.encode("<|0.00|>", false).unwrap().get_ids()[0] as i32
    }

    #[inline]
    pub fn is_timestamp(&self, token: i32) -> bool {
        (self.timestamp_begin()..=self.vocab_size() as i32).contains(&token)
    }

    #[inline]
    pub fn is_multilingual(&self) -> bool {
        self.inner.get_vocab_size(true) >= 51865
    }

    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn encode(&self, text: &str, skip_special: bool) -> Result<Vec<i32>, tokenizers::Error> {
        Ok(self
            .inner
            .encode(text, skip_special)?
            .get_ids()
            .iter()
            .map(|x| *x as _)
            .collect())
    }

    pub fn decode(&self, tokens: &[u32], skip_special: bool) -> Result<String, tokenizers::Error> {
        self.inner.decode(tokens, skip_special)
    }
}
