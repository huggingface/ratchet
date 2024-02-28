use indexed_db_futures::prelude::*;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Debug, thiserror::Error)]
pub enum RatchetDBError {
    #[error("DomException {name} ({code}): {message}")]
    DomException {
        name: String,
        message: String,
        code: u16,
    },
    #[error(transparent)]
    SerdeError(#[from] serde_wasm_bindgen::Error),
}

impl From<indexed_db_futures::web_sys::DomException> for RatchetDBError {
    fn from(e: indexed_db_futures::web_sys::DomException) -> Self {
        Self::DomException {
            name: e.name(),
            message: e.message(),
            code: e.code(),
        }
    }
}

pub struct RatchetDB {
    name: String,
    pub(crate) inner: IdbDatabase,
}

type Result<A, E = RatchetDBError> = std::result::Result<A, E>;

impl RatchetDB {
    pub const DB_VERSION: u32 = 1;
    pub const MODEL_STORE: &'static str = "models";
    pub const TOKENIZER_STORE: &'static str = "tokenizers";

    fn serialize(o: &impl Serialize) -> Result<JsValue> {
        serde_wasm_bindgen::to_value(o).map_err(|e| e.into())
    }

    fn deserialize<T: for<'de> Deserialize<'de>>(o: Option<JsValue>) -> Result<Option<T>> {
        o.map(serde_wasm_bindgen::from_value)
            .transpose()
            .map_err(|e| e.into())
    }

    pub async fn open(name: &str) -> Result<Self, RatchetDBError> {
        let mut db_req: OpenDbRequest = IdbDatabase::open_u32(name, Self::DB_VERSION)?;

        db_req.set_on_upgrade_needed(Some(|evt: &IdbVersionChangeEvent| -> Result<(), JsValue> {
            let create_if_needed =
                |evt: &IdbVersionChangeEvent, store_key: &'static str| -> Result<(), JsValue> {
                    if let None = evt.db().object_store_names().find(|n| n == store_key) {
                        evt.db().create_object_store(store_key)?;
                    }
                    Ok(())
                };
            create_if_needed(evt, Self::MODEL_STORE)?;
            create_if_needed(evt, Self::TOKENIZER_STORE)?;
            Ok(())
        }));

        Ok(Self {
            name: name.to_string(),
            inner: db_req.await?,
        })
    }

    pub async fn get_model(&self, id: String) -> Result<Option<StoredModel>> {
        let tx = self
            .inner
            .transaction_on_one_with_mode(Self::MODEL_STORE, IdbTransactionMode::Readonly)?;
        let store = tx.object_store(Self::MODEL_STORE)?;
        let req = store.get(&Self::serialize(&id)?)?.await?;
        Self::deserialize(req)
    }

    pub async fn put_model(&self, id: String, model: StoredModel) -> Result<()> {
        let tx = self
            .inner
            .transaction_on_one_with_mode(Self::MODEL_STORE, IdbTransactionMode::Readwrite)?;
        let store = tx.object_store(Self::MODEL_STORE)?;
        store
            .put_key_val(&Self::serialize(&id)?, &Self::serialize(&model)?)?
            .await?;
        Ok(())
    }

    pub async fn get_tokenizer(&self, id: String) -> Result<Option<StoredTokenizer>> {
        let tx = self
            .inner
            .transaction_on_one_with_mode(Self::TOKENIZER_STORE, IdbTransactionMode::Readonly)?;
        let store = tx.object_store(Self::TOKENIZER_STORE)?;
        let req = store.get(&Self::serialize(&id)?)?.await?;
        Self::deserialize(req)
    }

    pub async fn put_tokenizer(&self, id: String, tokenizer: StoredTokenizer) -> Result<()> {
        let tx = self
            .inner
            .transaction_on_one_with_mode(Self::TOKENIZER_STORE, IdbTransactionMode::Readwrite)?;
        let store = tx.object_store(Self::TOKENIZER_STORE)?;
        store
            .put_key_val(&Self::serialize(&id)?, &Self::serialize(&tokenizer)?)?
            .await?;
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StoredModel {
    id: String, //"FL33TW00D/whisper-tiny"
    #[serde(with = "serde_bytes")]
    bytes: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StoredTokenizer {
    id: String, //"FL33TW00D/whisper-tiny"
    #[serde(with = "serde_bytes")]
    bytes: Vec<u8>,
}

#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    async fn test_db() {
        let db = RatchetDB::open("test_db").await.unwrap();
        db.put_model(
            "FL33TW00D/whisper-tiny".to_string(),
            StoredModel {
                id: "FL33TW00D/whisper-tiny".to_string(),
                bytes: vec![1, 2, 3, 4],
            },
        )
        .await
        .unwrap();
        let model = db
            .get_model("FL33TW00D/whisper-tiny".to_string())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(model.bytes, vec![1, 2, 3, 4]);
    }
}
