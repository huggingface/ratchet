use js_sys::{ArrayBuffer, Uint8Array, JSON};

use wasm_bindgen::{prelude::*, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

fn to_error(value: JsValue) -> JsError {
    JsError::new(
        JSON::stringify(&value)
            .map(|js_string| {
                js_string
                    .as_string()
                    .unwrap_or(String::from("An unknown error occurred."))
            })
            .unwrap_or(String::from("An unknown error occurred."))
            .as_str(),
    )
}
pub(crate) async fn fetch(url: &str) -> Result<Response, JsError> {
    let mut opts = RequestInit::new();
    opts.method("GET");
    opts.mode(RequestMode::Cors);

    let request = Request::new_with_str_and_init(&url, &opts).map_err(to_error)?;

    let window = web_sys::window().unwrap();
    let resp_value = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(to_error)?;

    assert!(resp_value.is_instance_of::<Response>());
    let resp: Response = resp_value.dyn_into().unwrap();

    Ok(resp)
}
