//! The same pure compiler used by native MCP. No daemon/socket, filesystem,
//! network, model or durable-log writer is included in this adapter.
#[path = "../../vestige-core/src/h2sk/mod.rs"]
pub mod h2sk;
use serde::Deserialize;
use wasm_bindgen::prelude::*;
#[derive(Deserialize)]
struct Compile {
    prefix: h2sk::LogPrefix,
    #[serde(default)]
    options: h2sk::CompileOptions,
}
#[derive(Deserialize)]
struct Admit {
    event: h2sk::LogEvent,
    #[serde(default)]
    prior: Vec<h2sk::LogEvent>,
    #[serde(default)]
    edges: Vec<h2sk::LogEdge>,
}
#[derive(Deserialize)]
struct Verify {
    receipt: h2sk::Receipt,
    options: h2sk::CompileOptions,
    input: h2sk::GateInput,
}
fn read<T: serde::de::DeserializeOwned>(input: &str) -> Result<T, JsValue> {
    if input.len() > 32 * 1024 * 1024 {
        return Err(JsValue::from_str("input_resource_limit"));
    }
    serde_json::from_str(input).map_err(|_| JsValue::from_str("invalid_h2sk_input"))
}
fn json<T: serde::Serialize>(value: &T) -> Result<String, JsValue> {
    serde_json::to_string(value).map_err(|_| JsValue::from_str("invalid_h2sk_output"))
}
#[wasm_bindgen]
pub fn compile_packet(input: &str) -> Result<String, JsValue> {
    let request: Compile = read(input)?;
    json(
        &h2sk::compile_log_prefix(&request.prefix, &request.options)
            .map_err(|e| JsValue::from_str(&e))?,
    )
}
#[wasm_bindgen]
pub fn admit_statement(input: &str) -> Result<String, JsValue> {
    let request: Admit = read(input)?;
    let (event, edges) =
        h2sk::admit::admit_statement(request.event, &request.prior, &request.edges);
    json(&serde_json::json!({"event":event,"edges":edges}))
}
#[wasm_bindgen]
pub fn verify_reply(input: &str) -> Result<String, JsValue> {
    let request: Verify = read(input)?;
    json(&h2sk::verify_draft(
        &request.receipt,
        &request.options,
        &request.input,
    ))
}
#[wasm_bindgen]
pub fn reader_context(input: &str) -> Result<String, JsValue> {
    let request: Verify = read(input)?;
    let ids = request
        .options
        .view_ids
        .unwrap_or_else(|| {
            request
                .receipt
                .section
                .iter()
                .map(|s| s.id.clone())
                .collect()
        })
        .into_iter()
        .collect();
    Ok(h2sk::compile::prepare_payload(&request.receipt, &ids, 7000))
}
