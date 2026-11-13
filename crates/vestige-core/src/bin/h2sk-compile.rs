//! Structured stdin boundary shared by admission adapters and local web builds.
//! The live MCP packet calls the Rust compiler directly, never this sidecar.
use serde::Deserialize;
use std::io::{Read, Write};
use vestige_core::h2sk::{self, CompileOptions, GateInput, LogEvent, LogPrefix};
#[derive(Deserialize)]
struct Request {
    #[serde(default)]
    prefix: LogPrefix,
    #[serde(default)]
    options: CompileOptions,
    #[serde(default)]
    operation: String,
    #[serde(default)]
    statements: Vec<LogEvent>,
    #[serde(default)]
    draft: Option<GateInput>,
}
fn run(request: Request) -> Result<serde_json::Value, String> {
    if request.operation == "admit" {
        let mut prefix = request.prefix;
        for event in request.statements {
            let (event, edges) = h2sk::admit::admit_statement(event, &prefix.events, &prefix.edges);
            prefix.offset = prefix.offset.max(event.seq);
            prefix.events.push(event);
            prefix.edges.extend(edges);
        }
        return serde_json::to_value(prefix).map_err(|e| e.to_string());
    }
    if !request.operation.is_empty() && request.operation != "compile" {
        return Err("unknown_operation".into());
    }
    let receipt = h2sk::compile_log_prefix(&request.prefix, &request.options)?;
    if let Some(draft) = request.draft {
        return serde_json::to_value(h2sk::verify_draft(&receipt, &request.options, &draft))
            .map_err(|e| e.to_string());
    }
    serde_json::to_value(receipt).map_err(|e| e.to_string())
}
fn main() {
    let mut bytes = Vec::new();
    let result = (|| {
        std::io::stdin()
            .take(32 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        if bytes.len() > 32 * 1024 * 1024 {
            return Err("input_resource_limit".into());
        }
        run(serde_json::from_slice(&bytes).map_err(|e| format!("invalid_input:{e}"))?)
    })();
    match result {
        Ok(value) => {
            serde_json::to_writer(std::io::stdout(), &value).unwrap();
            std::io::stdout().write_all(b"\n").unwrap();
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
