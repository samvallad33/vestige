//! MCP App receipt card (`ui://vestige/receipt/{id}`) — SEP-1865.
//!
//! Renders ONE persisted retrieval receipt as a self-contained HTML document
//! that hosts with MCP Apps support (Claude Desktop, Cursor) fetch through
//! `resources/read` and show inline next to the conversation. The URI scheme
//! is `ui://`, which is what tells a host "this resource is an app view", and
//! the MIME type is the SEP-1865 app profile of HTML.
//!
//! Hard boundaries, per #241:
//! * NO network access — no external fonts, scripts, images, or fetches. A
//!   document-level CSP meta tag denies everything but the inline stylesheet,
//!   so even a rendering bug cannot phone home.
//! * NO memory content beyond what the receipt already carries — ids, trust
//!   numbers, suppression reasons, the activation path. The card is a view of
//!   the receipt, not a new retrieval surface.
//! * Server-side rendered — plain HTML + CSS, zero JavaScript. The card needs
//!   no postMessage conversation with the host, so it works in any app-capable
//!   host and degrades to a static document everywhere else.

use std::sync::Arc;

use vestige_core::{Receipt, Storage};

/// The SEP-1865 MIME type for MCP App resources. The `profile` parameter is
/// what distinguishes an app document from a plain web page.
pub const MIME_TYPE: &str = "text/html;profile=mcp-app";

/// URI template advertised through `resources/templates/list`.
pub const URI_TEMPLATE: &str = "ui://vestige/receipt/{id}";

/// Prefix of every concrete receipt-card URI.
const URI_PREFIX: &str = "ui://vestige/receipt/";

/// The stable `ui://vestige/receipt/{id}` URI for one receipt.
pub fn resource_uri(receipt_id: &str) -> String {
    format!("{URI_PREFIX}{receipt_id}")
}

/// If `uri` names a receipt card, return the receipt id it targets.
///
/// Rejects traversal (`..`, `/`) and empty ids so a hand-crafted URI can only
/// ever name a receipt id lookup, never another resource path.
pub fn parse_uri(uri: &str) -> Option<&str> {
    let id = uri.strip_prefix(URI_PREFIX)?;
    if id.is_empty() || id.contains('/') || id.contains("..") {
        return None;
    }
    Some(id)
}

/// Read one receipt card as an HTML string.
///
/// Errors read like resource-lookup failures (matched by the caller into
/// "not found") or internal failures; success is the document text.
pub async fn read(storage: &Arc<Storage>, uri: &str) -> Result<String, String> {
    let id = parse_uri(uri).ok_or_else(|| format!("Unknown resource scheme: {uri}"))?;
    let receipt = storage
        .get_receipt(id)
        .map_err(|error| format!("receipt lookup failed: {error}"))?
        .ok_or_else(|| format!("Resource not found: {uri}"))?;
    Ok(render_html(&receipt))
}

/// Escape a string for safe interpolation into HTML text and attribute
/// positions. The receipt fields rendered here come from our own database,
/// but ids and reasons pass through user-influenced content (memory ids are
/// derived from agent-supplied strings), so they are escaped like untrusted
/// input — because at render time, in someone else's iframe, they are.
fn escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Render the receipt as the complete HTML document.
fn render_html(receipt: &Receipt) -> String {
    let trust_pct = (receipt.trust_floor.clamp(0.0, 1.0) * 100.0).round() as u64;
    let trust_floor = format!("{:.2}", receipt.trust_floor);
    let (risk_label, risk_class) = match receipt.decay_risk {
        vestige_core::DecayRisk::Low => ("low", "ok"),
        vestige_core::DecayRisk::Medium => ("medium", "mid"),
        vestige_core::DecayRisk::High => ("high", "bad"),
    };

    let mut body = String::with_capacity(4096);

    // Retrieved, best-first — the ids that actually informed the answer.
    if receipt.retrieved.is_empty() {
        body.push_str("<p class=\"empty\">No memories informed this answer.</p>");
    } else {
        body.push_str("<h2>Retrieved");
        body.push_str(&format!(" <span class=\"count\">{}</span>", receipt.retrieved.len()));
        body.push_str("</h2><ol class=\"ids\">");
        for id in &receipt.retrieved {
            body.push_str(&format!("<li><code>{}</code></li>", escape(id)));
        }
        body.push_str("</ol>");
    }

    // Suppressed — what the agent chose NOT to use, and why.
    body.push_str("<h2>Suppressed");
    body.push_str(&format!(
        " <span class=\"count\">{}</span>",
        receipt.suppressed.len()
    ));
    body.push_str("</h2>");
    if receipt.suppressed.is_empty() {
        body.push_str("<p class=\"empty\">Nothing was withheld.</p>");
    } else {
        body.push_str("<ul class=\"suppressed\">");
        for entry in &receipt.suppressed {
            body.push_str(&format!(
                "<li><code>{}</code><span class=\"reason\">{}</span></li>",
                escape(&entry.id),
                escape(entry.reason.as_str()),
            ));
        }
        body.push_str("</ul>");
    }

    // Activation path — how spreading activation surfaced the set.
    body.push_str("<h2>Activation path</h2>");
    if receipt.activation_path.is_empty() {
        body.push_str("<p class=\"empty\">No multi-hop path (direct retrieval).</p>");
    } else {
        body.push_str("<ol class=\"path\">");
        for hop in &receipt.activation_path {
            body.push_str(&format!("<li><code>{}</code></li>", escape(hop)));
        }
        body.push_str("</ol>");
    }

    // Mutations the retrieval triggered, if any.
    if !receipt.mutations.is_empty() {
        body.push_str("<h2>Mutations</h2><ul class=\"mutations\">");
        for mutation in &receipt.mutations {
            body.push_str(&format!(
                "<li><code>{}</code> {}</li>",
                escape(&mutation.id),
                escape(&mutation.kind),
            ));
        }
        body.push_str("</ul>");
    }

    format!(
        concat!(
            "<!DOCTYPE html>\n",
            "<html lang=\"en\">\n<head>\n",
            "<meta charset=\"utf-8\">\n",
            // App-sandbox hardening: nothing loads from the network, nothing
            // executes. Mirrors the no-network boundary of #241 at the
            // document level as well as the host's iframe sandbox.
            "<meta http-equiv=\"Content-Security-Policy\" ",
            "content=\"default-src 'none'; style-src 'unsafe-inline'\">\n",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n",
            "<title>Retrieval receipt {receipt_id}</title>\n",
            "<style>\n",
            ":root {{ color-scheme: light dark; }}\n",
            "* {{ box-sizing: border-box; }}\n",
            "body {{ font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;",
            " margin: 0; padding: 16px; font-size: 14px; line-height: 1.45;",
            " background: Canvas; color: CanvasText; }}\n",
            "header {{ display: flex; flex-wrap: wrap; align-items: baseline;",
            " gap: 8px; border-bottom: 1px solid color-mix(in srgb, CanvasText 15%, transparent);",
            " padding-bottom: 10px; margin-bottom: 14px; }}\n",
            "h1 {{ font-size: 15px; margin: 0; }}\n",
            "code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;",
            " font-size: 12px; background: color-mix(in srgb, CanvasText 8%, transparent);",
            " padding: 1px 5px; border-radius: 4px; }}\n",
            "h2 {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em;",
            " margin: 16px 0 6px; opacity: 0.75; }}\n",
            ".count {{ opacity: 0.6; text-transform: none; letter-spacing: 0; }}\n",
            ".badge {{ font-size: 11px; padding: 2px 8px; border-radius: 999px;",
            " border: 1px solid; }}\n",
            ".badge.ok {{ color: #1a7f37; border-color: #1a7f37; }}\n",
            ".badge.mid {{ color: #9a6700; border-color: #9a6700; }}\n",
            ".badge.bad {{ color: #cf222e; border-color: #cf222e; }}\n",
            "@media (prefers-color-scheme: dark) {{\n",
            " .badge.ok {{ color: #3fb950; border-color: #3fb950; }}\n",
            " .badge.mid {{ color: #d29922; border-color: #d29922; }}\n",
            " .badge.bad {{ color: #f85149; border-color: #f85149; }}\n",
            "}}\n",
            ".trustbar {{ height: 6px; border-radius: 3px; margin-top: 8px;",
            " background: color-mix(in srgb, CanvasText 12%, transparent); overflow: hidden; }}\n",
            ".trustbar > div {{ height: 100%; border-radius: 3px; }}\n",
            ".trustbar.ok > div {{ background: #1a7f37; }}\n",
            ".trustbar.mid > div {{ background: #9a6700; }}\n",
            ".trustbar.bad > div {{ background: #cf222e; }}\n",
            "ol, ul {{ margin: 4px 0; padding-left: 22px; }}\n",
            "li {{ margin: 3px 0; }}\n",
            ".reason {{ opacity: 0.7; margin-left: 8px; font-size: 12px; }}\n",
            ".empty {{ opacity: 0.6; font-style: italic; margin: 4px 0; }}\n",
            "footer {{ margin-top: 18px; padding-top: 10px;",
            " border-top: 1px solid color-mix(in srgb, CanvasText 15%, transparent);",
            " font-size: 11px; opacity: 0.65; }}\n",
            "</style>\n</head>\n<body>\n",
            "<header><h1>Retrieval receipt</h1>",
            "<code>{receipt_id}</code>",
            "<span class=\"badge {risk_class}\">decay risk: {risk_label}</span></header>\n",
            "<h2>Trust floor</h2>",
            "<div><strong>{trust_floor}</strong> <span class=\"count\">/ 1.00 \
             — the weakest link the answer rests on</span></div>",
            "<div class=\"trustbar {risk_class}\"><div style=\"width: {trust_pct}%\"></div></div>\n",
            "{body}\n",
            "<footer>Rendered server-side by vestige {version} from the persisted receipt. \
             No network access; no memory content beyond the receipt itself.</footer>\n",
            "</body>\n</html>\n",
        ),
        receipt_id = escape(&receipt.receipt_id),
        risk_label = risk_label,
        risk_class = risk_class,
        trust_floor = trust_floor,
        trust_pct = trust_pct,
        version = env!("CARGO_PKG_VERSION"),
        body = body,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vestige_core::{DecayRisk, Receipt, SuppressedReceiptEntry};
    use vestige_core::trace::SuppressReason;

    fn sample_receipt() -> Receipt {
        let mut receipt = Receipt::build_with_unique(
            chrono::Utc::now(),
            "testrun1",
            "abcdef",
            vec!["mem-alpha".to_string(), "mem-beta".to_string()],
            vec![SuppressedReceiptEntry::new(
                "mem-gamma",
                SuppressReason::LowTrust,
            )],
            vec!["mem-alpha -> mem-beta".to_string()],
            &[0.92, 0.64],
            Vec::new(),
        );
        receipt.decay_risk = DecayRisk::Medium;
        receipt
    }

    #[test]
    fn uri_round_trip() {
        let receipt = sample_receipt();
        let uri = resource_uri(&receipt.receipt_id);
        assert_eq!(parse_uri(&uri), Some(receipt.receipt_id.as_str()));
        assert!(uri.starts_with("ui://vestige/receipt/"));
    }

    #[test]
    fn parse_uri_rejects_traversal_and_foreign_schemes() {
        assert!(parse_uri("ui://vestige/receipt/").is_none());
        assert!(parse_uri("ui://vestige/receipt/a/../b").is_none());
        assert!(parse_uri("ui://vestige/receipt/a/b").is_none());
        assert!(parse_uri("memory://stats").is_none());
        assert!(parse_uri("ui://other/receipt/x").is_none());
    }

    #[test]
    fn html_carries_receipt_facts_and_nothing_executable() {
        let receipt = sample_receipt();
        let html = render_html(&receipt);
        assert!(html.contains(&escape(&receipt.receipt_id)));
        assert!(html.contains("mem-alpha"));
        assert!(html.contains("mem-beta"));
        assert!(html.contains("low_trust"));
        assert!(html.contains("mem-alpha -&gt; mem-beta"));
        assert!(html.contains("0.64"));
        // No-network boundary, enforced in the document itself.
        assert!(html.contains("default-src 'none'"));
        assert!(!html.to_ascii_lowercase().contains("<script"));
        assert!(!html.to_ascii_lowercase().contains("http://"));
        assert!(!html.to_ascii_lowercase().contains("https://"));
    }

    #[test]
    fn html_escapes_hostile_ids() {
        let mut receipt = sample_receipt();
        receipt.retrieved = vec!["<script>alert(1)</script>".to_string()];
        let html = render_html(&receipt);
        assert!(!html.contains("<script>alert"));
        assert!(html.contains("&lt;script&gt;"));
    }
}
