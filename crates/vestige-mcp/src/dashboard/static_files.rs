//! Embedded SvelteKit dashboard static file server.
//!
//! Uses `rust-embed` (release compression + debug hot-reload from disk +
//! built-in SHA-256 `ETag` hashing) to serve the SvelteKit SPA at `/dashboard/`.

use axum::extract::Path;
use axum::http::{StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use rust_embed::RustEmbed;

/// Embed the SvelteKit build output (`apps/dashboard/build/`).
/// In release builds assets are compressed into the binary; in debug builds
/// files are streamed directly from disk for instant UI iteration.
#[derive(RustEmbed)]
#[folder = "../../apps/dashboard/build"]
struct DashboardAssets;

fn format_etag(hash: &[u8; 32]) -> String {
    let mut out = String::with_capacity(18);
    out.push('"');
    for b in &hash[..8] {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{b:02x}");
    }
    out.push('"');
    out
}

/// Serve the SvelteKit dashboard index.
pub async fn serve_dashboard_spa() -> impl IntoResponse {
    match DashboardAssets::get("index.html") {
        Some(file) => Html(String::from_utf8_lossy(file.data.as_ref()).into_owned()).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            "Dashboard not built. Run: cd apps/dashboard && pnpm build",
        )
            .into_response(),
    }
}

/// Serve static assets from the embedded SvelteKit build.
pub async fn serve_dashboard_asset(Path(path): Path<String>) -> Response {
    let clean_path = path.trim_start_matches('/');
    if let Some(file) = DashboardAssets::get(clean_path) {
        let mime = mime_guess::from_path(clean_path)
            .first_or_octet_stream()
            .to_string();
        let etag = format_etag(&file.metadata.sha256_hash());
        let cache = if clean_path.contains("_app/") {
            "public, max-age=31536000, immutable"
        } else {
            "public, max-age=60"
        };

        return (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, mime),
                (header::CACHE_CONTROL, cache.to_string()),
                (header::ETAG, etag),
            ],
            file.data.into_owned(),
        )
            .into_response();
    }

    // SPA fallback: serve index.html for client-side routing.
    match DashboardAssets::get("index.html") {
        Some(file) => (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "text/html".to_string()),
                (header::ETAG, format_etag(&file.metadata.sha256_hash())),
            ],
            file.data.into_owned(),
        )
            .into_response(),
        None => (StatusCode::NOT_FOUND, "Not found").into_response(),
    }
}
