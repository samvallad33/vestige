//! Link C23 strto* shims when the prebuilt ONNX Runtime archive is used
//! on linux-gnu. See `compat/isoc23_shim.c`.

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let ort_download = std::env::var("CARGO_FEATURE_ORT_DOWNLOAD").is_ok();

    println!("cargo:rerun-if-changed=compat/isoc23_shim.c");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_ORT_DOWNLOAD");

    if target_os == "linux" && target_env == "gnu" && ort_download {
        cc::Build::new()
            .file("compat/isoc23_shim.c")
            .warnings(true)
            .compile("vestige_isoc23_shim");
    }
}
