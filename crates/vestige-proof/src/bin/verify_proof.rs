//! `vestige-verify-proof`: the standalone offline verifier. It links only
//! `vestige-proof`, so it cannot open a Vestige store.

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(vestige_proof::cli::run(&argv));
}
