//! Shared scope validation for the legacy recall modes.

use vestige_core::DEFAULT_MEMORY_SCOPE;

pub(super) fn validate_scope(raw_scope: Option<&str>) -> Result<(), String> {
    let scope = raw_scope.unwrap_or(DEFAULT_MEMORY_SCOPE).trim();

    // Preserve the existing UTF-8 byte limit, despite the error saying "characters".
    if scope.is_empty() || scope.len() > 200 || scope.chars().any(char::is_control) {
        return Err(
            "Invalid scope: expected a non-empty identifier of at most 200 visible characters"
                .to_string(),
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const INVALID_SCOPE: &str =
        "Invalid scope: expected a non-empty identifier of at most 200 visible characters";

    #[test]
    fn omitted_scope_uses_the_default() {
        assert_eq!(validate_scope(None), Ok(()));
        assert_eq!(validate_scope(Some(DEFAULT_MEMORY_SCOPE)), Ok(()));
    }

    #[test]
    fn validation_trims_surrounding_whitespace() {
        for scope in [
            " project:Exact ",
            "\tproject:Exact\r\n",
            "\u{2003}project:Exact\u{2003}",
        ] {
            assert_eq!(validate_scope(Some(scope)), Ok(()), "{scope:?}");
        }
    }

    #[test]
    fn empty_and_whitespace_only_scopes_are_invalid() {
        for scope in ["", " ", "\t\r\n", "\u{2003}"] {
            assert_eq!(validate_scope(Some(scope)), Err(INVALID_SCOPE.into()));
        }
    }

    #[test]
    fn embedded_control_characters_are_invalid() {
        for scope in [
            "project\nname",
            "project\tname",
            "project\0name",
            "project\u{7f}name",
        ] {
            assert_eq!(validate_scope(Some(scope)), Err(INVALID_SCOPE.into()));
        }
    }

    #[test]
    fn the_byte_limit_is_checked_after_trimming() {
        let at_limit = "x".repeat(200);
        let over_limit = "x".repeat(201);
        let padded = format!("  {at_limit}\t");

        assert_eq!(validate_scope(Some(&at_limit)), Ok(()));
        assert_eq!(validate_scope(Some(&padded)), Ok(()));
        assert_eq!(validate_scope(Some(&over_limit)), Err(INVALID_SCOPE.into()));
    }

    #[test]
    fn multibyte_scopes_keep_the_existing_byte_limit() {
        let at_limit = "é".repeat(100);
        let over_limit = "é".repeat(101);

        assert_eq!(validate_scope(Some(&at_limit)), Ok(()));
        assert_eq!(validate_scope(Some(&over_limit)), Err(INVALID_SCOPE.into()));
    }

    #[tokio::test]
    async fn lookup_and_reason_validate_scope_before_backend_refusal() {
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let cognitive = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::cognitive::CognitiveEngine::new(),
        ));

        for mode in ["lookup", "reason"] {
            let error = super::super::recall::execute(
                &storage,
                &cognitive,
                &vestige_core::OutputConfig::default(),
                Some(serde_json::json!({
                    "mode": mode,
                    "query": "free text, not an exact handle",
                    "scope": " "
                })),
            )
            .await
            .unwrap_err();

            assert_eq!(error, INVALID_SCOPE, "{mode}");
        }
    }

    #[tokio::test]
    async fn contradictions_keeps_backend_refusal_before_scope_validation() {
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let error = super::super::contradictions::execute(
            &storage,
            Some(serde_json::json!({"scope": " "})),
        )
        .await
        .unwrap_err();

        assert!(error.starts_with("similarity_disabled:"), "{error}");
    }
}
