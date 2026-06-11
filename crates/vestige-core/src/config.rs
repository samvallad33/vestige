//! Vestige configuration file (`vestige.toml`).
//!
//! Phase 2 "Configurable Output" of the adoption roadmap. A small, optional
//! config file lives alongside the SQLite database in the active Vestige data
//! directory (`<data_dir>/vestige.toml`). It lets users tune the default shape
//! of high-traffic MCP responses (detail level, result limit, output profile)
//! without recompiling and without a cloud service.
//!
//! Precedence, from highest to lowest:
//!
//! 1. An explicit MCP call parameter (e.g. `detail_level` on a `search` call).
//! 2. The config file `[defaults]` (and the selected output profile).
//! 3. The built-in default, which preserves the historical behavior so nothing
//!    changes for users who never write a `vestige.toml`.
//!
//! The parser is intentionally a tiny, dependency-free subset of TOML: section
//! headers (`[defaults]`) and `key = value` lines with string or integer
//! values. This keeps the local-first binary lean and avoids pulling a full
//! TOML crate into the dependency tree for a three-key schema. Unknown keys and
//! unknown sections are ignored so the file can grow in future phases without
//! breaking older binaries.

use std::path::{Path, PathBuf};

/// Canonical config file name, resolved inside the active data directory.
pub const CONFIG_FILE: &str = "vestige.toml";

/// Output profiles preset a coherent bundle of detail/field choices.
///
/// `Default` MUST reproduce the pre-Phase-2 behavior exactly so existing users
/// see no change. The other profiles are opt-in presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputProfile {
    /// Smallest responses: brief detail, scores and timestamps suppressed.
    /// Use when context budget matters more than provenance.
    Lean,
    /// Historical behavior. `summary` detail with content + dates. Unchanged.
    #[default]
    Default,
    /// Maximum provenance: `full` detail with every field, score, and timestamp.
    /// Use when reviewing or debugging memory state.
    Audit,
    /// Like `audit` but tuned for larger result sets (higher default limit).
    Research,
}

impl OutputProfile {
    /// Parse a profile name. Returns `None` for unknown names so the caller can
    /// decide whether that is an error (MCP param) or ignorable (config file).
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "lean" => Some(Self::Lean),
            "default" => Some(Self::Default),
            "audit" => Some(Self::Audit),
            "research" => Some(Self::Research),
            _ => None,
        }
    }

    /// Canonical lowercase name, suitable for echoing back in responses.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lean => "lean",
            Self::Default => "default",
            Self::Audit => "audit",
            Self::Research => "research",
        }
    }

    /// The detail level this profile presets when the user has not set one
    /// explicitly via an MCP param or `[defaults] detail_level`.
    pub fn detail_level(self) -> &'static str {
        match self {
            Self::Lean => "brief",
            Self::Default => "summary",
            Self::Audit | Self::Research => "full",
        }
    }

    /// The result limit this profile presets when the user has not set one
    /// explicitly. `None` means "use the tool's own historical default", which
    /// keeps `default` fully backward-compatible.
    pub fn limit(self) -> Option<i32> {
        match self {
            Self::Lean => Some(5),
            Self::Default => None,
            Self::Audit => None,
            Self::Research => Some(25),
        }
    }

    /// Whether scores (combined/keyword/semantic) should be shown by default.
    /// Lean drops them to save tokens; the rest keep whatever the detail level
    /// already includes.
    pub fn show_scores(self) -> bool {
        !matches!(self, Self::Lean)
    }

    /// Whether timestamps should be shown by default. Lean drops them.
    pub fn show_timestamps(self) -> bool {
        !matches!(self, Self::Lean)
    }
}

/// The `[defaults]` table from `vestige.toml`. All fields optional.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OutputDefaults {
    /// Default detail level (`brief` | `summary` | `full`). Overrides the
    /// profile's preset detail level when set.
    pub detail_level: Option<String>,
    /// Default result limit for high-traffic tools. Overrides the profile's
    /// preset limit when set.
    pub limit: Option<i32>,
    /// Selected output profile. Defaults to `default` (historical behavior).
    pub profile: OutputProfile,
}

/// Parsed `vestige.toml`. Currently only the `[defaults]` table is meaningful;
/// the struct exists so future phases can add tables without churn.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VestigeConfig {
    pub defaults: OutputDefaults,
}

impl VestigeConfig {
    /// Resolve the config path for a given data directory.
    pub fn path_for_data_dir(data_dir: &Path) -> PathBuf {
        data_dir.join(CONFIG_FILE)
    }

    /// Load config from a data directory. A missing or unreadable file yields
    /// the built-in default (never an error) so a fresh install just works.
    /// A present-but-malformed file is parsed leniently: only well-formed lines
    /// are honored.
    pub fn load_from_data_dir(data_dir: &Path) -> Self {
        let path = Self::path_for_data_dir(data_dir);
        match std::fs::read_to_string(&path) {
            Ok(contents) => Self::parse(&contents),
            Err(_) => Self::default(),
        }
    }

    /// Parse the minimal TOML subset. Lenient by design.
    pub fn parse(contents: &str) -> Self {
        let mut config = Self::default();
        let mut section = String::new();

        for raw in contents.lines() {
            let line = strip_comment(raw).trim();
            if line.is_empty() {
                continue;
            }

            if let Some(name) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                section = name.trim().to_ascii_lowercase();
                continue;
            }

            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let key = key.trim().to_ascii_lowercase();
            let value = unquote(value.trim());

            if section == "defaults" {
                match key.as_str() {
                    "detail_level" => {
                        let v = value.trim().to_ascii_lowercase();
                        if matches!(v.as_str(), "brief" | "summary" | "full") {
                            config.defaults.detail_level = Some(v);
                        }
                    }
                    "limit" => {
                        if let Ok(n) = value.trim().parse::<i32>()
                            && n > 0
                        {
                            config.defaults.limit = Some(n);
                        }
                    }
                    "profile" => {
                        if let Some(p) = OutputProfile::from_name(&value) {
                            config.defaults.profile = p;
                        }
                    }
                    _ => {}
                }
            }
        }

        config
    }

    /// Effective output config after applying the profile, with `[defaults]`
    /// detail_level / limit overriding the profile presets.
    pub fn output(&self) -> OutputConfig {
        let profile = self.defaults.profile;
        OutputConfig {
            profile,
            detail_level: self
                .defaults
                .detail_level
                .clone()
                .unwrap_or_else(|| profile.detail_level().to_string()),
            limit: self.defaults.limit.or_else(|| profile.limit()),
            show_scores: profile.show_scores(),
            show_timestamps: profile.show_timestamps(),
        }
    }
}

/// The resolved, ready-to-apply output configuration handed to MCP tools.
///
/// Tools treat each field as the *fallback* used only when the corresponding
/// explicit MCP call parameter is absent, preserving the precedence
/// `MCP param > config file > built-in default`.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputConfig {
    pub profile: OutputProfile,
    pub detail_level: String,
    pub limit: Option<i32>,
    pub show_scores: bool,
    pub show_timestamps: bool,
}

impl Default for OutputConfig {
    /// The built-in default == the historical behavior == the `default` profile.
    fn default() -> Self {
        VestigeConfig::default().output()
    }
}

impl OutputConfig {
    /// Resolve the detail level to use, given an optional explicit MCP param.
    /// Explicit param always wins (precedence layer 1).
    pub fn resolve_detail_level(&self, explicit: Option<&str>) -> String {
        explicit
            .map(|s| s.to_string())
            .unwrap_or_else(|| self.detail_level.clone())
    }

    /// Resolve the limit to use, given an optional explicit MCP param and the
    /// tool's own built-in fallback (used only when neither param nor config
    /// supplies one).
    pub fn resolve_limit(&self, explicit: Option<i32>, builtin_default: i32) -> i32 {
        explicit
            .or(self.limit)
            .unwrap_or(builtin_default)
    }
}

/// Strip a `#` comment that is not inside a quoted string.
fn strip_comment(line: &str) -> &str {
    let mut in_quotes = false;
    for (idx, ch) in line.char_indices() {
        match ch {
            '"' => in_quotes = !in_quotes,
            '#' if !in_quotes => return &line[..idx],
            _ => {}
        }
    }
    line
}

/// Remove a single layer of matching surrounding double quotes, if present.
fn unquote(value: &str) -> String {
    let bytes = value.as_bytes();
    if bytes.len() >= 2 && bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"' {
        value[1..value.len() - 1].to_string()
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_preserves_historical_behavior() {
        let out = OutputConfig::default();
        assert_eq!(out.profile, OutputProfile::Default);
        assert_eq!(out.detail_level, "summary");
        assert_eq!(out.limit, None);
        assert!(out.show_scores);
        assert!(out.show_timestamps);
    }

    #[test]
    fn empty_or_missing_file_is_default() {
        assert_eq!(VestigeConfig::parse(""), VestigeConfig::default());
        assert_eq!(VestigeConfig::parse("\n\n# just a comment\n"), VestigeConfig::default());
    }

    #[test]
    fn parses_defaults_table() {
        let cfg = VestigeConfig::parse(
            r#"
            [defaults]
            detail_level = "full"
            limit = 25
            profile = "research"
            "#,
        );
        assert_eq!(cfg.defaults.detail_level.as_deref(), Some("full"));
        assert_eq!(cfg.defaults.limit, Some(25));
        assert_eq!(cfg.defaults.profile, OutputProfile::Research);
    }

    #[test]
    fn unquoted_and_commented_values() {
        let cfg = VestigeConfig::parse(
            "[defaults]\nprofile = lean # inline comment\nlimit = 7\n",
        );
        assert_eq!(cfg.defaults.profile, OutputProfile::Lean);
        assert_eq!(cfg.defaults.limit, Some(7));
    }

    #[test]
    fn invalid_values_are_ignored() {
        let cfg = VestigeConfig::parse(
            "[defaults]\ndetail_level = \"loud\"\nlimit = -3\nprofile = \"galaxy\"\n",
        );
        // All invalid -> fall back to defaults.
        assert_eq!(cfg.defaults.detail_level, None);
        assert_eq!(cfg.defaults.limit, None);
        assert_eq!(cfg.defaults.profile, OutputProfile::Default);
    }

    #[test]
    fn unknown_sections_and_keys_ignored() {
        let cfg = VestigeConfig::parse(
            "[future_phase]\nfoo = 1\n[defaults]\nprofile = audit\nbar = baz\n",
        );
        assert_eq!(cfg.defaults.profile, OutputProfile::Audit);
    }

    #[test]
    fn profile_presets() {
        // lean: brief + dropped scores/timestamps + small limit
        let lean = VestigeConfig::parse("[defaults]\nprofile=lean").output();
        assert_eq!(lean.detail_level, "brief");
        assert_eq!(lean.limit, Some(5));
        assert!(!lean.show_scores);
        assert!(!lean.show_timestamps);

        // audit: full detail, no forced limit
        let audit = VestigeConfig::parse("[defaults]\nprofile=audit").output();
        assert_eq!(audit.detail_level, "full");
        assert_eq!(audit.limit, None);

        // research: full detail, larger limit
        let research = VestigeConfig::parse("[defaults]\nprofile=research").output();
        assert_eq!(research.detail_level, "full");
        assert_eq!(research.limit, Some(25));
    }

    #[test]
    fn explicit_defaults_override_profile_presets() {
        // profile=lean would give brief/limit 5, but explicit keys win.
        let out = VestigeConfig::parse(
            "[defaults]\nprofile=lean\ndetail_level=\"full\"\nlimit=42\n",
        )
        .output();
        assert_eq!(out.detail_level, "full");
        assert_eq!(out.limit, Some(42));
    }

    #[test]
    fn precedence_mcp_param_wins() {
        let out = VestigeConfig::parse("[defaults]\nprofile=lean").output();
        // Config says brief, but an explicit MCP param wins.
        assert_eq!(out.resolve_detail_level(Some("full")), "full");
        // No explicit param -> config (lean -> brief).
        assert_eq!(out.resolve_detail_level(None), "brief");
    }

    #[test]
    fn precedence_limit_layers() {
        let out = VestigeConfig::parse("[defaults]\nprofile=research").output();
        // explicit param wins over everything
        assert_eq!(out.resolve_limit(Some(3), 10), 3);
        // no param -> config (research -> 25)
        assert_eq!(out.resolve_limit(None, 10), 25);
        // default profile has no limit -> builtin fallback used
        let def = OutputConfig::default();
        assert_eq!(def.resolve_limit(None, 10), 10);
    }
}
