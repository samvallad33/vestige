#![forbid(unsafe_code)]

use std::fmt;

/// Errors returned by `vestige-codegraph`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodeGraphError {
    Io(String),
    Git(String),
    Parse(String),
    InvalidLine { path: String, line: u32 },
}

impl fmt::Display for CodeGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
            Self::Git(msg) => write!(f, "git repository error: {msg}"),
            Self::Parse(msg) => write!(f, "AST parse error: {msg}"),
            Self::InvalidLine { path, line } => {
                write!(f, "invalid 1-based line {line} for path {path}")
            }
        }
    }
}

impl std::error::Error for CodeGraphError {}

impl From<std::io::Error> for CodeGraphError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err.to_string())
    }
}
