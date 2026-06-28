//! Code-Specific Embeddings
//!
//! Specialized embedding handling for source code:
//! - Language-aware tokenization
//! - Structure preservation
//! - Semantic chunking
//!
//! Future: Support for code-specific embedding models.

use super::local::{Embedding, EmbeddingError, EmbeddingService};

// ============================================================================
// CODE EMBEDDING
// ============================================================================

/// Code-aware embedding generator
pub struct CodeEmbedding {
    /// General embedding service (fallback)
    service: EmbeddingService,
}

impl Default for CodeEmbedding {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeEmbedding {
    /// Create a new code embedding generator
    pub fn new() -> Self {
        Self {
            service: EmbeddingService::new(),
        }
    }

    /// Check if ready
    pub fn is_ready(&self) -> bool {
        self.service.is_ready()
    }

    /// Initialize the embedding model
    pub fn init(&self) -> Result<(), EmbeddingError> {
        self.service.init()
    }

    /// Generate embedding for code
    ///
    /// Currently uses the general embedding model with code preprocessing.
    /// Future: Use code-specific models like CodeBERT.
    pub fn embed_code(
        &self,
        code: &str,
        language: Option<&str>,
    ) -> Result<Embedding, EmbeddingError> {
        // Preprocess code for better embedding
        let processed = self.preprocess_code(code, language);
        self.service.embed(&processed)
    }

    /// Preprocess code for embedding
    fn preprocess_code(&self, code: &str, language: Option<&str>) -> String {
        let mut result = String::new();

        // Add language hint if available
        if let Some(lang) = language {
            result.push_str(&format!("[{}] ", lang.to_uppercase()));
        }

        // Clean and normalize code
        let cleaned = self.clean_code(code);
        result.push_str(&cleaned);

        result
    }

    /// Clean code by removing excessive whitespace and normalizing
    fn clean_code(&self, code: &str) -> String {
        let lines: Vec<&str> = code
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .filter(|l| !self.is_comment_only(l))
            .collect();

        lines.join(" ")
    }

    /// Check if a line is only a comment.
    ///
    /// Conservative on purpose: the previous version treated any line starting
    /// with `#` or `*` as a comment, which deleted real code — C preprocessor
    /// directives (`#include`, `#define`), pointer derefs / multiplication
    /// (`*ptr = 5;`), CSS `* { ... }`, etc. We only strip unambiguous comment
    /// markers; a leading `*` counts only as a block-comment continuation
    /// (`* ...`), never a bare `*expr`.
    fn is_comment_only(&self, line: &str) -> bool {
        let trimmed = line.trim();
        trimmed.starts_with("//")
            || trimmed.starts_with("/*")
            || trimmed == "*"
            || trimmed.starts_with("* ")
            || trimmed.starts_with("*/")
    }

    /// Extract semantic chunks from code
    ///
    /// Splits code into meaningful chunks for separate embedding.
    pub fn chunk_code(&self, code: &str, language: Option<&str>) -> Vec<CodeChunk> {
        let mut chunks = Vec::new();
        let lines: Vec<&str> = code.lines().collect();

        // Simple chunking based on empty lines and definitions
        let mut current_chunk = Vec::new();
        let mut chunk_type = ChunkType::Block;

        for line in lines {
            let trimmed = line.trim();

            // Detect chunk boundaries
            if self.is_definition_start(trimmed, language) {
                // Save previous chunk if not empty
                if !current_chunk.is_empty() {
                    chunks.push(CodeChunk {
                        content: current_chunk.join("\n"),
                        chunk_type,
                        language: language.map(String::from),
                    });
                    current_chunk.clear();
                }
                chunk_type = self.get_chunk_type(trimmed, language);
            }

            current_chunk.push(line);
        }

        // Save final chunk
        if !current_chunk.is_empty() {
            chunks.push(CodeChunk {
                content: current_chunk.join("\n"),
                chunk_type,
                language: language.map(String::from),
            });
        }

        chunks
    }

    /// Check if a line starts a new definition
    fn is_definition_start(&self, line: &str, language: Option<&str>) -> bool {
        match language {
            Some("rust") => {
                line.starts_with("fn ")
                    || line.starts_with("pub fn ")
                    || line.starts_with("struct ")
                    || line.starts_with("pub struct ")
                    || line.starts_with("enum ")
                    || line.starts_with("impl ")
                    || line.starts_with("trait ")
            }
            Some("python") => {
                line.starts_with("def ")
                    || line.starts_with("class ")
                    || line.starts_with("async def ")
            }
            Some("javascript") | Some("typescript") => {
                line.starts_with("function ")
                    || line.starts_with("class ")
                    || line.starts_with("const ")
                    || line.starts_with("export ")
            }
            _ => {
                // Generic detection
                line.starts_with("function ")
                    || line.starts_with("def ")
                    || line.starts_with("class ")
                    || line.starts_with("fn ")
            }
        }
    }

    /// Determine chunk type from definition line
    fn get_chunk_type(&self, line: &str, _language: Option<&str>) -> ChunkType {
        if line.contains("fn ") || line.contains("function ") || line.contains("def ") {
            ChunkType::Function
        } else if line.contains("class ") || line.contains("struct ") {
            ChunkType::Class
        } else if line.contains("impl ") || line.contains("trait ") {
            ChunkType::Implementation
        } else {
            ChunkType::Block
        }
    }
}

/// A chunk of code for embedding
#[derive(Debug, Clone)]
pub struct CodeChunk {
    /// The code content
    pub content: String,
    /// Type of chunk (function, class, etc.)
    pub chunk_type: ChunkType,
    /// Programming language if known
    pub language: Option<String>,
}

/// Types of code chunks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkType {
    /// A function or method
    Function,
    /// A class or struct
    Class,
    /// An implementation block
    Implementation,
    /// A generic code block
    Block,
    /// An import statement
    Import,
    /// A comment or documentation
    Comment,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_embedding_creation() {
        let ce = CodeEmbedding::new();
        // Just verify creation succeeds - is_ready() may return true
        // if fastembed can load the model
        let _ = ce.is_ready();
    }

    #[test]
    fn test_clean_code() {
        let ce = CodeEmbedding::new();
        let code = r#"
            // This is a comment
            fn hello() {
                println!("Hello");
            }
        "#;

        let cleaned = ce.clean_code(code);
        assert!(!cleaned.contains("// This is a comment"));
        assert!(cleaned.contains("fn hello()"));
    }

    #[test]
    fn test_chunk_code_rust() {
        let ce = CodeEmbedding::new();
        // Trim the code to avoid empty initial chunk from leading newline
        let code = r#"fn foo() {
    println!("foo");
}

fn bar() {
    println!("bar");
}"#;

        let chunks = ce.chunk_code(code, Some("rust"));
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].chunk_type, ChunkType::Function);
        assert_eq!(chunks[1].chunk_type, ChunkType::Function);
    }

    #[test]
    fn test_chunk_code_python() {
        let ce = CodeEmbedding::new();
        let code = r#"
def hello():
    print("hello")

class Greeter:
    def greet(self):
        print("greet")
        "#;

        let chunks = ce.chunk_code(code, Some("python"));
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_is_definition_start() {
        let ce = CodeEmbedding::new();

        assert!(ce.is_definition_start("fn hello()", Some("rust")));
        assert!(ce.is_definition_start("pub fn hello()", Some("rust")));
        assert!(ce.is_definition_start("def hello():", Some("python")));
        assert!(ce.is_definition_start("class Foo:", Some("python")));
        assert!(ce.is_definition_start("function foo() {", Some("javascript")));
    }
}
