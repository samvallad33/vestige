#![forbid(unsafe_code)]

//! Incremental `tree-sitter` AST symbol and call-edge extraction.
//!
//! Symbol IDs are `BLAKE3(kind || qualified_path || ast_shape_digest)` and never
//! depend on line numbers, while `shape_digest` normalizes away the declared symbol name
//! so exact renames/moves share the same `shape_digest` without fuzzy similarity.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use tree_sitter::{InputEdit, Language, Node, Parser, Tree};

use crate::error::CodeGraphError;

pub type SymbolId = [u8; 32];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LanguageKind {
    Rust,
    Python,
    TypeScript,
}

impl LanguageKind {
    pub fn from_path(path: &str) -> Option<Self> {
        if path.ends_with(".rs") {
            Some(Self::Rust)
        } else if path.ends_with(".py") {
            #[cfg(feature = "python")]
            {
                Some(Self::Python)
            }
            #[cfg(not(feature = "python"))]
            {
                None
            }
        } else if path.ends_with(".ts") || path.ends_with(".tsx") {
            #[cfg(feature = "typescript")]
            {
                Some(Self::TypeScript)
            }
            #[cfg(not(feature = "typescript"))]
            {
                None
            }
        } else {
            None
        }
    }

    fn ts_language(self) -> Result<Language, CodeGraphError> {
        match self {
            Self::Rust => Ok(tree_sitter_rust::LANGUAGE.into()),
            #[cfg(feature = "python")]
            Self::Python => Ok(tree_sitter_python::LANGUAGE.into()),
            #[cfg(not(feature = "python"))]
            Self::Python => Err(CodeGraphError::Parse(
                "python feature is not enabled".to_owned(),
            )),
            #[cfg(feature = "typescript")]
            Self::TypeScript => Ok(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
            #[cfg(not(feature = "typescript"))]
            Self::TypeScript => Err(CodeGraphError::Parse(
                "typescript feature is not enabled".to_owned(),
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Module,
    Class,
    Interface,
    TypeAlias,
    Constant,
}

impl SymbolKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::Struct => "struct",
            Self::Enum => "enum",
            Self::Trait => "trait",
            Self::Impl => "impl",
            Self::Module => "module",
            Self::Class => "class",
            Self::Interface => "interface",
            Self::TypeAlias => "type_alias",
            Self::Constant => "constant",
        }
    }
}

/// Extracted AST symbol with line-independent `id` and name-normalized `shape_digest`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Symbol {
    pub id: SymbolId,
    pub shape_digest: [u8; 32],
    pub kind: SymbolKind,
    pub name: String,
    pub qualified_path: String,
    pub path: String,
    pub byte_range: (usize, usize),
    /// 1-based inclusive line range `(start_line, end_line)`.
    pub line_range: (u32, u32),
    pub parent: Option<SymbolId>,
}

/// Directed AST call edge between caller symbol ID and callee identifier/path.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CallEdge {
    pub caller: SymbolId,
    pub callee: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedFile {
    pub path: String,
    pub symbols: Vec<Symbol>,
    pub call_edges: Vec<CallEdge>,
}

/// Stateful incremental tree-sitter parser caching the prior `Tree` per file path.
pub struct IncrementalParser {
    trees: BTreeMap<String, (LanguageKind, Tree)>,
}

impl Default for IncrementalParser {
    fn default() -> Self {
        Self::new()
    }
}

impl IncrementalParser {
    pub fn new() -> Self {
        Self {
            trees: BTreeMap::new(),
        }
    }

    /// Parses `source` incrementally if `path` was parsed earlier (applying `edit` if supplied).
    pub fn parse_file(
        &mut self,
        path: &str,
        source: &str,
        edit: Option<&InputEdit>,
    ) -> Result<ParsedFile, CodeGraphError> {
        let Some(lang_kind) = LanguageKind::from_path(path) else {
            return Ok(ParsedFile {
                path: path.to_owned(),
                symbols: Vec::new(),
                call_edges: Vec::new(),
            });
        };

        let mut parser = Parser::new();
        parser
            .set_language(&lang_kind.ts_language()?)
            .map_err(|e| CodeGraphError::Parse(e.to_string()))?;

        // Reuse the cached tree only when the caller describes the delta.
        // A full-buffer reparse (no `InputEdit`) must not keep the old tree:
        // tree-sitter would otherwise retain the previous span and drop symbols
        // that exist only in the new source.
        let prev_tree = if let Some(input_edit) = edit {
            if let Some((prev_lang, old_tree)) = self.trees.get_mut(path) {
                if *prev_lang == lang_kind {
                    old_tree.edit(input_edit);
                    Some(&*old_tree)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let tree = parser
            .parse(source, prev_tree)
            .ok_or_else(|| CodeGraphError::Parse(format!("tree-sitter failed to parse {path}")))?;

        let mut symbols = Vec::new();
        let mut call_edges = Vec::new();
        let mut scope_stack: Vec<(String, SymbolId)> = Vec::new();

        collect_ast(
            lang_kind,
            tree.root_node(),
            path,
            source,
            &mut scope_stack,
            &mut symbols,
            &mut call_edges,
        );

        symbols.sort();
        call_edges.sort();
        call_edges.dedup();

        self.trees.insert(path.to_owned(), (lang_kind, tree));

        Ok(ParsedFile {
            path: path.to_owned(),
            symbols,
            call_edges,
        })
    }
}

fn collect_ast(
    lang: LanguageKind,
    node: Node<'_>,
    file_path: &str,
    source: &str,
    scope_stack: &mut Vec<(String, SymbolId)>,
    symbols: &mut Vec<Symbol>,
    call_edges: &mut Vec<CallEdge>,
) {
    if let Some((kind, name, name_node_range)) = extract_symbol_header(lang, node, source) {
        let parent = scope_stack.last().map(|(_, id)| *id);
        let qualified_path = if scope_stack.is_empty() {
            format!("{file_path}::{name}")
        } else {
            let prefix: Vec<&str> = scope_stack.iter().map(|(n, _)| n.as_str()).collect();
            format!("{file_path}::{}::{name}", prefix.join("::"))
        };

        let shape_digest = compute_ast_shape_digest(node, source, name_node_range);
        let id = compute_symbol_id(kind, &qualified_path, &shape_digest);

        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        symbols.push(Symbol {
            id,
            shape_digest,
            kind,
            name: name.clone(),
            qualified_path,
            path: file_path.to_owned(),
            byte_range: (node.start_byte(), node.end_byte()),
            line_range: (start_line, end_line),
            parent,
        });

        scope_stack.push((name, id));
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            collect_ast(
                lang,
                child,
                file_path,
                source,
                scope_stack,
                symbols,
                call_edges,
            );
        }
        scope_stack.pop();
        return;
    }

    if let Some(callee) = extract_call_target(lang, node, source) {
        if let Some(&(_, caller_id)) = scope_stack.last() {
            call_edges.push(CallEdge {
                caller: caller_id,
                callee,
            });
        }
    }

    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        collect_ast(
            lang,
            child,
            file_path,
            source,
            scope_stack,
            symbols,
            call_edges,
        );
    }
}

type SymbolHeader = (SymbolKind, String, Option<(usize, usize)>);

fn extract_symbol_header(
    lang: LanguageKind,
    node: Node<'_>,
    source: &str,
) -> Option<SymbolHeader> {
    let kind_str = node.kind();
    let sym_kind = match lang {
        LanguageKind::Rust => match kind_str {
            "function_item" => SymbolKind::Function,
            "struct_item" => SymbolKind::Struct,
            "enum_item" => SymbolKind::Enum,
            "trait_item" => SymbolKind::Trait,
            "impl_item" => SymbolKind::Impl,
            "mod_item" => SymbolKind::Module,
            "type_item" => SymbolKind::TypeAlias,
            "const_item" | "static_item" => SymbolKind::Constant,
            _ => return None,
        },
        LanguageKind::Python => match kind_str {
            "function_definition" => SymbolKind::Function,
            "class_definition" => SymbolKind::Class,
            _ => return None,
        },
        LanguageKind::TypeScript => match kind_str {
            "function_declaration" | "method_definition" => SymbolKind::Function,
            "class_declaration" => SymbolKind::Class,
            "interface_declaration" => SymbolKind::Interface,
            "type_alias_declaration" => SymbolKind::TypeAlias,
            _ => return None,
        },
    };

    if let Some(name_node) = node.child_by_field_name("name") {
        let text = node_text(name_node, source);
        return Some((
            sym_kind,
            text.to_owned(),
            Some((name_node.start_byte(), name_node.end_byte())),
        ));
    }

    if sym_kind == SymbolKind::Impl {
        if let Some(type_node) = node.child_by_field_name("type") {
            let text = node_text(type_node, source);
            return Some((sym_kind, format!("impl {text}"), None));
        }
    }

    None
}

fn extract_call_target(lang: LanguageKind, node: Node<'_>, source: &str) -> Option<String> {
    let kind_str = node.kind();
    match lang {
        LanguageKind::Rust | LanguageKind::TypeScript => {
            if kind_str == "call_expression" {
                let func_node = node.child_by_field_name("function")?;
                return Some(node_text(func_node, source).to_owned());
            }
        }
        LanguageKind::Python => {
            if kind_str == "call" {
                let func_node = node.child_by_field_name("function")?;
                return Some(node_text(func_node, source).to_owned());
            }
        }
    }
    None
}

fn node_text<'a>(node: Node<'_>, source: &'a str) -> &'a str {
    source
        .get(node.start_byte()..node.end_byte())
        .unwrap_or("")
}

/// Computes a normalized structural AST shape digest that excludes comments, whitespace,
/// line numbers, and the symbol's own declared identifier byte span (`name_span`).
fn compute_ast_shape_digest(
    root: Node<'_>,
    source: &str,
    name_span: Option<(usize, usize)>,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vestige.codegraph.ast_shape.v1\0");
    hash_ast_node(root, source, name_span, &mut hasher);
    *hasher.finalize().as_bytes()
}

fn hash_ast_node(
    node: Node<'_>,
    source: &str,
    name_span: Option<(usize, usize)>,
    hasher: &mut blake3::Hasher,
) {
    let kind = node.kind();
    if kind == "line_comment" || kind == "block_comment" || kind == "comment" {
        return;
    }

    let span = (node.start_byte(), node.end_byte());
    if Some(span) == name_span {
        hasher.update(b"SYM_NAME\0");
        return;
    }

    hasher.update(b"(");
    hasher.update(kind.as_bytes());
    hasher.update(b"\0");

    if node.named_child_count() == 0 {
        let leaf = node_text(node, source).trim();
        hasher.update(&(leaf.len() as u64).to_le_bytes());
        hasher.update(leaf.as_bytes());
    } else {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            hash_ast_node(child, source, name_span, hasher);
        }
    }
    hasher.update(b")");
}

/// Deterministic symbol ID derived from kind, qualified path, and AST shape digest (never line numbers).
pub fn compute_symbol_id(
    kind: SymbolKind,
    qualified_path: &str,
    shape_digest: &[u8; 32],
) -> SymbolId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vestige.codegraph.symbol.v1\0");
    hasher.update(kind.as_str().as_bytes());
    hasher.update(b"\0");
    hasher.update(qualified_path.as_bytes());
    hasher.update(b"\0");
    hasher.update(shape_digest);
    *hasher.finalize().as_bytes()
}
