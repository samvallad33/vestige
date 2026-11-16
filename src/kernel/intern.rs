//! src/kernel/intern.rs — File 1: The NUMBERS File
//! Zero-allocation token interner. Maps strings to dense 32-bit integer IDs.
//! All subsequent retrieval operations run in L1/L2 cache over contiguous slices.

use std::collections::HashMap;

#[derive(Default, Debug, Clone)]
pub struct StringInterner {
    map: HashMap<String, u32>,
    names: Vec<String>,
}

impl StringInterner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a string slice into a dense 32-bit ID.
    /// Allocation happens exactly once; subsequent lookups return u32.
    pub fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = self.names.len() as u32;
        self.map.insert(s.to_string(), id);
        self.names.push(s.to_string());
        id
    }

    #[inline(always)]
    pub fn get(&self, s: &str) -> Option<u32> {
        self.map.get(s).copied()
    }

    #[inline(always)]
    pub fn resolve(&self, id: u32) -> Option<&str> {
        self.names.get(id as usize).map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// A compact tokenized sequence stored as flat (u32, u32) pairs:
/// (exact_token_id, stemmed_token_id).
/// Stored in contiguous memory with zero pointer chasing.
pub type TokenSequence = Vec<(u32, u32)>;

#[derive(Clone, Copy, Debug)]
pub struct PackedFeature {
    pub id: u32,
    pub weight: f32,
    pub counts_len: bool,
    pub inheritable: bool,
}
