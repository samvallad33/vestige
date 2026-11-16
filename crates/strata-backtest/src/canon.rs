//! Canonical JSON (sorted object keys, no whitespace) and Q32.32 display.

use std::collections::BTreeMap;

use strata_kernel::canonical::to_q32_32;

/// Quantize a model real to Q32.32. The JSON stores this integer, not `f64`.
pub fn quantize(value: f64) -> i64 {
    to_q32_32(value)
}

/// Eight truncated fraction digits of a Q32.32 integer.
///
/// `frac * 10^8 / 2^32` drops the remainder. The integer from [`quantize`]
/// is the exact published value; this string is only the markdown rendering.
pub fn q_decimal(q: i64) -> String {
    let neg = q < 0;
    let v = q.unsigned_abs();
    let ip = v >> 32;
    let frac = (v & 0xFFFF_FFFF) as u128;
    let digits = frac * 100_000_000 / (1u128 << 32);
    format!("{}{ip}.{digits:08}", if neg { "-" } else { "" })
}

/// Lowercase hex, no `0x` prefix.
pub fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

/// A JSON value restricted to the types the manifests use.
#[derive(Clone, Debug)]
pub enum Json {
    /// Signed integer, including every Q32.32 field.
    Int(i64),
    /// Unsigned integer (sequence numbers, counts).
    U64(u64),
    /// String.
    Str(String),
    /// Array, order preserved.
    Arr(Vec<Json>),
    /// Object. Keys are sorted on render.
    Obj(BTreeMap<String, Json>),
}

impl Json {
    /// Object from pairs. Later duplicates replace earlier ones.
    pub fn obj(pairs: Vec<(&str, Json)>) -> Self {
        let mut map = BTreeMap::new();
        for (key, value) in pairs {
            map.insert(key.to_string(), value);
        }
        Json::Obj(map)
    }

    /// Canonical encoding plus a trailing newline.
    pub fn canonical(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out.push('\n');
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Int(value) => push_i64(out, *value),
            Json::U64(value) => out.push_str(&value.to_string()),
            Json::Str(value) => push_string(out, value),
            Json::Arr(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Json::Obj(map) => {
                out.push('{');
                for (index, (key, value)) in map.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    push_string(out, key);
                    out.push(':');
                    value.write(out);
                }
                out.push('}');
            }
        }
    }
}

fn push_i64(out: &mut String, value: i64) {
    if value < 0 {
        out.push('-');
        out.push_str(&value.unsigned_abs().to_string());
    } else {
        out.push_str(&value.to_string());
    }
}

fn push_string(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{code:04x}", code = c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn objects_sort_keys_and_end_with_a_newline() {
        let raw = Json::obj(vec![("b", Json::Int(1)), ("a", Json::Str("x".into()))]).canonical();
        assert_eq!(raw, "{\"a\":\"x\",\"b\":1}\n");
    }

    #[test]
    fn q_decimal_truncates_toward_zero() {
        assert_eq!(q_decimal(0), "0.00000000");
        assert_eq!(q_decimal(quantize(1.0)), "1.00000000");
        assert_eq!(q_decimal(quantize(0.5)), "0.50000000");
        assert_eq!(q_decimal(quantize(-0.5)), "-0.50000000");
        assert_eq!(q_decimal(quantize(-1.5)), "-1.50000000");
    }

    #[test]
    fn hex_is_lowercase() {
        assert_eq!(hex_bytes(&[0x0a, 0xff]), "0aff");
    }
}
