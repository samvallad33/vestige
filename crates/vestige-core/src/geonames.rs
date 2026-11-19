//! GeoNames Geographical Gazetteer (CC BY 4.0, whole unedited official dump files).
//!
//! Bundled verbatim under `crates/vestige-core/data/geonames/`:
//! - `countryInfo.txt` (all sovereign countries, capitals, continents, ISO codes)
//! - `admin1CodesASCII.txt` (all first-order administrative states/provinces/regions)
//! - `cities15000.txt` (all world cities with population >= 15,000)
//!
//! Every file's LF-normalized sha256 is pinned in constants below and verified by both
//! unit tests and `scripts/hardcode_scan.sh`. Provides deterministic class->instance
//! (`"state"` -> states, `"country"` -> countries, `"city"` -> major cities) and
//! bidirectional containment (`city <-> state <-> country <-> continent`) lookups.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, OnceLock};

const COUNTRY_INFO: &str = include_str!("../data/geonames/countryInfo.txt");
const ADMIN1_CODES: &str = include_str!("../data/geonames/admin1CodesASCII.txt");
const CITIES_15000: &str = include_str!("../data/geonames/cities15000.txt");

/// Pinned LF-normalized sha256 of `countryInfo.txt`.
pub const COUNTRY_INFO_SHA256: &str =
    "f0693bb94cbc5492d0b68fceb54b04b2addd0a710fb2f34e6e19268d61479512";
/// Pinned LF-normalized sha256 of `admin1CodesASCII.txt`.
pub const ADMIN1_CODES_SHA256: &str =
    "2e4867f5397acb056ea26377601b57af37cb4c81399770225018e9e439e91ad3";
/// Pinned LF-normalized sha256 of `cities15000.txt`.
pub const CITIES_15000_SHA256: &str =
    "1a0282bec53214296770884ecde9f134a932bf3fad2bb7ee277918e7535d1fb3";

struct GeoTable {
    /// Lowercased single-token place name -> related place tokens `(neighbor, weight_u16)`.
    neighbors: HashMap<String, Vec<(String, u16)>>,
    /// Distinct single-word tokens of sovereign countries.
    country_tokens: Vec<String>,
    /// Distinct single-word tokens of first-order states/provinces.
    state_tokens: Vec<String>,
    /// Distinct single-word tokens of major world cities (`pop >= 100_000` or capital).
    city_tokens: Vec<String>,
}

fn place_tokens(name: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for raw in name.split(|c: char| !c.is_ascii_alphabetic()) {
        if raw.len() >= 3 {
            let w = raw.to_ascii_lowercase();
            // hardcode-scan: closed-class (geographical prepositions/articles inside multiword place names)
            if matches!(w.as_str(), "the" | "and" | "for" | "del" | "las" | "los" | "san" | "new" | "old") {
                continue;
            }
            if !out.contains(&w) {
                out.push(w);
            }
        }
    }
    out
}

fn continent_name(code: &str) -> Option<&'static str> {
    match code {
        "AF" => Some("africa"),
        "AN" => Some("antarctica"),
        "AS" => Some("asia"),
        "EU" => Some("europe"),
        "NA" => Some("america"),
        "OC" => Some("oceania"),
        "SA" => Some("america"),
        _ => None,
    }
}

fn add_edge(map: &mut HashMap<String, Vec<(String, u16)>>, a: &str, b: &str, w: u16) {
    if a == b || a.len() < 3 || b.len() < 3 {
        return;
    }
    let list = map.entry(a.to_string()).or_default();
    if let Some(existing) = list.iter_mut().find(|(k, _)| k == b) {
        if w > existing.1 {
            existing.1 = w;
        }
    } else if list.len() < 48 {
        list.push((b.to_string(), w));
    }
}

fn parse_geonames() -> GeoTable {
    let mut neighbors: HashMap<String, Vec<(String, u16)>> = HashMap::new();
    let mut country_by_iso: HashMap<String, Vec<String>> = HashMap::new();
    let mut continent_by_iso: HashMap<String, &'static str> = HashMap::new();
    let mut country_tokens: Vec<String> = Vec::new();
    let mut state_tokens: Vec<String> = Vec::new();
    let mut city_tokens: Vec<String> = Vec::new();

    // 1. Parse `countryInfo.txt` (ISO=0, Country=4, Capital=5, Continent=8).
    for line in COUNTRY_INFO.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 9 {
            continue;
        }
        let iso = f[0].trim();
        let country = f[4].trim();
        let capital = f[5].trim();
        let cont = f[8].trim();
        let c_toks = place_tokens(country);
        if c_toks.is_empty() {
            continue;
        }
        for t in &c_toks {
            if !country_tokens.contains(t) {
                country_tokens.push(t.clone());
            }
        }
        country_by_iso.insert(iso.to_string(), c_toks.clone());
        if let Some(cn) = continent_name(cont) {
            continent_by_iso.insert(iso.to_string(), cn);
            for ct in &c_toks {
                add_edge(&mut neighbors, ct, cn, 75);
                add_edge(&mut neighbors, cn, ct, 60);
            }
        }
        let cap_toks = place_tokens(capital);
        for cap in &cap_toks {
            if !city_tokens.contains(cap) {
                city_tokens.push(cap.clone());
            }
            for ct in &c_toks {
                add_edge(&mut neighbors, cap, ct, 92);
                add_edge(&mut neighbors, ct, cap, 84);
            }
            if let Some(cn) = continent_name(cont) {
                add_edge(&mut neighbors, cap, cn, 65);
            }
        }
    }

    // 2. Parse `admin1CodesASCII.txt` (`ISO.ADMIN1`=0, ascii_name=2).
    let mut admin1_by_key: HashMap<String, Vec<String>> = HashMap::new();
    for line in ADMIN1_CODES.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 3 {
            continue;
        }
        let key = f[0].trim();
        let Some((iso, _)) = key.split_once('.') else {
            continue;
        };
        let a_toks = place_tokens(f[2].trim());
        if a_toks.is_empty() {
            continue;
        }
        // Keep all US/CA/AU/GB/DE/FR/IN/BR/MX/CN/JP/IT/ES admin1 names and distinctive global admin1 names.
        for at in &a_toks {
            if !state_tokens.contains(at) {
                state_tokens.push(at.clone());
            }
        }
        if let Some(c_toks) = country_by_iso.get(iso) {
            for at in &a_toks {
                for ct in c_toks {
                    add_edge(&mut neighbors, at, ct, 88);
                    add_edge(&mut neighbors, ct, at, 70);
                }
                if let Some(&cn) = continent_by_iso.get(iso) {
                    add_edge(&mut neighbors, at, cn, 60);
                }
            }
        }
        admin1_by_key.insert(key.to_string(), a_toks);
    }

    // 3. Parse `cities15000.txt` (asciiname=2, country_code=8, admin1_code=10, population=14).
    for line in CITIES_15000.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 15 {
            continue;
        }
        let pop: u64 = f[14].trim().parse().unwrap_or(0);
        if pop < 50_000 {
            continue;
        }
        let city_toks = place_tokens(f[2].trim());
        let iso = f[8].trim();
        let adm1 = f[10].trim();
        let akey = format!("{iso}.{adm1}");
        for ct in &city_toks {
            if pop >= 150_000 && !city_tokens.contains(ct) {
                city_tokens.push(ct.clone());
            }
            if let Some(a_toks) = admin1_by_key.get(&akey) {
                for at in a_toks {
                    add_edge(&mut neighbors, ct, at, 90);
                    add_edge(&mut neighbors, at, ct, 72);
                }
            }
            if let Some(c_toks) = country_by_iso.get(iso) {
                for country_t in c_toks {
                    add_edge(&mut neighbors, ct, country_t, 86);
                }
            }
        }
    }

    country_tokens.sort_unstable();
    country_tokens.dedup();
    state_tokens.sort_unstable();
    state_tokens.dedup();
    city_tokens.sort_unstable();
    city_tokens.dedup();
    for list in neighbors.values_mut() {
        list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    }

    GeoTable {
        neighbors,
        country_tokens,
        state_tokens,
        city_tokens,
    }
}

fn table() -> &'static GeoTable {
    static T: OnceLock<GeoTable> = OnceLock::new();
    T.get_or_init(parse_geonames)
}

type GeoCache = Mutex<HashMap<String, Vec<(&'static str, u16)>>>;

/// Return deterministic geographical containment/association neighbors `(place_token, weight_u16)`
/// for a place name token (`city <-> state <-> country <-> continent`) from GeoNames.
pub fn geo_neighbors(word: &str) -> Vec<(&'static str, u16)> {
    static CACHE: OnceLock<GeoCache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = word.to_ascii_lowercase();
    if let Ok(guard) = cache.lock()
        && let Some(hit) = guard.get(&key)
    {
        return hit.clone();
    }
    let tbl = table();
    let mut acc: BTreeMap<&'static str, u16> = BTreeMap::new();
    if let Some(list) = tbl.neighbors.get(&key) {
        for (dst, w) in list {
            let e = acc.entry(dst.as_str()).or_insert(0);
            *e = (*e).max(*w);
        }
    }
    let mut out: Vec<(&'static str, u16)> = acc.into_iter().collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    if let Ok(mut guard) = cache.lock() {
        guard.insert(key, out.clone());
    }
    out
}

/// Return gazetteer instance tokens for a geographical class noun (`state`/`province`/`region`,
/// `country`/`nation`, `city`/`town`/`capital`). Caller intersects strictly with the local
/// conversation vocabulary and weights by corpus IC.
pub fn geo_class_instances(class_word: &str) -> &'static [String] {
    let tbl = table();
    match class_word.to_ascii_lowercase().as_str() {
        "state" | "states" | "province" | "provinces" | "region" | "regions" => &tbl.state_tokens,
        "country" | "countries" | "nation" | "nations" | "homeland" => &tbl.country_tokens,
        "city" | "cities" | "town" | "towns" | "capital" | "capitals" => &tbl.city_tokens,
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn geonames_files_match_pinned_sha256() {
        for (name, data, expected) in [
            ("countryInfo.txt", COUNTRY_INFO, COUNTRY_INFO_SHA256),
            ("admin1CodesASCII.txt", ADMIN1_CODES, ADMIN1_CODES_SHA256),
            ("cities15000.txt", CITIES_15000, CITIES_15000_SHA256),
        ] {
            let normalized = data.replace("\r\n", "\n");
            let digest = Sha256::digest(normalized.as_bytes());
            let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(hex, expected, "GeoNames checksum mismatch for {name}");
        }
    }

    #[test]
    fn geonames_resolves_states_countries_and_containment() {
        assert!(geo_class_instances("state").iter().any(|s| s == "california"));
        assert!(geo_class_instances("country").iter().any(|s| s == "sweden"));
        let austin = geo_neighbors("austin");
        assert!(
            austin.iter().any(|&(w, _)| w == "texas"),
            "expected austin -> texas: {austin:?}"
        );
        let stockholm = geo_neighbors("stockholm");
        assert!(
            stockholm.iter().any(|&(w, _)| w == "sweden"),
            "expected stockholm -> sweden: {stockholm:?}"
        );
    }
}
