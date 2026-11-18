#!/usr/bin/env python3
"""
Deterministic, benchmark-agnostic CC0 topological containment table builder:
Queries Wikidata's public CC0 1.0 SPARQL endpoint using TSV streaming across general
ontological class hierarchies (`wdt:P279*` administrative divisions, human settlements,
museums/theatres, artworks/collections, central institutions, and foods/drinks,
including English `skos:altLabel` aliases) and packs normalized English edges into a
sha256-pinned TSV at `crates/vestige-core/data/wikidata-cc0-4rel.tsv`.
"""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

OUT_PATH = Path("crates/vestige-core/data/wikidata-cc0-4rel.tsv")
ENDPOINT = "https://query.wikidata.org/sparql"
UA = "VestigeTopologicalBuilder/1.2 (https://github.com/samvallad33/vestige; CC0 static index)"

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def unquote_tsv(cell: str) -> str:
    cell = cell.strip()
    if "@" in cell and cell.startswith('"'):
        cell = cell.rsplit("@", 1)[0]
    if cell.startswith('"') and cell.endswith('"') and len(cell) >= 2:
        cell = cell[1:-1].replace('\\"', '"')
    if cell.startswith("^^") or "^^" in cell:
        cell = cell.split("^^", 1)[0].strip('"')
    return cell


def norm_label(raw: str) -> str | None:
    if not raw or (raw.startswith("Q") and raw[1:].isdigit()):
        return None
    folded = (
        unicodedata.normalize("NFKD", raw)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    folded = folded.replace("-", " ")
    toks = TOKEN_RE.findall(folded)
    if not (1 <= len(toks) <= 6):
        return None
    phrase = " ".join(toks)
    if len(phrase) < 3 or len(phrase) > 48:
        return None
    return phrase


def run_sparql_tsv(name: str, query: str) -> list[tuple[str, str, str, str]]:
    data_bytes = urllib.parse.urlencode({"query": query}).encode("utf-8")
    for attempt in range(6):
        try:
            req = urllib.request.Request(
                ENDPOINT,
                data=data_bytes,
                headers={
                    "User-Agent": UA,
                    "Accept": "text/tab-separated-values",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Connection": "close",
                },
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                lines = resp.read().decode("utf-8", errors="replace").splitlines()
                if not lines:
                    return []
                hdr = [h.lstrip("?") for h in lines[0].split("\t")]
                idx_s = hdr.index("srcLabel")
                idx_a = hdr.index("alt") if "alt" in hdr else -1
                idx_d = hdr.index("dstLabel")
                idx_w = hdr.index("sl")
                out = []
                for ln in lines[1:]:
                    cols = ln.split("\t")
                    if len(cols) <= max(idx_s, idx_d, idx_w):
                        continue
                    s_v = unquote_tsv(cols[idx_s])
                    a_v = unquote_tsv(cols[idx_a]) if 0 <= idx_a < len(cols) else ""
                    d_v = unquote_tsv(cols[idx_d])
                    w_v = unquote_tsv(cols[idx_w])
                    out.append((s_v, a_v, d_v, w_v))
                print(f"  [{name}] fetched {len(out)} TSV rows")
                time.sleep(1.5)
                return out
        except Exception as exc:
            print(f"  [{name}] attempt {attempt + 1} error: {exc}")
            time.sleep(4 * (attempt + 1))
    print(f"  [{name}] WARNING: query skipped after 6 attempts; keeping existing TSV edges")
    return []


QUERIES: list[tuple[str, str, str, int]] = [
    (
        "admin_divisions",
        "P17",
        """
SELECT ?srcLabel ?alt ?dstLabel ?sl WHERE {
  ?cls wdt:P279* wd:Q10864048 .
  ?src wdt:P31 ?cls ; wikibase:sitelinks ?sl .
  { ?src wdt:P17 ?dst } UNION { ?src wdt:P131 ?dst }
  FILTER(?sl >= 14)
  ?src rdfs:label ?srcLabel . FILTER(LANG(?srcLabel) = "en")
  ?dst rdfs:label ?dstLabel . FILTER(LANG(?dstLabel) = "en")
  OPTIONAL { ?src skos:altLabel ?alt . FILTER(LANG(?alt) = "en") }
}
LIMIT 35000
""",
        16,
    ),
    (
        "cities_sl_ge_22",
        "P17",
        """
SELECT ?srcLabel ?dstLabel ?sl WHERE {
  VALUES ?cls { wd:Q515 wd:Q1549591 wd:Q3957 wd:Q484170 wd:Q262166 wd:Q7930989 wd:Q15284 wd:Q486972 wd:Q5119 }
  ?src wdt:P31 ?cls ; wikibase:sitelinks ?sl ; wdt:P17 ?dst .
  FILTER(?sl >= 22)
  ?src rdfs:label ?srcLabel . FILTER(LANG(?srcLabel) = "en")
  ?dst rdfs:label ?dstLabel . FILTER(LANG(?dstLabel) = "en")
}
LIMIT 40000
""",
        0,
    ),
    (
        "capitals_and_metropolises_p17",
        "P17",
        """
SELECT ?srcLabel ?alt ?dstLabel ?sl WHERE {
  { ?cls wdt:P279* wd:Q5119 } UNION { ?cls wdt:P279* wd:Q1549591 }
  ?src wdt:P31 ?cls ; wikibase:sitelinks ?sl ; wdt:P17 ?dst .
  FILTER(?sl >= 25)
  ?src rdfs:label ?srcLabel . FILTER(LANG(?srcLabel) = "en")
  ?dst rdfs:label ?dstLabel . FILTER(LANG(?dstLabel) = "en")
  OPTIONAL { ?src skos:altLabel ?alt . FILTER(LANG(?alt) = "en") }
}
LIMIT 15000
""",
        25,
    ),
    (
        "cities_sl_14_to_21",
        "P17",
        """
SELECT ?srcLabel ?dstLabel ?sl WHERE {
  VALUES ?cls { wd:Q515 wd:Q3957 wd:Q484170 wd:Q262166 wd:Q7930989 wd:Q15284 }
  ?src wdt:P31 ?cls ; wikibase:sitelinks ?sl ; wdt:P17 ?dst .
  FILTER(?sl >= 14 && ?sl < 22)
  ?src rdfs:label ?srcLabel . FILTER(LANG(?srcLabel) = "en")
  ?dst rdfs:label ?dstLabel . FILTER(LANG(?dstLabel) = "en")
}
LIMIT 32000
""",
        18,
    ),
    (
        "towns_sl_7_to_13",
        "P17",
        """
SELECT ?srcLabel ?dstLabel ?sl WHERE {
  VALUES ?cls { wd:Q515 wd:Q3957 wd:Q484170 wd:Q7930989 }
  ?src wdt:P31 ?cls ; wikibase:sitelinks ?sl ; wdt:P17 ?dst .
  FILTER(?sl >= 7 && ?sl < 14)
  ?src rdfs:label ?srcLabel . FILTER(LANG(?srcLabel) = "en")
  ?dst rdfs:label ?dstLabel . FILTER(LANG(?dstLabel) = "en")
}
LIMIT 32000
""",
        20,
    ),
    (
        "national_city_subclasses",
        "P17",
        """
SELECT ?srcLabel ?dstLabel ?sl WHERE {
  { ?cls wdt:P279* wd:Q515 . FILTER(?cls != wd:Q515) }
  UNION
  { ?cls wdt:P279* wd:Q3957 . FILTER(?cls != wd:Q3957) }
  ?src wdt:P31 ?cls ; wikibase:sitelinks ?sl ; wdt:P17 ?dst .
  FILTER(?sl >= 8)
  ?src rdfs:label ?srcLabel . FILTER(LANG(?srcLabel) = "en")
  ?dst rdfs:label ?dstLabel . FILTER(LANG(?dstLabel) = "en")
}
LIMIT 35000
""",
        20,
    ),
    (
        "municipalities_and_communes_p17",
        "P17",
        """
SELECT ?srcLabel ?dstLabel ?sl WHERE {
  ?cls wdt:P279* wd:Q15284 .
  ?src wdt:P31 ?cls ; wikibase:sitelinks ?sl ; wdt:P17 ?dst .
  FILTER(?sl >= 22)
  ?src rdfs:label ?srcLabel . FILTER(LANG(?srcLabel) = "en")
  ?dst rdfs:label ?dstLabel . FILTER(LANG(?dstLabel) = "en")
}
LIMIT 35000
""",
        22,
    ),
    (
        "cities_world_sl_ge_28_p17",
        "P17",
        """
SELECT ?srcLabel ?dstLabel ?sl WHERE {
  ?cls wdt:P279* wd:Q515 .
  ?src wdt:P31 ?cls ; wikibase:sitelinks ?sl ; wdt:P17 ?dst .
  FILTER(?sl >= 28)
  ?src rdfs:label ?srcLabel . FILTER(LANG(?srcLabel) = "en")
  ?dst rdfs:label ?dstLabel . FILTER(LANG(?dstLabel) = "en")
}
LIMIT 30000
""",
        28,
    ),
    (
        "municipal_subdivisions_p131",
        "P131",
        """
SELECT ?srcLabel ?dstLabel ?sl WHERE {
  ?cls wdt:P279* wd:Q15284 .
  ?src wdt:P31 ?cls ; wikibase:sitelinks ?sl ; wdt:P131 ?dst .
  FILTER(?sl >= 18)
  ?src rdfs:label ?srcLabel . FILTER(LANG(?srcLabel) = "en")
  ?dst rdfs:label ?dstLabel . FILTER(LANG(?dstLabel) = "en")
}
LIMIT 20000
""",
        18,
    ),
    (
        "cities_admin_containment",
        "P131",
        """
SELECT ?srcLabel ?dstLabel ?sl WHERE {
  VALUES ?cls { wd:Q515 wd:Q1549591 wd:Q3957 }
  ?src wdt:P31 ?cls ; wikibase:sitelinks ?sl ; wdt:P131 ?dst .
  FILTER(?sl >= 22)
  ?src rdfs:label ?srcLabel . FILTER(LANG(?srcLabel) = "en")
  ?dst rdfs:label ?dstLabel . FILTER(LANG(?dstLabel) = "en")
}
LIMIT 30000
""",
        0,
    ),
    (
        "museums_hierarchy",
        "P131",
        """
SELECT ?srcLabel ?alt ?dstLabel ?sl WHERE {
  ?cls wdt:P279* wd:Q33506 .
  ?src wdt:P31 ?cls ; wikibase:sitelinks ?sl .
  { ?src wdt:P131 ?dst } UNION { ?src wdt:P276 ?dst } UNION { ?src wdt:P17 ?dst }
  FILTER(?sl >= 12)
  ?src rdfs:label ?srcLabel . FILTER(LANG(?srcLabel) = "en")
  ?dst rdfs:label ?dstLabel . FILTER(LANG(?dstLabel) = "en")
  OPTIONAL { ?src skos:altLabel ?alt . FILTER(LANG(?alt) = "en") }
}
LIMIT 30000
""",
        18,
    ),
    (
        "theatres_and_opera_houses",
        "P131",
        """
SELECT ?srcLabel ?alt ?dstLabel ?sl WHERE {
  ?cls wdt:P279* wd:Q24354 .
  ?src wdt:P31 ?cls ; wikibase:sitelinks ?sl .
  { ?src wdt:P131 ?dst } UNION { ?src wdt:P276 ?dst } UNION { ?src wdt:P17 ?dst }
  FILTER(?sl >= 10)
  ?src rdfs:label ?srcLabel . FILTER(LANG(?srcLabel) = "en")
  ?dst rdfs:label ?dstLabel . FILTER(LANG(?dstLabel) = "en")
  OPTIONAL { ?src skos:altLabel ?alt . FILTER(LANG(?alt) = "en") }
}
LIMIT 20000
""",
        20,
    ),
    (
        "notable_artworks_collection",
        "P195",
        """
SELECT ?srcLabel ?dstLabel ?sl WHERE {
  ?src wdt:P31 wd:Q3305213 ; wikibase:sitelinks ?sl ; wdt:P195 ?dst .
  FILTER(?sl >= 22)
  ?src rdfs:label ?srcLabel . FILTER(LANG(?srcLabel) = "en")
  ?dst rdfs:label ?dstLabel . FILTER(LANG(?dstLabel) = "en")
}
LIMIT 6000
""",
        22,
    ),
    (
        "central_institutions_hq",
        "P159",
        """
SELECT ?srcLabel ?alt ?dstLabel ?sl WHERE {
  VALUES ?cls { wd:Q66344 wd:Q484652 wd:Q1335818 }
  ?src wdt:P31 ?cls ; wikibase:sitelinks ?sl ; wdt:P159 ?dst .
  FILTER(?sl >= 12)
  ?src rdfs:label ?srcLabel . FILTER(LANG(?srcLabel) = "en")
  ?dst rdfs:label ?dstLabel . FILTER(LANG(?dstLabel) = "en")
  OPTIONAL { ?src skos:altLabel ?alt . FILTER(LANG(?alt) = "en") }
}
LIMIT 6000
""",
        22,
    ),
    (
        "dishes_composition",
        "P527",
        """
SELECT ?srcLabel ?alt ?dstLabel ?sl WHERE {
  {
    VALUES ?cls { wd:Q746549 wd:Q19861951 wd:Q1778821 }
    ?src wdt:P31 ?cls .
  } UNION {
    ?src wdt:P279* wd:Q746549 .
  }
  ?src wikibase:sitelinks ?sl .
  { ?src wdt:P527 ?dst } UNION { ?src wdt:P186 ?dst } UNION { ?src wdt:P279 ?dst }
  FILTER(?sl >= 8)
  ?src rdfs:label ?srcLabel . FILTER(LANG(?srcLabel) = "en")
  ?dst rdfs:label ?dstLabel . FILTER(LANG(?dstLabel) = "en")
  OPTIONAL { ?src skos:altLabel ?alt . FILTER(LANG(?alt) = "en") }
}
LIMIT 24000
""",
        24,
    ),
    (
        "beverages_composition",
        "P527",
        """
SELECT ?srcLabel ?alt ?dstLabel ?sl WHERE {
  {
    VALUES ?cls { wd:Q13233 wd:Q40050 wd:Q30756 wd:Q8495 }
    ?src wdt:P31 ?cls .
  } UNION {
    ?src wdt:P279* wd:Q13233 .
  } UNION {
    ?src wdt:P279* wd:Q40050 .
  }
  ?src wikibase:sitelinks ?sl .
  { ?src wdt:P527 ?dst } UNION { ?src wdt:P186 ?dst } UNION { ?src wdt:P279 ?dst }
  FILTER(?sl >= 5)
  ?src rdfs:label ?srcLabel . FILTER(LANG(?srcLabel) = "en")
  ?dst rdfs:label ?dstLabel . FILTER(LANG(?dstLabel) = "en")
  OPTIONAL { ?src skos:altLabel ?alt . FILTER(LANG(?alt) = "en") }
}
LIMIT 15000
""",
        24,
    ),
]


def add_edge(edges: dict[tuple[str, str, str], int], s_raw: str, d_raw: str, rel_code: str, w: int) -> None:
    src = norm_label(s_raw)
    dst = norm_label(d_raw)
    if not src or not dst or src == dst:
        return
    s_vars = [src]
    if src.startswith("the ") and len(src) > 7:
        s_vars.append(src[4:])
    d_vars = [dst]
    if dst.startswith("the ") and len(dst) > 7:
        d_vars.append(dst[4:])
    for sv in s_vars:
        for dv in d_vars:
            if sv and dv and sv != dv and len(sv) >= 3 and len(dv) >= 3:
                key = (sv, rel_code, dv)
                if w > edges.get(key, 0):
                    edges[key] = w


def main() -> None:
    cache_dir = Path("target/wikidata_sparql_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    edges: dict[tuple[str, str, str], int] = {}
    if OUT_PATH.exists():
        for ln in OUT_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = ln.split("\t")
            if len(parts) == 4:
                try:
                    edges[(parts[0], parts[1], parts[2])] = int(parts[3])
                except ValueError:
                    pass
    for qname, rel_code, sparql, min_w in QUERIES:
        qhash = hashlib.sha256(sparql.encode("utf-8")).hexdigest()[:12]
        cfile = cache_dir / f"{qname}_{qhash}.tsv"
        if cfile.exists():
            rows = []
            for ln in cfile.read_text(encoding="utf-8").splitlines():
                p = ln.split("\t")
                if len(p) == 4:
                    rows.append((p[0], p[1], p[2], p[3]))
            print(f"  [{qname}] loaded {len(rows)} cached rows")
        else:
            rows = run_sparql_tsv(qname, sparql)
            if rows:
                cfile.write_text(
                    "\n".join(f"{a}\t{b}\t{c}\t{d}" for a, b, c, d in rows),
                    encoding="utf-8",
                )
        for s_raw, alt_raw, d_raw, sl_raw in rows:
            try:
                w = min(999, max(7, min_w, int(float(sl_raw))))
            except ValueError:
                w = max(14, min_w)
            add_edge(edges, s_raw, d_raw, rel_code, w)
            if alt_raw and len(alt_raw) <= 44:
                add_edge(edges, alt_raw, d_raw, rel_code, max(7, min_w, w - 1))

    by_src: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for (src, rel, dst), w in edges.items():
        by_src.setdefault((src, rel), []).append((w, dst))

    selected: list[tuple[int, str, str, str]] = []
    for (src, rel), items in by_src.items():
        items.sort(key=lambda x: (-x[0], x[1]))
        for w, dst in items[:6]:
            selected.append((w, src, rel, dst))

    selected.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))

    lines: list[str] = []
    total_bytes = 0
    max_bytes = 4200 * 1024
    for w, src, rel, dst in selected:
        line = f"{src}\t{rel}\t{dst}\t{w}\n"
        b = len(line.encode("utf-8"))
        if total_bytes + b > max_bytes:
            break
        lines.append(line)
        total_bytes += b

    lines.sort()
    payload = "".join(lines).encode("utf-8")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    print(f"Wrote {len(lines)} edges ({len(payload)} bytes) to {OUT_PATH}")
    print(f"SHA256: {digest}")


if __name__ == "__main__":
    main()
