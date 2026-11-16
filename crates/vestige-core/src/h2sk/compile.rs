use super::*;
use std::collections::{BTreeSet, VecDeque};

fn hash<T: Serialize>(value: &T) -> String {
    blake3::hash(&serde_json::to_vec(value).expect("serializable receipt"))
        .to_hex()
        .to_string()
}
pub fn verify_receipt(receipt: &Receipt) -> bool {
    let mut copy = receipt.clone();
    copy.hash.clear();
    receipt.hash == hash(&copy)
}
pub fn compile_log_prefix(prefix: &LogPrefix, options: &CompileOptions) -> Result<Receipt, String> {
    if prefix.events.len() > 100_000 || prefix.edges.len() > 800_000 {
        return Err("compile_resource_limit".into());
    }
    // Compile the selected sources plus their exact slot history. Current
    // state cannot be decided from an isolated old hit when its controlling
    // successor was not lexically selected. Nothing pulls an unrelated slot.
    let projected;
    let prefix = if let Some(view) = &options.view_ids {
        let slots: BTreeSet<_> = prefix
            .events
            .iter()
            .filter(|event| view.contains(&event.id))
            .flat_map(|event| event.tags.iter().filter_map(|tag| slot_key(event, tag)))
            .collect();
        let events: Vec<_> = prefix
            .events
            .iter()
            .filter(|event| {
                view.contains(&event.id)
                    || event
                        .tags
                        .iter()
                        .filter_map(|tag| slot_key(event, tag))
                        .any(|key| slots.contains(&key))
            })
            .cloned()
            .collect();
        let ids: BTreeSet<_> = events.iter().map(|event| event.id.as_str()).collect();
        let edges = prefix
            .edges
            .iter()
            .filter(|edge| ids.contains(edge.from.as_str()) && ids.contains(edge.to.as_str()))
            .cloned()
            .collect();
        projected = LogPrefix {
            events,
            edges,
            offset: prefix.offset,
            source_root: prefix.source_root.clone(),
            previous_receipt_hash: prefix.previous_receipt_hash.clone(),
        };
        &projected
    } else {
        prefix
    };
    let mut ids = BTreeMap::new();
    for event in &prefix.events {
        if event.seq > prefix.offset
            || event.id.is_empty()
            || event.text.len() > 64 * 1024
            || ids.insert(event.id.clone(), event).is_some()
        {
            return Err("invalid_log_prefix".into());
        }
    }
    let mut edges = Vec::new();
    for edge in &prefix.edges {
        let (Some(a), Some(b)) = (ids.get(&edge.from), ids.get(&edge.to)) else {
            return Err("edge_endpoint_missing".into());
        };
        if edge.seq > prefix.offset || edge.seq < a.seq || edge.seq < b.seq || edge.from == edge.to
        {
            return Err("invalid_edge_sequence".into());
        }
        // Existing general-purpose native edges remain valid inclusion
        // evidence. Only exact slot/value members below can turn them into a
        // currency row; no slot or value is inferred to fill that gap.
        edges.push(edge.clone());
    }
    edges.sort_by(|a, b| {
        a.seq
            .cmp(&b.seq)
            .then_with(|| a.from.cmp(&b.from))
            .then_with(|| a.to.cmp(&b.to))
    });
    let mut slots: BTreeMap<String, Vec<&LogEvent>> = BTreeMap::new();
    let mut tally = BTreeMap::new();
    for event in &prefix.events {
        for tag in event.tags.iter().collect::<BTreeSet<_>>() {
            if tag.starts_with("symbol:") {
                *tally.entry(tag.clone()).or_default() += 1;
            }
            if tag.starts_with("slot:")
                && !values(event).is_empty()
                && admit::modality(&event.text, &event.tags).authoritative()
                && let Some(key) = slot_key(event, tag)
            {
                slots.entry(key).or_default().push(event);
            }
        }
    }
    let mut currency = Vec::new();
    let mut frustration = Vec::new();
    for (slot, events) in slots {
        let members: BTreeSet<_> = events.iter().map(|e| e.id.clone()).collect();
        let links: Vec<_> = edges
            .iter()
            .filter(|e| e.kind.currency() && members.contains(&e.from) && members.contains(&e.to))
            .collect();
        let targets: BTreeSet<_> = links.iter().map(|e| e.to.clone()).collect();
        // Detect cycles even in disconnected components, never silently choose
        // the newest timestamp as a replacement for a recorded edge.
        let mut indegree: BTreeMap<String, usize> =
            members.iter().map(|id| (id.clone(), 0)).collect();
        let mut onward: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for edge in &links {
            *indegree.get_mut(&edge.to).unwrap() += 1;
            onward.entry(&edge.from).or_default().push(&edge.to);
        }
        let mut queue: VecDeque<_> = indegree
            .iter()
            .filter(|(_, n)| **n == 0)
            .map(|(id, _)| id.clone())
            .collect();
        let mut visited = 0;
        while let Some(at) = queue.pop_front() {
            visited += 1;
            for to in onward.get(at.as_str()).into_iter().flatten() {
                let n = indegree.get_mut(*to).unwrap();
                *n -= 1;
                if *n == 0 {
                    queue.push_back((*to).to_string());
                }
            }
        }
        if visited != members.len() {
            return Err("currency_cycle".into());
        }
        let mut terminals: Vec<_> = events
            .iter()
            .filter(|e| !targets.contains(&e.id))
            .copied()
            .collect();
        terminals.sort_by(|a, b| b.seq.cmp(&a.seq).then_with(|| a.id.cmp(&b.id)));
        let distinct: BTreeSet<_> = terminals.iter().map(|e| values(e)).collect();
        let apex = if distinct.len() == 1 {
            terminals.first().map(|e| e.id.clone())
        } else {
            None
        };
        let horosphere: Vec<_> = apex
            .as_ref()
            .map(|id| {
                links
                    .iter()
                    .filter(|e| &e.from == id)
                    .map(|e| e.to.clone())
                    .collect()
            })
            .unwrap_or_default();
        let off_spine = members
            .iter()
            .filter(|id| Some(*id) != apex.as_ref() && !horosphere.contains(id))
            .cloned()
            .collect();
        if apex.is_none() {
            frustration.push(format!("NoUniqueApex:{slot}"));
        }
        currency.push(CurrencyRow {
            slot,
            status: if apex.is_some() {
                CurrencyStatus::UniqueApex
            } else {
                CurrencyStatus::NoUniqueApex
            },
            apex,
            contenders: terminals.iter().map(|e| e.id.clone()).collect(),
            horosphere,
            off_spine,
            values: events.iter().map(|e| (e.id.clone(), values(e))).collect(),
        });
    }
    let mut overruled = BTreeMap::new();
    for row in &currency {
        if let Some(apex) = &row.apex {
            let mut pending = VecDeque::from([apex.clone()]);
            let mut seen = BTreeSet::new();
            while let Some(at) = pending.pop_front() {
                if !seen.insert(at.clone()) {
                    continue;
                }
                for edge in edges.iter().filter(|edge| {
                    edge.kind.currency() && edge.from == at && row.values.contains_key(&edge.to)
                }) {
                    overruled.insert(edge.to.clone(), apex.clone());
                    pending.push_back(edge.to.clone());
                }
            }
        }
    }
    let mut section = Vec::new();
    let mut impulse = Vec::new();
    for event in &prefix.events {
        let mut banner = None;
        for row in &currency {
            if row.apex.as_ref() == Some(&event.id) {
                banner = Some(format!(
                    "[SHEPARD STATUS: CONTROLLING CURRENT LAW AS OF {}]",
                    event.source_date.as_deref().unwrap_or("UNKNOWN DATE")
                ));
            } else if overruled.get(&event.id) == row.apex.as_ref()
                && row.apex.is_some()
                && let Some(apex) = &row.apex
            {
                let date = ids[apex].source_date.as_deref().unwrap_or("UNKNOWN DATE");
                banner = Some(format!(
                    "[SHEPARD STATUS: OVERRULED BY {apex} ON {date} — CITE ONLY FOR PREVIOUS / INITIAL / ORIGINALLY QUERIES]"
                ));
            }
        }
        let mut deltas = Vec::new();
        if let Some(from) = &event.source_date {
            if let Some(today) = &options.today
                && let Some(delta) = pragmatics::delta(from, today)
            {
                deltas.push(delta);
            }
            // A bounded exact pairwise matrix for the retrieved five-doc view.
            for other in prefix
                .events
                .iter()
                .take(5)
                .filter(|other| other.id != event.id)
            {
                if let Some(to) = &other.source_date
                    && let Some(delta) = pragmatics::delta(from, to)
                {
                    deltas.push(delta);
                }
            }
        }
        let modality = admit::modality(&event.text, &event.tags);
        section.push(Section {
            id: event.id.clone(),
            quote: event.text.clone(),
            canonical: pragmatics::canonicalize(
                &event.text,
                event.source_date.as_deref(),
                &options.bindings,
            ),
            source_date: event.source_date.clone(),
            modality,
            banner,
            deltas,
        });
        let ancestor_depth = currency.iter().find_map(|row| {
            if row.apex.as_ref() == Some(&event.id) {
                Some(0)
            } else if row.horosphere.contains(&event.id) {
                Some(1)
            } else {
                None
            }
        });
        let basis = event
            .stability_q
            .filter(|q| *q > 0)
            .map(|q| ("logged_stability", q as f64 / 4294967296.0))
            .or_else(|| {
                ancestor_depth.map(|depth| ("recorded_ancestor_depth", f64::from(depth + 1)))
            });
        if let Some((basis, tau)) = basis {
            let elapsed = event
                .source_date
                .as_deref()
                .zip(options.today.as_deref())
                .and_then(|(a, b)| pragmatics::delta(a, b))
                .map(|d| d.days.max(0) as f64)
                .unwrap_or(0.0);
            impulse.push(Impulse {
                id: event.id.clone(),
                basis: basis.into(),
                tau_micros_days: (tau * 1e6).round() as u64,
                activation_ppm: (libm::exp(-elapsed / tau) * 1e6).round() as u32,
                previous_receipt_hash: prefix.previous_receipt_hash.clone(),
            });
        }
    }
    let mut source = Vec::new();
    for event in &prefix.events {
        source.extend_from_slice(blake3::hash(event.text.as_bytes()).as_bytes());
    }
    let mut receipt = Receipt {
        version: 1,
        offset: prefix.offset,
        source_root: prefix.source_root.clone(),
        query_hash: hash(options),
        previous_receipt_hash: prefix.previous_receipt_hash.clone(),
        tally,
        currency,
        section,
        frustration,
        inclusion: edges,
        impulse,
        hologram: spectral::project(&source),
        gate: GateVerdict::Allow,
        hash: String::new(),
    };
    if !options.symbols.is_empty()
        && !options
            .symbols
            .iter()
            .any(|s| receipt.tally.contains_key(s))
    {
        receipt.gate = GateVerdict::Abstain {
            answer: "not mentioned; no information available".into(),
        };
    }
    receipt.hash = hash(&receipt);
    Ok(receipt)
}

/// Source-carrying reader view. Banners and dates precede their source, never
/// silently replace a source quote. The stored receipt itself is untruncated.
pub fn prepare_payload(receipt: &Receipt, ids: &BTreeSet<String>, max_bytes: usize) -> String {
    #[derive(Serialize)]
    struct ReaderRow<'a> {
        id: &'a str,
        text: &'a str,
        banner: Option<&'a str>,
        modality: &'a str,
        date: Option<&'a str>,
        canonical: Option<&'a str>,
        deltas: &'a [pragmatics::Delta],
    }
    let mut rows = Vec::new();
    let mut visible = ids.clone();
    for row in &receipt.currency {
        if row.values.keys().any(|id| ids.contains(id)) {
            if let Some(apex) = &row.apex {
                visible.insert(apex.clone());
            }
            visible.extend(row.horosphere.iter().cloned());
            rows.push(format!(
                "[COMPILED STATE TRAJECTORY] {} {}",
                row.slot,
                serde_json::to_string(&serde_json::json!({"status":row.status,"current":row.apex,"previous":row.horosphere,"unresolved":row.contenders})).unwrap()
            ));
        }
    }
    for section in receipt
        .section
        .iter()
        .filter(|row| visible.contains(&row.id))
    {
        rows.push(
            serde_json::to_string(&ReaderRow {
                id: &section.id,
                text: &section.quote,
                banner: section.banner.as_deref(),
                modality: section.modality.banner(),
                date: section.source_date.as_deref(),
                canonical: if section.canonical != section.quote {
                    Some(&section.canonical)
                } else {
                    None
                },
                deltas: &section.deltas,
            })
            .unwrap(),
        );
    }
    let mut output = format!(
        "H2SK receipt={} offset={} source={}\n",
        receipt.hash, receipt.offset, receipt.source_root
    );
    for row in rows {
        if output.len() + row.len() < max_bytes {
            output.push_str(&row);
            output.push('\n');
        }
    }
    output
}
