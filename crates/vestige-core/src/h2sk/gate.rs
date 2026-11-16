//! A bounded exact-evidence verifier, not a general semantic truth oracle.
use super::*;
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateVerdict {
    Allow,
    Abstain { answer: String },
    FalseAbstention { controlling: Vec<String> },
    StaleLiteral { controlling: Vec<String> },
    SliceMismatch,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GateInput {
    pub draft: String,
    #[serde(default)]
    pub expected_receipt_hash: Option<String>,
}
pub fn verify_draft(receipt: &Receipt, options: &CompileOptions, input: &GateInput) -> GateVerdict {
    let query_hash = blake3::hash(&serde_json::to_vec(options).expect("serializable options"))
        .to_hex()
        .to_string();
    if !compile::verify_receipt(receipt)
        || receipt.query_hash != query_hash
        || input
            .expected_receipt_hash
            .as_ref()
            .is_some_and(|s| s != &receipt.hash)
    {
        return GateVerdict::SliceMismatch;
    }
    let touches = options
        .symbols
        .iter()
        .any(|s| receipt.tally.contains_key(s));
    if !options.symbols.is_empty() && !touches {
        return GateVerdict::Abstain {
            answer: "not mentioned; no information available".into(),
        };
    }
    let query = options.query.to_lowercase();
    let draft = input.draft.to_lowercase();
    let current = ["currently", "now", "today", "current"]
        .iter()
        .any(|word| pragmatics::whole(&query, word));
    let historical = ["previous", "initial", "originally", "used to", "before"]
        .iter()
        .any(|word| pragmatics::whole(&query, word));
    let in_view = |id: &str| receipt.section.iter().any(|section| section.id == id);
    let controlling: Vec<_> = receipt
        .currency
        .iter()
        .filter_map(|row| row.apex.as_ref().filter(|id| in_view(id)).cloned())
        .collect();
    if current && !historical {
        for row in &receipt.currency {
            if let Some(apex) = &row.apex {
                if !in_view(apex) {
                    continue;
                }
                let active = row.values.get(apex).cloned().unwrap_or_default();
                let active_said = active.iter().any(|v| {
                    pragmatics::whole(
                        &draft,
                        &v.split_once(':')
                            .map_or(v.as_str(), |(_, v)| v)
                            .to_lowercase(),
                    )
                });
                let stale_said = row
                    .values
                    .iter()
                    .filter(|(id, _)| *id != apex)
                    .flat_map(|(_, values)| values)
                    .filter(|v| !active.contains(v))
                    .any(|v| {
                        pragmatics::whole(
                            &draft,
                            &v.split_once(':')
                                .map_or(v.as_str(), |(_, v)| v)
                                .to_lowercase(),
                        )
                    });
                if stale_said && !active_said {
                    return GateVerdict::StaleLiteral {
                        controlling: vec![apex.clone()],
                    };
                }
            }
        }
    }
    let abstaining = [
        "not mentioned",
        "no information available",
        "not enough information",
        "do not have enough information",
        "don't have enough information",
    ]
    .iter()
    .any(|s| draft.contains(s));
    let preference = receipt
        .section
        .iter()
        .any(|s| in_view(&s.id) && s.modality == Modality::ActivePreference);
    let advice = ["suggest", "any tips", "what should", "recommend"]
        .iter()
        .any(|s| query.contains(s));
    if abstaining && (touches && !controlling.is_empty() || advice && preference) {
        let controlling = if controlling.is_empty() {
            receipt
                .section
                .iter()
                .filter(|s| in_view(&s.id) && s.modality == Modality::ActivePreference)
                .map(|s| s.id.clone())
                .collect()
        } else {
            controlling
        };
        return GateVerdict::FalseAbstention { controlling };
    }
    GateVerdict::Allow
}
