//! One conservative, dataset-blind admission function. No gold labels, question
//! types or relevance annotations are accepted. Unparsed prose stays verbatim.
use super::*;
pub fn modality(text: &str, tags: &[String]) -> Modality {
    let lower = text.to_ascii_lowercase();
    if tags.iter().any(|s| s == "modality:rejected")
        || ["decided against ", "rejected ", "ruled out "]
            .iter()
            .any(|s| lower.contains(s))
    {
        return Modality::Rejected;
    }
    if tags.iter().any(|s| s == "modality:hypothetical")
        || [
            "i wish ",
            "what if ",
            "if i ",
            "might ",
            "maybe ",
            "would like ",
            "plan to ",
            "planning to ",
        ]
        .iter()
        .any(|s| lower.contains(s))
        || lower.ends_with('?')
    {
        return Modality::Hypothetical;
    }
    if tags.iter().any(|s| {
        s.starts_with("preference:") || s.starts_with("gear:") || s == "modality:preference"
    }) || [
        " prefer ",
        " likes ",
        " dislikes ",
        "i like ",
        "i own ",
        "i use ",
    ]
    .iter()
    .any(|s| lower.contains(s))
    {
        return Modality::ActivePreference;
    }
    if tags.iter().any(|s| s == "modality:executed")
        || tags.iter().any(|s| s.starts_with("slot:"))
            && tags.iter().any(|s| {
                ["lit:", "num:", "date:", "id:"]
                    .iter()
                    .any(|p| s.starts_with(p))
            })
    {
        return Modality::Executed;
    }
    Modality::Unspecified
}
pub fn admit_statement(
    mut event: LogEvent,
    prior: &[LogEvent],
    prior_edges: &[LogEdge],
) -> (LogEvent, Vec<LogEdge>) {
    let original = event.text.trim();
    let lower = original.to_ascii_lowercase();
    for (phrase, date) in pragmatics::resolved_dates(original, event.source_date.as_deref()) {
        event
            .tags
            .push(format!("deixis:{}={date}", phrase.replace(' ', "-")));
    }
    // Only explicit state templates admit a slot. An unsupported implication
    // (e.g. shaking -> specific phobia) never becomes a medical fact.
    let mut parsed = false;
    for (relation, slot, preference) in [
        (" lives in ", "residence", false),
        (" moved to ", "residence", false),
        (" works at ", "employer", false),
        (" prefers ", "preference", true),
        (" likes ", "preference", true),
        (" owns ", "gear", true),
    ] {
        if let Some(at) = lower.find(relation) {
            let subject = original[..at]
                .trim_start_matches("Correction: ")
                .trim_start_matches("Actually, ")
                .trim();
            let subject = if lower.starts_with("correction: ") {
                original[12..at].trim()
            } else if lower.starts_with("actually, ") {
                original[10..at].trim()
            } else {
                subject
            };
            let value = original[at + relation.len()..]
                .trim()
                .trim_end_matches(['.', '!'])
                .trim();
            if subject.is_empty()
                || value.is_empty()
                || subject.len() > 120
                || subject.contains('=')
                || value.len() > 240
                || value.contains('?')
            {
                break;
            }
            if matches!(
                modality(original, &event.tags),
                Modality::Hypothetical | Modality::Rejected
            ) {
                break;
            }
            event.tags.push(format!("slot:{subject}:{slot}"));
            event.tags.push(format!("lit:{value}"));
            event
                .tags
                .push(format!("slot-value:{subject}:{slot}=lit:{value}"));
            event.tags.push(format!("symbol:{subject}"));
            event.tags.push(
                if preference {
                    "modality:preference"
                } else {
                    "modality:executed"
                }
                .into(),
            );
            parsed = true;
            break;
        }
    }
    match modality(original, &event.tags) {
        Modality::Hypothetical => event.tags.push("modality:hypothetical".into()),
        Modality::Rejected => event.tags.push("modality:rejected".into()),
        Modality::ActivePreference => event.tags.push("modality:preference".into()),
        Modality::Executed => event.tags.push("modality:executed".into()),
        Modality::Unspecified => {}
    }
    event.tags.sort();
    event.tags.dedup();
    let mut edges = Vec::new();
    let correcting = lower.starts_with("correction:") || lower.starts_with("actually,");
    let replacing = lower.contains(" moved to ");
    if parsed && (correcting || replacing) {
        let slots: BTreeMap<_, _> = event
            .tags
            .iter()
            .filter(|t| t.starts_with("slot:"))
            .map(|s| (s, ()))
            .collect();
        let earlier: Vec<_> = prior
            .iter()
            .filter(|old| {
                old.tags.iter().any(|s| slots.contains_key(s))
                    && !values(old).is_empty()
                    && values(old) != values(&event)
                    && !prior_edges
                        .iter()
                        .any(|edge| edge.kind.currency() && edge.to == old.id)
            })
            .collect();
        // Ambiguous history requires an explicit endpoint, never latest-wins.
        if earlier.len() == 1 {
            edges.push(LogEdge {
                from: event.id.clone(),
                to: earlier[0].id.clone(),
                kind: if correcting {
                    EdgeKind::Corrects
                } else {
                    EdgeKind::Supersedes
                },
                seq: event.seq,
            });
        }
    }
    (event, edges)
}
