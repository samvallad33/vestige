//! Stable lookup evidence and whole-card budgets. No model inference or cache claims.
use chrono::{DateTime, Utc};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const SERVER_RESERVE: usize = 256;

fn size(value: &Value) -> usize {
    value.to_string().len()
}

fn account(value: &mut Value) {
    for _ in 0..3 {
        value["tokensUsed"] = json!((size(value) + SERVER_RESERVE).div_ceil(4));
    }
}

fn role(card: &Value) -> &'static str {
    if card.get("validFrom").is_none() || card.get("validUntil").is_none() {
        return "unknown";
    }
    let now = Utc::now();
    let parse = |key| {
        card.get(key)
            .and_then(Value::as_str)
            .and_then(|raw| DateTime::parse_from_rfc3339(raw).ok())
    };
    if parse("validFrom").is_some_and(|at| at > now) {
        "future"
    } else if parse("validUntil").is_some_and(|at| at <= now) {
        "historical"
    } else {
        "current"
    }
}

/// Preserve caller-visible masks; never reload omitted content from storage.
pub(crate) fn finish(
    mut response: Value,
    budget: Option<i32>,
    packet: bool,
    known: Option<&str>,
    boundary: &Value,
) -> Value {
    if packet {
        response["detailLevel"] = json!("brief");
        let mut cards: Vec<Value> = response["results"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|card| {
                let mut stable = serde_json::Map::new();
                for key in [
                    "id",
                    "content",
                    "nodeType",
                    "tags",
                    "source",
                    "sourceRecord",
                    "createdAt",
                    "updatedAt",
                    "validFrom",
                    "validUntil",
                ] {
                    if let Some(value) = card.get(key) {
                        stable.insert(key.into(), value.clone());
                    }
                }
                stable.insert("temporalState".into(), json!(role(card)));
                Value::Object(stable)
            })
            .collect();
        cards.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
        response["results"] = json!(cards);
        // Diagnostics and changing rank scores cannot enter the stable packet.
        for field in [
            "associations",
            "contextReinstatement",
            "competitionSuppressed",
            "learningModeDetected",
            "fusionMode",
        ] {
            response.as_object_mut().unwrap().remove(field);
        }
        response["packetVersion"] = json!(1);
        response["evidenceIncomplete"] = json!(
            response
                .get("expandable")
                .and_then(Value::as_array)
                .is_some_and(|ids| !ids.is_empty())
        );
    }

    if let Some(raw) = budget {
        let limit = raw.clamp(100, 100_000) as usize;
        let usable = limit * 4 - SERVER_RESERVE;
        response["tokenBudgetLimit"] = json!(limit);
        response["budgetUnit"] = json!("utf8_bytes_div_4_ceiling");
        // Keep receipts by ID instead of embedding a second evidence copy.
        response["detailLevel"] = json!("brief");
        // Reserve packet identity before choosing cards so hashing cannot overflow.
        if packet {
            response["packetId"] = json!("0".repeat(64));
            response["notModified"] = json!(false);
        }
        account(&mut response);
        if size(&response) > usable {
            response["evidenceIncomplete"] = json!(true);
            for field in [
                "associations",
                "contextReinstatement",
                "competitionSuppressed",
                "learningModeDetected",
                "hint",
                "expandable",
                "tokenBudgetUsed",
                "tokenBudget",
                "fusionMode",
            ] {
                response.as_object_mut().unwrap().remove(field);
            }
            // A known dissent group is indivisible. If it cannot fit with the
            // response, omit its evidence rather than returning only one side.
            if response.get("contradictionProtected").is_some() {
                response["results"] = json!([]);
                response
                    .as_object_mut()
                    .unwrap()
                    .remove("contradictionProtected");
            }
        }
        loop {
            response["total"] = json!(response["results"].as_array().map_or(0, Vec::len));
            account(&mut response);
            if size(&response) <= usable {
                break;
            }
            let removed = response["results"].as_array_mut().and_then(Vec::pop);
            if removed.is_none() {
                // An oversized query/scope is omitted whole, never string-sliced.
                response = json!({"results":[], "truncated":true, "detailLevel":"brief", "tokenBudgetLimit":limit,
                    "budgetUnit":"utf8_bytes_div_4_ceiling"});
                break;
            }
            response["evidenceIncomplete"] = json!(true);
        }
        account(&mut response);
    }

    // Only complete packets can be acknowledged/reused. A truncated reply must
    // be refreshed at a larger budget rather than cached as complete knowledge.
    if packet && response["evidenceIncomplete"] == false {
        let payload = json!({"version":1, "boundary":boundary, "results":response["results"],
                             "dissent":response.get("contradictionProtected")});
        let id = format!("{:x}", Sha256::digest(payload.to_string().as_bytes()));
        response["packetId"] = json!(id);
        response["notModified"] = json!(known == Some(id.as_str()));
        if known == Some(id.as_str()) {
            response["results"] = json!([]);
            response["total"] = json!(0);
        }
    } else if packet {
        response.as_object_mut().unwrap().remove("packetId");
        response.as_object_mut().unwrap().remove("notModified");
    }
    if budget.is_some() {
        account(&mut response);
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    fn response() -> Value {
        json!({"query":"policy", "scope":"user", "results":[
            {"id":"b", "content":"second", "combinedScore":0.8, "retentionStrength":0.5},
            {"id":"a", "content":"first", "combinedScore":0.7, "retentionStrength":0.6}], "total":2})
    }
    #[test]
    fn packet_is_stable_across_rank_and_score_changes() {
        let first = finish(response(), None, true, None, &json!({"scope":"user"}));
        let mut other = response();
        other["results"].as_array_mut().unwrap().reverse();
        other["results"][0]["combinedScore"] = json!(0.1);
        let second = finish(
            other,
            None,
            true,
            first["packetId"].as_str(),
            &json!({"scope":"user"}),
        );
        assert_eq!(first["packetId"], second["packetId"]);
        assert_eq!(second["notModified"], true);
        assert_eq!(second["results"], json!([]));
    }
    #[test]
    fn content_and_boundary_changes_force_refresh() {
        let first = finish(response(), None, true, None, &json!({"scope":"user"}));
        let mut changed = response();
        changed["results"][0]["content"] = json!("corrected");
        for (value, boundary) in [
            (changed, json!({"scope":"user"})),
            (response(), json!({"scope":"project"})),
        ] {
            let next = finish(value, None, true, first["packetId"].as_str(), &boundary);
            assert_ne!(first["packetId"], next["packetId"]);
            assert_eq!(next["notModified"], false);
        }
    }
    #[test]
    fn budgets_hold_for_unicode_and_large_metadata() {
        for budget in [100, 101, 256, 500, 1000] {
            let mut value = response();
            value["query"] = json!("🍃".repeat(10000));
            let result = finish(value, Some(budget), true, None, &Value::Null);
            assert!(size(&result) + SERVER_RESERVE <= budget as usize * 4);
            assert!(result["evidenceIncomplete"] == true || result["truncated"] == true);
            assert!(result.get("packetId").is_none());
        }
    }
    #[test]
    fn incomplete_dissent_is_not_returned_as_one_sided_evidence() {
        let mut value = response();
        value["results"][0]["content"] = json!("large".repeat(2000));
        value["contradictionProtected"] = json!({"memoryIds":["a","b"]});
        let result = finish(value, Some(300), false, None, &Value::Null);
        assert_eq!(result["results"], json!([]));
        assert_eq!(result["evidenceIncomplete"], true);
    }
}
