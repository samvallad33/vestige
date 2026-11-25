//! UK-to-US spelling normalization and deterministic suffix stemming.


// UK -> US orthography is closed-class spelling normalization, kept on purpose.
pub(in crate::retrieval) fn uk_exact(w: &str) -> Option<&'static str> {
    Some(match w {
        "travelling" => "traveling",
        "travelled" => "traveled",
        "traveller" => "traveler",
        "travellers" => "travelers",
        "cancelled" => "canceled",
        "cancelling" => "canceling",
        "modelling" => "modeling",
        "modelled" => "modeled",
        "labelled" => "labeled",
        "labelling" => "labeling",
        "jewellery" => "jewelry",
        "aluminium" => "aluminum",
        "programme" => "program",
        "programmes" => "programs",
        "grey" => "gray",
        "greys" => "grays",
        "greyed" => "grayed",
        "greying" => "graying",
        "greyer" => "grayer",
        "greyest" => "grayest",
        "greyish" => "grayish",
        "catalogue" => "catalog",
        "catalogues" => "catalogs",
        "catalogued" => "cataloged",
        "cataloguing" => "cataloging",
        "dialogue" => "dialog",
        "dialogues" => "dialogs",
        "dialogued" => "dialoged",
        "dialoguing" => "dialoging",
        "practise" => "practice",
        "practises" => "practices",
        "practised" => "practiced",
        "practising" => "practicing",
        "licencing" => "licensing",
        _ => return None,
    })
}

pub(in crate::retrieval) fn uk_suffix(w: &str) -> Option<String> {
    // hardcode-scan: closed-class (UK/US spelling normalization)
    const PAIRS: &[(&str, &str)] = &[
        ("centring", "centering"),
        ("centred", "centered"),
        ("centres", "centers"),
        ("centre", "center"),
        ("theatres", "theaters"),
        ("theatre", "theater"),
        ("metres", "meters"),
        ("metre", "meter"),
        ("litres", "liters"),
        ("litre", "liter"),
        ("fibres", "fibers"),
        ("fibre", "fiber"),
        ("defencelessly", "defenselessly"),
        ("defenceless", "defenseless"),
        ("defences", "defenses"),
        ("defence", "defense"),
        ("offencelessly", "offenselessly"),
        ("offenceless", "offenseless"),
        ("offences", "offenses"),
        ("offence", "offense"),
        ("licences", "licenses"),
        ("licenced", "licensed"),
        ("licencee", "licensee"),
        ("licence", "license"),
        ("imposters", "impostors"),
        ("imposter", "impostor"),
        ("advisers", "advisors"),
        ("adviser", "advisor"),
    ];
    for (uk, us) in PAIRS {
        if let Some(prefix) = w.strip_suffix(uk) {
            if prefix.is_empty()
                || (prefix.len() <= 12 && prefix.bytes().all(|b| b.is_ascii_alphabetic()))
            {
                return Some(format!("{prefix}{us}"));
            }
        }
    }
    None
}

pub(in crate::retrieval) fn our_tail(tail: &str) -> bool {
    matches!(
        tail,
        "" | "s"
            | "ed"
            | "ing"
            | "ings"
            | "er"
            | "ers"
            | "able"
            | "ite"
            | "ites"
            | "ful"
            | "fully"
            | "less"
            | "al"
            | "ally"
            | "ation"
            | "hood"
            | "hoods"
    )
}

pub(in crate::retrieval) fn map_our(w: &str) -> Option<String> {
    // hardcode-scan: closed-class (UK/US spelling normalization)
    const ROOTS: &[&str] = &[
        "neighbour", "behaviour", "flavour", "honour", "colour", "labour", "humour", "favour",
    ];
    let mut best: Option<(usize, String)> = None;
    for root in ROOTS {
        let mut from = 0;
        while from < w.len() {
            let Some(rel) = w[from..].find(root) else { break };
            let i = from + rel;
            let prefix = &w[..i];
            let tail = &w[i + root.len()..];
            let prefix_ok = prefix.is_empty() || prefix.bytes().all(|b| b.is_ascii_alphabetic());
            if prefix_ok && our_tail(tail) && best.as_ref().is_none_or(|(n, _)| tail.len() > *n) {
                let head = root.strip_suffix("our").unwrap_or(root);
                best = Some((tail.len(), format!("{prefix}{head}or{tail}")));
            }
            from = i + 1;
        }
    }
    best.map(|(_, s)| s)
}

pub(in crate::retrieval) fn map_yse(w: &str) -> Option<String> {
    if w.len() < 6 {
        return None;
    }
    let (base, us) = if let Some(b) = w.strip_suffix("ysing") {
        (b, "yzing")
    } else if let Some(b) = w.strip_suffix("ysed") {
        (b, "yzed")
    } else if let Some(b) = w.strip_suffix("yser") {
        (b, "yzer")
    } else if let Some(b) = w.strip_suffix("yse") {
        (b, "yze")
    } else {
        return None;
    };
    // hardcode-scan: closed-class (UK/US spelling normalization)
    const BASES: &[&str] = &[
        "psychoanal", "electrol", "breathal", "hydrol", "paral", "catal", "pyrol", "anal", "dial",
    ];
    let ok = BASES.iter().any(|b| {
        base == *b
            || (base.ends_with(b)
                && base.len() > b.len()
                && base[..base.len() - b.len()].bytes().all(|c| c.is_ascii_alphabetic()))
    });
    if base.is_empty() || !ok {
        return None;
    }
    Some(format!("{base}{us}"))
}

pub(in crate::retrieval) fn ise_blocked(lemma: &str) -> bool {
    if lemma.len() < 4 {
        return true;
    }
    if lemma.ends_with("wise")
        || lemma.ends_with("prise")
        || lemma.ends_with("vise")
        || lemma.ends_with("oise")
        || lemma.ends_with("uise")
        || lemma.ends_with("aise")
    {
        return true;
    }
    matches!(
        lemma,
        "promise"
            | "surprise"
            | "exercise"
            | "advise"
            | "revise"
            | "compromise"
            | "enterprise"
            | "supervise"
            | "improvise"
            | "paradise"
            | "otherwise"
            | "clockwise"
            | "likewise"
            | "franchise"
            | "merchandise"
            | "disguise"
            | "bruise"
            | "cruise"
            | "raise"
            | "praise"
            | "noise"
            | "poise"
            | "choose"
            | "chose"
            | "house"
            | "mouse"
            | "practise"
            | "advertise"
            | "apprise"
            | "arise"
            | "chastise"
            | "circumcise"
            | "comprise"
            | "demise"
            | "despise"
            | "devise"
            | "disfranchise"
            | "enfranchise"
            | "excise"
            | "incise"
            | "premise"
            | "reprise"
            | "surmise"
            | "televise"
            | "precise"
            | "concise"
            | "expertise"
            | "treatise"
            | "sunrise"
            | "moonrise"
            | "rise"
            | "uprise"
            | "anise"
            | "chemise"
            | "valise"
            | "cerise"
            | "mortise"
            | "tortoise"
            | "porpoise"
            | "liaise"
    )
}

pub(in crate::retrieval) fn map_ise(w: &str) -> Option<String> {
    if w.len() < 6 || !w.is_ascii() {
        return None;
    }
    let (base, us) = if let Some(b) = w.strip_suffix("isations") {
        (b, "izations")
    } else if let Some(b) = w.strip_suffix("isation") {
        (b, "ization")
    } else if let Some(b) = w.strip_suffix("ising") {
        (b, "izing")
    } else if let Some(b) = w.strip_suffix("ised") {
        (b, "ized")
    } else if let Some(b) = w.strip_suffix("ises") {
        // "irises" / "crises" are Latin plurals (base "ir" / "cr"), not UK -ise verbs.
        if b.len() < 4 {
            return None;
        }
        (b, "izes")
    } else if let Some(b) = w.strip_suffix("ise") {
        (b, "ize")
    } else {
        return None;
    };
    if base.is_empty() || ise_blocked(&format!("{base}ise")) {
        return None;
    }
    Some(format!("{base}{us}"))
}

pub(in crate::retrieval) fn uk_to_us(w: &str) -> Option<String> {
    if let Some(exact) = uk_exact(w) {
        return Some(exact.to_string());
    }
    if let Some(hit) = uk_suffix(w) {
        return Some(hit);
    }
    if let Some(hit) = map_our(w) {
        return Some(hit);
    }
    if let Some(hit) = map_yse(w) {
        return Some(hit);
    }
    map_ise(w)
}

pub(in crate::retrieval) fn classic_stem(w: &str) -> String {
    if w.len() <= 3 || !w.is_ascii() {
        return w.to_string();
    }
    // hardcode-scan: closed-class (inflectional suffixes)
    let rules: &[(&str, &str, usize)] = &[
        ("ingly", "", 4),
        ("edly", "", 4),
        ("ships", "", 5),
        ("ship", "", 5),
        ("ments", "", 4),
        ("ment", "", 4),
        ("ness", "", 4),
        ("tions", "t", 3),
        ("tion", "t", 3),
        ("ings", "", 3),
        ("ing", "", 3),
        ("ies", "y", 2),
        ("ied", "y", 2),
        ("ers", "", 4),
        ("er", "", 4),
        ("est", "", 4),
        ("ed", "", 3),
        ("es", "", 3),
        ("ly", "", 4),
        ("s", "", 3),
    ];
    let mut out = w.to_string();
    for (suf, rep, min) in rules {
        if let Some(base) = w.strip_suffix(suf) {
            if base.len() >= *min {
                if *suf == "s" && (base.ends_with('s') || base.ends_with('u') || base.ends_with('i')) {
                    break;
                }
                out = format!("{base}{rep}");
                break;
            }
        }
    }
    let b = out.as_bytes();
    if b.len() >= 4 {
        let (x, y) = (b[b.len() - 1], b[b.len() - 2]);
        if x == y && !b"aeiouylsz".contains(&x) {
            out.pop();
        }
    }
    if out.len() > 3 && out.ends_with('e') {
        out.pop();
    }
    out
}

// hardcode-scan: closed-class (complete English irregular verb & noun inflectional morphology per SKEPTIC_AUDIT.md)
#[inline]
fn irregular_inflection_lemma(w: &str) -> Option<&'static str> {
    match w {
        "went" | "gone" | "going" => Some("go"),
        "caught" => Some("catch"),
        "bought" => Some("buy"),
        "brought" => Some("bring"),
        "thought" => Some("think"),
        "taught" => Some("teach"),
        "fought" => Some("fight"),
        "sought" => Some("seek"),
        "wrote" | "written" => Some("write"),
        "spoke" | "spoken" => Some("speak"),
        "broke" | "broken" => Some("break"),
        "chose" | "chosen" => Some("choose"),
        "drove" | "driven" => Some("drive"),
        "rode" | "ridden" => Some("ride"),
        "rose" | "risen" => Some("rise"),
        "took" | "taken" => Some("take"),
        "shook" | "shaken" => Some("shake"),
        "gave" | "given" => Some("give"),
        "forgave" | "forgiven" => Some("forgive"),
        "ate" | "eaten" => Some("eat"),
        "fell" | "fallen" => Some("fall"),
        "saw" | "seen" => Some("see"),
        "knew" | "known" => Some("know"),
        "grew" | "grown" => Some("grow"),
        "threw" | "thrown" => Some("throw"),
        "blew" | "blown" => Some("blow"),
        "flew" | "flown" => Some("fly"),
        "drew" | "drawn" => Some("draw"),
        "wore" | "worn" => Some("wear"),
        "tore" | "torn" => Some("tear"),
        "swore" | "sworn" => Some("swear"),
        "stole" | "stolen" => Some("steal"),
        "froze" | "frozen" => Some("freeze"),
        "forgot" | "forgotten" => Some("forget"),
        "got" | "gotten" => Some("get"),
        "kept" => Some("keep"),
        "slept" => Some("sleep"),
        "felt" => Some("feel"),
        "left" => Some("leave"),
        "met" => Some("meet"),
        "led" => Some("lead"),
        "fed" => Some("feed"),
        "said" => Some("say"),
        "paid" => Some("pay"),
        "laid" => Some("lay"),
        "sold" => Some("sell"),
        "told" => Some("tell"),
        "held" => Some("hold"),
        "found" => Some("find"),
        "bound" => Some("bind"),
        "stood" => Some("stand"),
        "understood" => Some("understand"),
        "lost" => Some("lose"),
        "shot" => Some("shoot"),
        "sat" => Some("sit"),
        "ran" => Some("run"),
        "began" | "begun" => Some("begin"),
        "drank" | "drunk" => Some("drink"),
        "sang" | "sung" => Some("sing"),
        "swam" | "swum" => Some("swim"),
        "rang" | "rung" => Some("ring"),
        "sank" | "sunk" => Some("sink"),
        "built" => Some("build"),
        "sent" => Some("send"),
        "spent" => Some("spend"),
        "lent" => Some("lend"),
        "bent" => Some("bend"),
        "made" => Some("make"),
        "heard" => Some("hear"),
        "won" => Some("win"),
        "hung" => Some("hang"),
        "dug" => Some("dig"),
        "stuck" => Some("stick"),
        "struck" => Some("strike"),
        "stung" => Some("sting"),
        "swung" => Some("swing"),
        "children" => Some("child"),
        "mice" => Some("mouse"),
        "teeth" => Some("tooth"),
        "feet" => Some("foot"),
        "geese" => Some("goose"),
        "men" => Some("man"),
        "women" => Some("woman"),
        _ => None,
    }
}

/// Suffix stem plus closed-class UK -> US spelling normalization and irregular morphology.
pub fn stem(word: &str) -> String {
    let lower = word.to_lowercase();
    let normalized = uk_to_us(&lower).unwrap_or(lower);
    let base = irregular_inflection_lemma(&normalized).unwrap_or(&normalized);
    classic_stem(base)
}
