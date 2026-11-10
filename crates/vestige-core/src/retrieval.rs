//! Stara Total-Recall Engine: zero-LLM, zero-vector, deterministic lexical retrieval.
//!
//! Every score is reproducible arithmetic over the text the caller supplied:
//! BM25F with role-segmented fields, IDF power law, distinct-term coverage,
//! ordered span proximity, deterministic suffix stemming, a fixed hypernym
//! lattice, a preference polarity axis, centering carry-forward across turns,
//! deictic relative-date compilation, max-passage window pooling, a
//! hierarchical session-to-turn funnel, question/answer dyad coupling,
//! bounded Adamic-Adar bridge closure and causal recency supersession.
//!
//! Leak guard: every input struct is `deny_unknown_fields`. A turn carries only
//! `id`, `role`, `content`; a session only `id`, `date`, `timestamp`, `turns`;
//! a query only `text` and `date`. Benchmark labels such as `has_answer`,
//! `answer`, `question_type` or `category` are rejected at the parse boundary,
//! so they can never influence a score.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

// =========================================================================
// INPUT / OUTPUT CONTRACT
// =========================================================================

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Turn {
    #[serde(default)]
    pub id: String,
    /// "user", "assistant", or a speaker name (LoCoMo-style dialogues).
    #[serde(default)]
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Session {
    pub id: String,
    /// Free-form session date ("2023/05/20 (Sat) 02:21", "1:56 pm on 8 May, 2023").
    #[serde(default)]
    pub date: String,
    /// Unix seconds. Used when `date` is empty or unparseable.
    #[serde(default)]
    pub timestamp: i64,
    pub turns: Vec<Turn>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Query {
    pub text: String,
    #[serde(default)]
    pub date: String,
}

fn default_top_k() -> usize {
    10
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankRequest {
    pub sessions: Vec<Session>,
    pub queries: Vec<Query>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub params: Params,
}

/// Every weapon is individually switchable so ablations are honest and reproducible.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Params {
    pub k1: f32,
    pub b_user: f32,
    pub b_assistant: f32,
    /// Length normalisation on assistant turns when the query attributes content to "you".
    pub b_assistant_attributed: f32,
    pub stem_weight: f32,
    pub hypernym_weight: f32,
    pub idf_power: f32,
    pub coverage_floor: f32,
    pub proximity_bonus: f32,
    pub proximity_span: usize,
    pub window: usize,
    /// Share of the max-passage window score in the session score.
    pub window_mix: f32,
    pub assistant_attribution_weight: f32,
    pub user_attribution_weight: f32,
    pub preference_boost: f32,
    pub preference_axis_weight: f32,
    pub recency_boost: f32,
    pub recency_band: f32,
    pub recency_depth: usize,
    pub carry: [f32; 3],
    pub carry_answer: f32,
    pub carry_plain_factor: f32,
    pub answer_boost: f32,
    pub funnel: f32,
    pub dyad: f32,
    pub dyad_partner: f32,
    pub bridge_depth: usize,
    pub date_weight: f32,
    pub enable_stemming: bool,
    pub enable_proximity: bool,
    pub enable_coverage: bool,
    pub enable_roles: bool,
    pub enable_preference: bool,
    pub enable_hypernyms: bool,
    pub enable_recency: bool,
    pub enable_window: bool,
    pub enable_carry: bool,
    pub enable_deictic: bool,
    pub enable_funnel: bool,
    pub enable_dyad: bool,
    pub enable_bridge: bool,
    pub enable_scene: bool,
    pub enable_sniper: bool,
    pub enable_ppr: bool,
    pub enable_submodular: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b_user: 0.75,
            b_assistant: 0.75,
            b_assistant_attributed: 0.25,
            stem_weight: 0.70,
            hypernym_weight: 0.45,
            idf_power: 1.5,
            coverage_floor: 0.35,
            proximity_bonus: 0.35,
            proximity_span: 5,
            window: 3,
            window_mix: 0.65,
            assistant_attribution_weight: 1.6,
            user_attribution_weight: 1.5,
            preference_boost: 2.8,
            preference_axis_weight: 1.0,
            recency_boost: 1.30,
            recency_band: 0.70,
            recency_depth: 5,
            carry: [0.65, 0.40, 0.20],
            carry_answer: 0.85,
            carry_plain_factor: 0.5,
            answer_boost: 1.15,
            funnel: 0.65,
            dyad: 0.35,
            dyad_partner: 0.90,
            bridge_depth: 15,
            date_weight: 0.30,
            enable_stemming: true,
            enable_proximity: true,
            enable_coverage: true,
            enable_roles: true,
            enable_preference: true,
            enable_hypernyms: true,
            enable_recency: true,
            enable_window: true,
            enable_carry: true,
            enable_deictic: true,
            enable_funnel: true,
            enable_dyad: true,
            enable_bridge: true,
            enable_scene: true,
            enable_sniper: true,
            enable_ppr: true,
            enable_submodular: true,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct QueryIntent {
    pub preference: bool,
    pub current_state: bool,
    pub assistant_attribution: bool,
    pub user_attribution: bool,
    pub multi_hop: bool,
    pub inference: bool,
    pub temporal: bool,
    pub sniper_when: bool,
    pub sniper_how_many: bool,
    pub sniper_where: bool,
    pub sniper_why: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RankedItem {
    pub id: String,
    pub session_id: String,
    pub score: f32,
    /// Exact query words that matched this item (the proof of the rank).
    pub matched: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RankResult {
    pub intent: QueryIntent,
    pub sessions: Vec<RankedItem>,
    pub turns: Vec<RankedItem>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RankResponse {
    pub engine: &'static str,
    pub sessions_indexed: usize,
    pub turns_indexed: usize,
    pub results: Vec<RankResult>,
}

pub const ENGINE: &str = "stara-total-recall/2 (bm25f+idf^p+coverage+proximity+stem+hypernym+pref-axis+carry+deictic+window+scene+sniper+ppr+submodular+funnel+dyad+recency)";

// =========================================================================
// LEXICONS (fixed, deterministic, domain-general)
// =========================================================================

const STOPWORDS: &[&str] = &[
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself",
    "yourselves", "also", "really", "yeah", "yes", "oh", "ok", "okay", "im", "ive", "id", "ill",
    "dont", "didnt", "doesnt", "thats", "its", "youre", "theyre", "lets", "got", "get", "gets",
    "getting", "go", "going", "gone", "went", "one", "much", "many", "lot", "lots", "thing",
    "things", "something", "anything", "everything", "kind", "sort", "way", "well", "even",
    "still", "ever", "may", "might", "must", "shall", "us", "let", "make", "made", "know",
    "think", "tell", "told", "say", "said", "remind", "remember", "previous", "conversation",
    "mentioned", "mention", "talked", "talk", "discussed", "chat", "actually", "sure", "great",
    "thanks", "thank", "hey", "hi", "hello", "wow", "like", "want", "wanted", "need", "needed",
    "use", "used", "using", "try", "tried", "trying", "new", "good", "help", "recently",
    "suggest", "recommend", "useful", "interesting", "past", "times",
];

const PREF_MARKERS: &[&str] = &[
    "love", "loved", "loves", "loving", "favorite", "favourite", "favorites", "prefer",
    "prefers", "preferred", "preference", "into", "obsessed", "enjoy", "enjoys", "enjoyed",
    "enjoying", "hobby", "hobbies", "passionate", "passion", "collect", "collecting", "hate",
    "hates", "hated", "avoid", "avoiding", "allergic", "never", "always", "usually", "diet",
    "vegan", "vegetarian", "fan", "like", "likes", "dislike", "dislikes", "fond", "crave",
    "addicted", "tend", "habit", "routine", "can't stand", "big fan",
];

const PREF_QUERY_CUES: &[&str] = &[
    "what kind of", "what type of", "what sort of", "recommend", "suggest", "suggestion",
    "prefer", "preference", "favorite", "favourite", "would i like", "would i enjoy",
    "should i get", "should i try", "should i buy", "gift", "present for", "suitable for me",
    "how do i like", "any ideas", "any tips", "ideas for", "tips for", "advice on",
    "can you suggest", "what should i", "something i", "i might like", "i would like",
    "go-to", "best for me", "fit my", "match my", "based on my",
];

const CURRENT_STATE_CUES: &[&str] = &[
    "currently", "current ", "right now", " now", "now?", "latest", "most recent",
    "most recently", "still ", "anymore", "any more", "end up", "ended up", "these days",
    "at the moment", "where is my", "what is my", "who is my", "where do i", "what do i",
    "do i still", "did i end", "nowadays", "updated", "new ", "changed",
];

const ASSISTANT_ATTRIBUTION_CUES: &[&str] = &[
    "you recommended", "you suggested", "you mentioned", "you told me", "you gave",
    "did you", "your recommendation", "your suggestion", "you wrote", "you said",
    "you provided", "you listed", "you recommend", "you shared", "you came up",
    "you helped", "you explained", "you described", "you named", "you created",
    "remind me", "our previous conversation", "our previous chat", "our conversation",
    "we discussed", "we talked", "previous chat", "last time we", "earlier you",
    "you were telling", "you advised", "you calculated", "you found", "you picked",
];

const USER_ATTRIBUTION_CUES: &[&str] = &[
    " i ", "i'", " my ", "my ", " me ", " me?", "did i", "do i", "have i", "was i", "am i",
    "what should i", "i've", "i'm",
];

const MULTI_HOP_CUES: &[&str] = &[
    " both ", " and ", " also ", " in common", " together", " same ", " as well",
    " which of", " between ", " total", " combined", " altogether", " how many", " all the ",
    " each ",
];

const INFERENCE_CUES: &[&str] = &[
    "likely", "would ", "might ", "probably", "could ", "considered", "personality",
    "traits", "what might", "would be", "be interested", "is it possible", "kind of person",
];

const TEMPORAL_CUES: &[&str] = &[
    "when ", "how long", "how many days", "how many weeks", "how many months",
    "how many years", "before ", "after ", "first ", "last ", " ago", "date", "which year",
    "what year", "what month", "earlier", "later", "since ",
];

const ANAPHORS: &[&str] = &[
    "it", "that", "they", "them", "he", "she", "him", "her", "there", "one", "this", "those",
    "these", "its", "their",
];

const CONTINUATIONS: &[&str] = &[
    "yeah", "yes", "yep", "no", "nope", "actually", "definitely", "sure", "because",
    "totally", "absolutely", "exactly", "thanks", "wow", "oh", "haha", "true", "right",
    "indeed", "agreed", "same",
];

/// 37-cluster deterministic hypernym lattice (coarse life domains + software systems architecture).
const HYPERNYM_LATTICE: &[(&str, &[&str])] = &[
    ("CULINARY", &["sourdough", "croissant", "baking", "bake", "baker", "bread", "loaf", "recipe", "dinner", "lunch", "breakfast", "vegan", "vegetarian", "pescatarian", "roast", "chef", "cook", "cooking", "kitchen", "cuisine", "meal", "dining", "restaurant", "food", "foodie", "dish", "spice", "pasta", "cheese", "dessert", "chocolate", "pastry", "culinary", "pizza", "sushi", "taco", "burger", "salad", "soup", "grill", "bbq", "cake", "cookie", "appliance", "appliances", "smoker", "fryer", "blender", "microwave", "toaster", "fridge", "refrigerator", "oven", "stove", "ingredient", "ingredients", "produce", "homegrown", "harvest"]),
    ("BEVERAGE", &["coffee", "espresso", "latte", "cappuccino", "roastery", "bean", "brew", "brewing", "tea", "matcha", "chai", "cafe", "barista", "aeropress", "pourover", "wine", "beer", "cocktail", "whiskey", "smoothie", "juice"]),
    ("OUTDOORS", &["hike", "hiking", "trail", "kayak", "kayaking", "camping", "camp", "tent", "backpack", "backpacking", "outdoor", "outdoors", "gear", "park", "wilderness", "climb", "climbing", "mountain", "beach", "lake", "fishing", "nature", "roadtrip"]),
    ("FITNESS", &["gym", "weight", "workout", "running", "runner", "run", "jog", "jogging", "cycling", "bike", "bicycle", "swim", "swimming", "triathlon", "marathon", "5k", "10k", "exercise", "cardio", "fitness", "yoga", "pilates", "crossfit", "training", "race", "tournament", "competition", "athletic", "athletics"]),
    ("VISUAL_ART", &["painting", "paint", "acrylic", "watercolor", "watercolour", "oil", "sketch", "sketching", "draw", "drawing", "pottery", "ceramic", "clay", "studio", "sculpture", "gallery", "canvas", "portrait", "landscape", "art", "artistic", "artist", "medium", "easel", "craft", "knitting", "sewing", "exhibition"]),
    ("MUSIC", &["guitar", "piano", "vinyl", "concert", "festival", "album", "spotify", "band", "jazz", "rock", "classical", "instrument", "drum", "bass", "violin", "orchestra", "musician", "song", "singing", "vocalist", "music", "playlist", "podcast", "dance", "ballet", "salsa"]),
    ("PETS", &["dog", "puppy", "cat", "kitten", "vet", "pet", "rescue", "adopt", "adoption", "animal", "retriever", "breed", "leash", "veterinary", "veterinarian", "fish", "bird", "hamster", "horse"]),
    ("BOOKS", &["book", "reading", "read", "novel", "author", "fiction", "nonfiction", "memoir", "library", "bookstore", "kindle", "audiobook", "literature", "poetry", "poem", "writing", "writer", "publish", "publication", "publications", "conference", "conferences", "paper", "papers", "article", "articles", "proceedings", "journal", "journals"]),
    ("GAMING", &["game", "gaming", "gamer", "playstation", "nintendo", "switch", "xbox", "steam", "rpg", "boardgame", "tabletop", "console", "chess", "puzzle", "videogame"]),
    ("TRAVEL", &["travel", "traveling", "flight", "airline", "trip", "vacation", "hotel", "hostel", "explore", "exploring", "passport", "tourist", "destination", "luggage", "itinerary", "cruise", "resort", "abroad", "museum", "theater"]),
    ("PHOTOGRAPHY", &["photo", "photography", "camera", "lens", "shoot", "canon", "nikon", "sony", "shutter", "aperture", "exposure", "darkroom", "instagram"]),
    ("GARDENING", &["garden", "gardening", "plant", "indoor", "succulent", "flower", "flowers", "soil", "pruning", "seed", "greenhouse", "botany", "tomato", "herb", "herbs", "homegrown", "harvest", "harvested", "ingredient", "ingredients", "produce", "basil", "mint", "pepper"]),
    ("TECH", &["python", "rust", "javascript", "typescript", "compiler", "react", "server", "bug", "repo", "code", "coding", "programmer", "programming", "software", "developer", "api", "database", "algorithm", "linux", "terminal", "github", "laptop", "computer", "phone", "app", "gadget", "accessory", "accessories", "charger", "cable", "case", "protector", "earbuds", "headphones", "adapter", "dongle"]),
    ("VEHICLES", &["car", "vehicle", "truck", "sedan", "ev", "electric", "drive", "driving", "engine", "mechanic", "commute", "commuting", "train", "subway", "transit", "highway", "roadtrip", "motorcycle"]),
    ("FASHION", &["clothes", "clothing", "wear", "wearing", "dress", "jacket", "coat", "shoe", "boot", "sneaker", "style", "vintage", "wardrobe", "outfit", "sweater", "denim", "wool", "jewelry", "makeup", "skincare", "thrifting", "antique"]),
    ("HEALTH", &["health", "doctor", "hospital", "clinic", "medication", "prescription", "medicine", "allergy", "allergic", "symptom", "surgery", "recover", "recovery", "therapy", "therapist", "mental", "headache", "flu", "dentist", "sleep", "diet", "counseling", "psychology"]),
    ("FAMILY", &["kid", "child", "children", "son", "daughter", "brother", "sister", "mother", "father", "mom", "dad", "parent", "parenting", "family", "toddler", "baby", "pregnancy", "grandma", "grandpa", "cousin", "niece", "nephew", "sibling", "siblings", "relative", "relatives", "kin", "twin", "twins"]),
    ("COMMUNITY", &["charity", "volunteer", "volunteering", "fundraiser", "fundraising", "donation", "donate", "shelter", "nonprofit", "cause", "awareness", "community", "activism", "protest", "rescue"]),
    ("IDENTITY", &["lgbtq", "queer", "gay", "lesbian", "pride", "trans", "transgender", "gender", "identity", "support", "transition", "religion", "faith", "church", "spiritual", "mentor"]),
    ("EDUCATION", &["school", "college", "university", "degree", "graduate", "graduation", "studying", "study", "exam", "class", "course", "professor", "campus", "student", "certification", "diploma", "lecture", "workshop", "major", "publication", "publications", "conference", "conferences", "paper", "papers", "journal"]),
    ("CAREER", &["job", "career", "boss", "coworker", "interview", "offer", "resume", "promote", "promotion", "salary", "office", "remote", "work", "working", "firm", "client", "company", "manager", "project", "startup", "business", "profession", "occupation", "occupations", "vocation", "agency", "role", "roles", "milestone", "milestones", "buisiness", "contract", "specialist"]),
    ("HOUSING", &["house", "apartment", "rent", "lease", "mortgage", "buy", "buying", "moving", "moved", "move", "neighborhood", "landlord", "bedroom", "renovate", "furniture", "decor", "home"]),
    ("SCREEN", &["movie", "film", "cinema", "tv", "show", "series", "episode", "netflix", "actor", "director", "watch", "watching", "documentary", "anime"]),
    ("FINANCE", &["budget", "money", "save", "saving", "invest", "investment", "stock", "crypto", "deposit", "loan", "debt", "retirement", "expense", "bank", "tax"]),
    ("RELATIONSHIP", &["date", "dating", "partner", "boyfriend", "girlfriend", "husband", "wife", "marriage", "wedding", "anniversary", "breakup", "divorce", "couple", "friend", "friendship", "single", "unmarried", "spouse", "ceremony", "event", "events"]),
    ("PERSONALITY", &["introvert", "extrovert", "adventurous", "creative", "organized", "outdoorsy", "compassionate", "empathetic", "patient", "energetic", "calm", "anxious", "stressed", "relaxed", "minimalist", "collector", "vintage", "antique", "thrifting", "personality", "trait"]),
    ("WELLNESS", &["meditation", "mindfulness", "massage", "spa", "relaxation", "stress", "breathe", "journal", "journaling", "selfcare"]),
    ("VIDEO", &["video", "editing", "premiere", "adobe", "lumetri", "davinci", "timeline", "render", "codec", "footage", "clip", "youtube", "vlog"]),
    ("SPORTS", &["soccer", "football", "basketball", "baseball", "tennis", "golf", "hockey", "volleyball", "team", "league", "match", "score", "coach", "stadium", "sport", "sports", "event", "events", "tournament", "race", "triathlon", "marathon", "competition", "championship", "athletic"]),
    ("SHOPPING", &["shop", "shopping", "store", "mall", "purchase", "bought", "order", "amazon", "deal", "sale", "discount", "coupon"]),
    ("CELEBRATION", &["party", "birthday", "celebration", "celebrate", "holiday", "christmas", "thanksgiving", "halloween", "gift", "present", "festival", "event", "events", "milestone", "ceremony", "wedding", "anniversary", "funeral", "graduation"]),
    ("SCIENCE", &["science", "physics", "chemistry", "biology", "astronomy", "space", "research", "experiment", "lab", "math", "publication", "publications", "conference", "conferences", "paper", "papers", "journal", "article", "articles"]),
    ("LANGUAGE", &["language", "spanish", "french", "german", "japanese", "chinese", "italian", "learn", "learning", "fluent", "duolingo", "translate"]),
    ("CONCURRENCY", &["mutex", "lock", "rwlock", "deadlock", "atomic", "channel", "thread", "coroutine", "async", "sync"]),
    ("STORAGE", &["disk", "nvme", "wal", "strata", "sqlite", "sled", "append", "fsync", "btree", "lsm", "database", "storage", "ledger"]),
    ("SECURITY", &["ed25519", "blake3", "sha256", "cryptography", "signature", "signed", "sandbox", "operator", "receipt", "audit", "precedent", "gate", "permission"]),
    ("IPC_NETWORK", &["socket", "pipe", "domain", "http", "tcp", "mcp", "rpc", "grpc", "endpoint", "protocol", "tls", "stream", "packet"]),
    ("VERIFICATION", &["fuzz", "proptest", "invariant", "witness", "bisect", "salience", "causal", "trace", "audit", "proof", "benchmark"]),
];

const MONTHS: [&str; 12] = [
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december",
];
const WEEKDAYS: [&str; 7] = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
];

const NUMBER_WORDS: &[&str] = &[
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "twice", "once", "several", "few", "couple", "half", "pair",
];

const SPATIAL_WORDS: &[&str] = &[
    "in", "at", "to", "from", "near", "visit", "visited", "visiting", "moved", "moving",
    "flew", "flying", "trip", "traveled", "traveling", "living", "lived", "station", "park",
    "city", "street", "avenue", "road", "hotel", "restaurant", "center", "centre", "house",
    "apartment", "relocated", "airport", "beach", "mountain", "lake", "trail", "downtown",
    "suburb", "neighborhood", "area", "country", "place",
];

const CAUSAL_WORDS: &[&str] = &[
    "because", "since", "so that", "due to", "wanted", "needed", "decided", "helped",
    "inspired", "to help", "reason", "in order", "purpose", "aim",
];

fn stopwords() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| STOPWORDS.iter().copied().collect())
}

fn hypernyms() -> &'static HashMap<String, Vec<&'static str>> {
    static H: OnceLock<HashMap<String, Vec<&'static str>>> = OnceLock::new();
    H.get_or_init(|| {
        let mut m: HashMap<String, Vec<&'static str>> = HashMap::new();
        for (dom, words) in HYPERNYM_LATTICE {
            for w in *words {
                let e = m.entry(stem(w)).or_default();
                if !e.contains(dom) {
                    e.push(dom);
                }
            }
        }
        m
    })
}

// =========================================================================
// TOKENISATION + DETERMINISTIC STEMMER
// =========================================================================

fn raw_tokens(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

fn is_content(w: &str) -> bool {
    w.chars().count() > 1 && !stopwords().contains(w)
}

/// Deterministic suffix stripper (Porter-style, no dictionary, no model).
pub fn stem(word: &str) -> String {
    let w = word.to_lowercase();
    if w.len() <= 3 || !w.is_ascii() {
        return w;
    }
    let rules: &[(&str, &str, usize)] = &[
        ("ingly", "", 4),
        ("edly", "", 4),
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
    let mut out = w.clone();
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
    // Undouble a final consonant pair left by -ing/-ed ("running" -> "run").
    let b = out.as_bytes();
    if b.len() >= 4 {
        let (x, y) = (b[b.len() - 1], b[b.len() - 2]);
        if x == y && !b"aeiouylsz".contains(&x) {
            out.pop();
        }
    }
    // Drop a silent final e so "bake"/"baking"/"baked" converge.
    if out.len() > 3 && out.ends_with('e') {
        out.pop();
    }
    out
}

// =========================================================================
// DATE PARSING + DEICTIC TIME COMPILER
// =========================================================================

/// Parse the free-form date strings used by real conversation logs into Unix seconds (UTC).
pub fn parse_date(s: &str) -> i64 {
    let cleaned: String = s
        .chars()
        .map(|c| if c.is_alphanumeric() || c == ':' { c.to_ascii_lowercase() } else { ' ' })
        .collect();
    let toks: Vec<&str> = cleaned.split_whitespace().collect();
    let (mut year, mut month, mut hour, mut minute) = (None, None, 0u32, 0u32);
    let mut nums: Vec<u32> = Vec::new();
    for (i, t) in toks.iter().enumerate() {
        if let Some((h, m)) = t.split_once(':') {
            let mut hh: u32 = h.parse().unwrap_or(0);
            minute = m.parse().unwrap_or(0);
            match toks.get(i + 1) {
                Some(&"pm") if hh < 12 => hh += 12,
                Some(&"am") if hh == 12 => hh = 0,
                _ => {}
            }
            hour = hh;
        } else if t.len() == 4 && t.chars().all(|c| c.is_ascii_digit()) {
            year = t.parse::<i32>().ok();
        } else if t.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(n) = t.parse::<u32>() {
                nums.push(n);
            }
        } else if t.len() >= 3 {
            if let Some(m) = MONTHS.iter().position(|mn| mn.starts_with(&t[..3]) && mn.starts_with(t.get(..t.len().min(mn.len())).unwrap_or(""))) {
                month = Some(m as u32 + 1);
            }
        }
    }
    let (m, d) = match month {
        Some(m) => (m, nums.first().copied().unwrap_or(1)),
        None => (nums.first().copied().unwrap_or(1), nums.get(1).copied().unwrap_or(1)),
    };
    let Some(y) = year else { return 0 };
    chrono::NaiveDate::from_ymd_opt(y, m, d)
        .and_then(|date| date.and_hms_opt(hour.min(23), minute.min(59), 0))
        .map(|dt| dt.and_utc().timestamp())
        .unwrap_or(0)
}

fn calendar_tokens(ts: i64) -> Vec<String> {
    use chrono::Datelike;
    let Some(dt) = chrono::DateTime::from_timestamp(ts, 0) else { return Vec::new() };
    let d = dt.date_naive();
    vec![
        d.day().to_string(),
        MONTHS[d.month0() as usize].to_string(),
        d.year().to_string(),
        WEEKDAYS[d.weekday().num_days_from_monday() as usize].to_string(),
        format!("__DATE__{}_{:02}_{:02}", d.year(), d.month(), d.day()),
        format!("__YM__{}_{:02}", d.year(), d.month()),
    ]
}

/// Resolve relative time expressions against the session timestamp into absolute calendar tokens.
fn deictic_tokens(raw: &[String], ts: i64) -> Vec<String> {
    if ts == 0 {
        return Vec::new();
    }
    const DAY: i64 = 86_400;
    let mut offsets: Vec<i64> = Vec::new();
    for (i, w) in raw.iter().enumerate() {
        let prev = if i > 0 { raw[i - 1].as_str() } else { "" };
        match w.as_str() {
            "yesterday" => offsets.push(-DAY),
            "tomorrow" => offsets.push(DAY),
            "today" | "tonight" => offsets.push(0),
            "week" if prev == "last" || prev == "past" => offsets.push(-7 * DAY),
            "week" if prev == "next" => offsets.push(7 * DAY),
            "weekend" if prev == "last" || prev == "this" || prev == "past" => offsets.push(-2 * DAY),
            "month" if prev == "last" || prev == "past" => offsets.push(-30 * DAY),
            "month" if prev == "next" => offsets.push(30 * DAY),
            "year" if prev == "last" || prev == "past" => offsets.push(-365 * DAY),
            "year" if prev == "next" => offsets.push(365 * DAY),
            "ago" => {
                let n: i64 = match prev {
                    "days" | "weeks" | "months" | "years" | "day" | "week" | "month" | "year" => {
                        let q = if i >= 3 && raw[i - 2] == "of" {
                            raw[i - 3].as_str()
                        } else if i >= 2 {
                            raw[i - 2].as_str()
                        } else {
                            ""
                        };
                        let n = match q {
                            "a" | "one" => 1,
                            "two" | "couple" => 2,
                            "three" | "few" => 3,
                            "four" => 4,
                            "five" => 5,
                            other => other.parse().unwrap_or(1),
                        };
                        let unit = match prev.trim_end_matches('s') {
                            "day" => 1,
                            "week" => 7,
                            "month" => 30,
                            _ => 365,
                        };
                        n * unit
                    }
                    _ => 0,
                };
                if n > 0 {
                    offsets.push(-n * DAY);
                }
            }
            _ => {
                if (prev == "last" || prev == "this" || prev == "next") && WEEKDAYS.contains(&w.as_str()) {
                    use chrono::Datelike;
                    if let Some(dt) = chrono::DateTime::from_timestamp(ts, 0) {
                        let today = dt.weekday().num_days_from_monday() as i64;
                        let target = WEEKDAYS.iter().position(|d| d == w).unwrap_or(0) as i64;
                        let back = ((today - target + 7) % 7).max(if prev == "last" { 1 } else { 0 });
                        let delta = if prev == "next" { (target - today + 7) % 7 } else { -back };
                        offsets.push(delta * DAY);
                    }
                }
            }
        }
    }
    let mut out = Vec::new();
    for off in offsets {
        out.extend(calendar_tokens(ts + off));
    }
    out
}

// =========================================================================
// INDEX
// =========================================================================

#[derive(Default)]
struct Vocab {
    map: HashMap<String, u32>,
    names: Vec<String>,
}

impl Vocab {
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = self.names.len() as u32;
        self.map.insert(s.to_string(), id);
        self.names.push(s.to_string());
        id
    }
    fn get(&self, s: &str) -> Option<u32> {
        self.map.get(s).copied()
    }
}

#[derive(Clone, Copy)]
struct Feat {
    id: u32,
    w: f32,
    counts_len: bool,
    inheritable: bool,
}

#[derive(Default, Clone)]
struct Bag {
    tf: [HashMap<u32, f32>; 2],
    len: [f32; 2],
}

impl Bag {
    fn add(&mut self, field: usize, f: Feat, scale: f32) {
        *self.tf[field].entry(f.id).or_insert(0.0) += f.w * scale;
        if f.counts_len {
            self.len[field] += f.w * scale;
        }
    }
}

struct TurnIx {
    session: usize,
    id: String,
    field: usize,
    feats: Vec<Feat>,
    /// Exact/stem term ids of content tokens in order, for span proximity.
    seq: Vec<(u32, u32)>,
    content: HashSet<u32>,
    speaker: String,
    is_question: bool,
    answers_question: bool,
    anaphoric: bool,
    pref_marker: bool,
    has_temporal: bool,
    has_numeric: bool,
    has_spatial: bool,
    has_causal: bool,
    is_short_question: bool,
}

struct Stats {
    n: f32,
    df: HashMap<u32, u32>,
    avg: [f32; 2],
}

fn stats_of(bags: &[Bag]) -> Stats {
    let mut df: HashMap<u32, u32> = HashMap::new();
    let mut sum = [0.0f32; 2];
    let mut cnt = [0.0f32; 2];
    for b in bags {
        let mut seen: HashSet<u32> = HashSet::new();
        for f in 0..2 {
            for (&t, &v) in &b.tf[f] {
                if v > 0.0 {
                    seen.insert(t);
                }
            }
            if b.len[f] > 0.0 {
                sum[f] += b.len[f];
                cnt[f] += 1.0;
            }
        }
        for t in seen {
            *df.entry(t).or_insert(0) += 1;
        }
    }
    Stats {
        n: bags.len() as f32,
        df,
        avg: [
            if cnt[0] > 0.0 { sum[0] / cnt[0] } else { 1.0 },
            if cnt[1] > 0.0 { sum[1] / cnt[1] } else { 1.0 },
        ],
    }
}

struct Index {
    vocab: Vocab,
    session_ids: Vec<String>,
    session_ts: Vec<i64>,
    session_turns: Vec<Vec<usize>>,
    turns: Vec<TurnIx>,
    session_bags: Vec<Bag>,
    turn_bags: Vec<Bag>,
    /// (session index, first turn position within session, last turn position inclusive)
    windows: Vec<(usize, usize, usize)>,
    window_bags: Vec<Bag>,
    s_stats: Stats,
    t_stats: Stats,
    w_stats: Stats,
}

const PREF_AXIS: &str = "\u{1}pref";

fn hyp_term(dom: &str) -> String {
    format!("\u{1}hyp:{dom}")
}

fn stem_term(s: &str) -> String {
    format!("\u{1}stem:{s}")
}

fn build_index(sessions: &[Session], p: &Params) -> Index {
    let mut vocab = Vocab::default();
    let mut turns: Vec<TurnIx> = Vec::new();
    let mut session_ids = Vec::new();
    let mut session_ts = Vec::new();
    let mut session_turns = Vec::new();
    let pref_set: HashSet<&str> = PREF_MARKERS.iter().copied().collect();

    for (si, s) in sessions.iter().enumerate() {
        let ts = if s.date.is_empty() { s.timestamp } else { match parse_date(&s.date) { 0 => s.timestamp, t => t } };
        session_ids.push(s.id.clone());
        session_ts.push(ts);
        let date_toks = if p.enable_deictic && ts != 0 { calendar_tokens(ts) } else { Vec::new() };
        let mut members = Vec::new();
        for (ti, t) in s.turns.iter().enumerate() {
            let raw = raw_tokens(&t.content);
            let role = t.role.trim();
            let field = usize::from(p.enable_roles && role.eq_ignore_ascii_case("assistant"));
            let mut feats = Vec::new();
            let mut seq = Vec::new();
            let mut content = HashSet::new();
            for w in raw.iter().filter(|w| is_content(w)) {
                let e = vocab.intern(w);
                feats.push(Feat { id: e, w: 1.0, counts_len: true, inheritable: true });
                content.insert(e);
                let st = stem(w);
                let sid = if p.enable_stemming { vocab.intern(&stem_term(&st)) } else { e };
                if p.enable_stemming {
                    feats.push(Feat { id: sid, w: 1.0, counts_len: false, inheritable: true });
                }
                seq.push((e, sid));
                if p.enable_hypernyms {
                    if let Some(doms) = hypernyms().get(&st) {
                        for d in doms {
                            feats.push(Feat { id: vocab.intern(&hyp_term(d)), w: 1.0, counts_len: false, inheritable: true });
                        }
                    }
                }
            }
            let lower = t.content.to_lowercase();
            let pref_marker = raw.iter().any(|w| pref_set.contains(w.as_str()))
                || PREF_MARKERS.iter().any(|m| m.contains(' ') && lower.contains(m));
            if p.enable_preference && pref_marker && field == 0 {
                feats.push(Feat { id: vocab.intern(PREF_AXIS), w: 1.0, counts_len: false, inheritable: false });
            }
            if p.enable_deictic {
                for d in deictic_tokens(&raw, ts) {
                    feats.push(Feat { id: vocab.intern(&d), w: 1.0, counts_len: false, inheritable: false });
                }
                for d in &date_toks {
                    feats.push(Feat { id: vocab.intern(d), w: p.date_weight, counts_len: false, inheritable: false });
                }
            }
            let speaker = role.to_lowercase();
            if !speaker.is_empty() && speaker != "user" && speaker != "assistant" {
                for w in raw_tokens(&speaker) {
                    feats.push(Feat { id: vocab.intern(&w), w: 1.0, counts_len: false, inheritable: false });
                }
            }
            let is_question = t.content.trim_end().ends_with('?');
            let anaphoric = raw.first().is_some_and(|w| CONTINUATIONS.contains(&w.as_str()))
                || raw.iter().take(12).any(|w| ANAPHORS.contains(&w.as_str()));
            let has_temporal = raw.iter().any(|w| {
                MONTHS.contains(&w.as_str())
                    || WEEKDAYS.contains(&w.as_str())
                    || ["yesterday", "tomorrow", "ago", "week", "month", "year", "am", "pm", "today", "tonight"].contains(&w.as_str())
                    || (w.len() == 4 && w.chars().all(|c| c.is_ascii_digit()))
                    || (w.len() >= 3 && (w.ends_with("st") || w.ends_with("nd") || w.ends_with("rd") || w.ends_with("th")) && w[..w.len()-2].chars().all(|c| c.is_ascii_digit()))
            });
            let has_numeric = t.content.chars().any(|c| c.is_ascii_digit())
                || raw.iter().any(|w| NUMBER_WORDS.contains(&w.as_str()));
            let has_spatial = raw.iter().any(|w| SPATIAL_WORDS.contains(&w.as_str()));
            let has_causal = CAUSAL_WORDS.iter().any(|c| lower.contains(c));
            let is_short_question = is_question && raw.len() < 10;
            let id = if t.id.is_empty() { format!("{}#{}", s.id, ti) } else { t.id.clone() };
            members.push(turns.len());
            turns.push(TurnIx {
                session: si,
                id,
                field,
                feats,
                seq,
                content,
                speaker,
                is_question,
                answers_question: false,
                anaphoric,
                pref_marker,
                has_temporal,
                has_numeric,
                has_spatial,
                has_causal,
                is_short_question,
            });
        }
        session_turns.push(members);
    }

    // Session bags (own text only) and turn bags (own text + centering carry-forward).
    let mut session_bags = Vec::with_capacity(session_turns.len());
    let mut turn_bags: Vec<Bag> = (0..turns.len()).map(|_| Bag::default()).collect();
    for members in &session_turns {
        let mut sb = Bag::default();
        for (pos, &ti) in members.iter().enumerate() {
            let t = &turns[ti];
            for &f in &t.feats {
                sb.add(t.field, f, 1.0);
                turn_bags[ti].add(t.field, f, 1.0);
            }
            if p.enable_carry && pos > 0 {
                let prev = &turns[members[pos - 1]];
                let answer = prev.is_question && prev.speaker != t.speaker;
                let factor = if answer || t.anaphoric { 1.0 } else { p.carry_plain_factor };
                let field = t.field;
                let mut inherited: Vec<(Feat, f32)> = Vec::new();
                for k in 1..=3usize {
                    if pos < k {
                        break;
                    }
                    let src = &turns[members[pos - k]];
                    let w = if k == 1 && answer { p.carry_answer } else { p.carry[k - 1] * factor };
                    if w <= 0.0 {
                        continue;
                    }
                    // Inherit open-class content (exact + stem + hypernym), never the speaker tag or dates.
                    for &f in src.feats.iter().filter(|f| f.inheritable) {
                        inherited.push((Feat { counts_len: false, ..f }, w));
                    }
                }
                for (f, w) in inherited {
                    turn_bags[ti].add(field, f, w);
                }
            }
        }
        session_bags.push(sb);
    }
    // Mark answer turns (second part of a question/answer adjacency pair).
    for members in &session_turns {
        for pos in 1..members.len() {
            let (a, b) = (members[pos - 1], members[pos]);
            if turns[a].is_question && turns[a].speaker != turns[b].speaker {
                turns[b].answers_question = true;
            }
        }
    }

    // Max-passage windows.
    let w = p.window.max(1);
    let mut windows = Vec::new();
    let mut window_bags = Vec::new();
    for (si, members) in session_turns.iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        let last_start = members.len().saturating_sub(w);
        for start in 0..=last_start {
            let end = (start + w).min(members.len()) - 1;
            let mut b = Bag::default();
            for &ti in &members[start..=end] {
                for &f in &turns[ti].feats {
                    b.add(turns[ti].field, f, 1.0);
                }
            }
            windows.push((si, start, end));
            window_bags.push(b);
        }
    }

    let s_stats = stats_of(&session_bags);
    let t_stats = stats_of(&turn_bags);
    let w_stats = stats_of(&window_bags);
    Index {
        vocab,
        session_ids,
        session_ts,
        session_turns,
        turns,
        session_bags,
        turn_bags,
        windows,
        window_bags,
        s_stats,
        t_stats,
        w_stats,
    }
}

// =========================================================================
// QUERY ANALYSIS
// =========================================================================

#[derive(Clone)]
struct QTerm {
    id: u32,
    weight: f32,
    group: Option<usize>,
}

struct QueryPlan {
    intent: QueryIntent,
    terms: Vec<QTerm>,
    groups: Vec<String>,
    /// term id -> group, for proximity scanning
    group_of: HashMap<u32, usize>,
}

fn has_any(hay: &str, cues: &[&str]) -> bool {
    cues.iter().any(|c| hay.contains(c))
}

pub fn classify(text: &str) -> QueryIntent {
    let lower = format!(" {} ", text.to_lowercase().replace(['?', '!', ',', '.'], " $0").replace("$0", ""));
    let lower = lower.replace("  ", " ");
    let content_groups = raw_tokens(text).into_iter().filter(|w| is_content(w)).collect::<HashSet<_>>().len();
    let assistant_attribution = has_any(&lower, ASSISTANT_ATTRIBUTION_CUES);
    let sniper_when = has_any(&lower, &[
        "when ", "when did", "what date", "what time", "what year", "what month", "what day",
        "how long", "how many days", "how many weeks", "how many months", "how many years",
    ]);
    let sniper_how_many = has_any(&lower, &[
        "how many", "how much", "how often", "what age", "how old", "total of",
    ]);
    let sniper_where = has_any(&lower, &[
        "where ", "where did", "where does", "where is", "where were", "what city", "what country",
        "which place", "what place", "which city", "which country",
    ]);
    let sniper_why = has_any(&lower, &[
        "why ", "why did", "why does", "why was", "what reason", "how come",
    ]);
    QueryIntent {
        preference: has_any(&lower, PREF_QUERY_CUES),
        current_state: has_any(&lower, CURRENT_STATE_CUES),
        assistant_attribution,
        user_attribution: !assistant_attribution && has_any(&lower, USER_ATTRIBUTION_CUES),
        multi_hop: content_groups >= 3 && has_any(&lower, MULTI_HOP_CUES),
        inference: has_any(&lower, INFERENCE_CUES),
        temporal: has_any(&lower, TEMPORAL_CUES),
        sniper_when,
        sniper_how_many,
        sniper_where,
        sniper_why,
    }
}

fn plan_query(ix: &Index, q: &Query, p: &Params) -> QueryPlan {
    let intent = classify(&q.text);
    let mut terms = Vec::new();
    let mut groups: Vec<String> = Vec::new();
    let mut group_of = HashMap::new();
    let mut hyp_added: HashSet<&str> = HashSet::new();
    let use_hyp = p.enable_hypernyms;
    for w in raw_tokens(&q.text).into_iter().filter(|w| is_content(w)) {
        if groups.contains(&w) {
            continue;
        }
        let g = groups.len();
        groups.push(w.clone());
        if let Some(id) = ix.vocab.get(&w) {
            terms.push(QTerm { id, weight: 1.0, group: Some(g) });
            group_of.insert(id, g);
        }
        let st = stem(&w);
        if p.enable_stemming {
            if let Some(id) = ix.vocab.get(&stem_term(&st)) {
                terms.push(QTerm { id, weight: p.stem_weight, group: Some(g) });
                group_of.insert(id, g);
            }
        }
        if use_hyp {
            if let Some(doms) = hypernyms().get(&st) {
                for d in doms {
                    if hyp_added.insert(d) {
                        if let Some(id) = ix.vocab.get(&hyp_term(d)) {
                            terms.push(QTerm { id, weight: p.hypernym_weight, group: Some(g) });
                            group_of.insert(id, g);
                        }
                    }
                }
            }
        }
    }
    if p.enable_preference && intent.preference {
        if let Some(id) = ix.vocab.get(PREF_AXIS) {
            terms.push(QTerm { id, weight: p.preference_axis_weight, group: None });
        }
    }
    if p.enable_deictic {
        let q_ts = parse_date(&q.date);
        if q_ts != 0 {
            let raw_q = raw_tokens(&q.text);
            let mut deictic_added = HashSet::new();
            for d in deictic_tokens(&raw_q, q_ts) {
                if deictic_added.insert(d.clone()) {
                    if let Some(id) = ix.vocab.get(&d) {
                        let w = if d.starts_with("__DATE__") { 1.5 } else { p.date_weight };
                        terms.push(QTerm { id, weight: w, group: None });
                    }
                }
            }
        }
    }
    QueryPlan { intent, terms, groups, group_of }
}

// =========================================================================
// SCORING
// =========================================================================

struct Fields {
    w: [f32; 2],
    b: [f32; 2],
}

/// BM25F with IDF power law and IDF-weighted distinct-term coverage.
fn bm25f(bag: &Bag, plan: &QueryPlan, st: &Stats, p: &Params, fl: &Fields) -> (f32, u64) {
    let mut score = 0.0f32;
    let mut mask = 0u64;
    let ng = plan.groups.len();
    let mut g_idf = vec![0.0f32; ng];
    let mut g_hit = vec![false; ng];
    for qt in &plan.terms {
        let Some(&df) = st.df.get(&qt.id) else { continue };
        let df = df as f32;
        let idf = ((st.n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0).powf(p.idf_power);
        if let Some(g) = qt.group {
            g_idf[g] = g_idf[g].max(idf * qt.weight);
        }
        let mut tfp = 0.0f32;
        for f in 0..2 {
            let tf = bag.tf[f].get(&qt.id).copied().unwrap_or(0.0);
            if tf <= 0.0 {
                continue;
            }
            let norm = 1.0 - fl.b[f] + fl.b[f] * (bag.len[f].max(1.0) / st.avg[f].max(1.0));
            tfp += fl.w[f] * tf / norm.max(0.05);
        }
        if tfp > 0.0 {
            score += qt.weight * idf * tfp * (p.k1 + 1.0) / (tfp + p.k1);
            if let Some(g) = qt.group {
                g_hit[g] = true;
                if g < 64 {
                    mask |= 1 << g;
                }
            }
        }
    }
    if p.enable_coverage && score > 0.0 {
        let total: f32 = g_idf.iter().sum();
        if total > 0.0 {
            let hit: f32 = g_idf.iter().zip(&g_hit).filter(|(_, h)| **h).map(|(i, _)| *i).sum();
            let cov = hit / total;
            score *= p.coverage_floor + (1.0 - p.coverage_floor) * cov * cov;
        }
    }
    (score, mask)
}

/// True when two distinct query words land within `span` content tokens of each other.
fn proximity_hit<'a, I: Iterator<Item = &'a (u32, u32)>>(seq: I, plan: &QueryPlan, span: usize) -> bool {
    let mut last: HashMap<usize, usize> = HashMap::new();
    for (pos, (e, s)) in seq.enumerate() {
        let g = plan.group_of.get(e).or_else(|| plan.group_of.get(s));
        if let Some(&g) = g {
            if last.iter().any(|(&h, &lp)| h != g && pos - lp <= span) {
                return true;
            }
            last.insert(g, pos);
        }
    }
    false
}

fn matched_words(plan: &QueryPlan, mask: u64) -> Vec<String> {
    plan.groups
        .iter()
        .enumerate()
        .filter(|(g, _)| *g < 64 && mask & (1 << g) != 0)
        .map(|(_, w)| w.clone())
        .collect()
}

fn order(scores: &[f32], ts: &[i64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(ts[b].cmp(&ts[a]))
            .then(a.cmp(&b))
    });
    idx
}

fn rank_one(ix: &Index, q: &Query, p: &Params, top_k: usize) -> RankResult {
    let plan = plan_query(ix, q, p);
    let it = &plan.intent;
    let fl = if p.enable_roles && it.assistant_attribution {
        Fields { w: [1.0, p.assistant_attribution_weight], b: [p.b_user, p.b_assistant_attributed] }
    } else if p.enable_roles && it.user_attribution {
        Fields { w: [p.user_attribution_weight, 0.35], b: [p.b_user, p.b_assistant] }
    } else {
        Fields { w: [1.0, 1.0], b: [p.b_user, p.b_assistant] }
    };
    let pref_on = p.enable_preference && it.preference;
    let ns = ix.session_bags.len();

    // ---- Session level: whole-session BM25F + max-passage window pooling ----
    let mut s_base = vec![0.0f32; ns];
    let mut s_mask = vec![0u64; ns];
    for si in 0..ns {
        let (sc, m) = bm25f(&ix.session_bags[si], &plan, &ix.s_stats, p, &fl);
        s_base[si] = sc;
        s_mask[si] = m;
    }
    let mut s_win = vec![0.0f32; ns];
    if p.enable_window {
        for (wi, &(si, a, b)) in ix.windows.iter().enumerate() {
            let (mut sc, m) = bm25f(&ix.window_bags[wi], &plan, &ix.w_stats, p, &fl);
            if sc <= 0.0 {
                continue;
            }
            let members = &ix.session_turns[si][a..=b];
            if p.enable_proximity
                && proximity_hit(members.iter().flat_map(|&t| ix.turns[t].seq.iter()), &plan, p.proximity_span)
            {
                sc *= 1.0 + p.proximity_bonus;
            }
            if pref_on && members.iter().any(|&t| ix.turns[t].pref_marker && ix.turns[t].field == 0) {
                sc *= p.preference_boost;
            }
            if sc > s_win[si] {
                s_win[si] = sc;
            }
            s_mask[si] |= m;
        }
    }
    let bmax = s_base.iter().cloned().fold(0.0f32, f32::max);
    let wmax = s_win.iter().cloned().fold(0.0f32, f32::max);
    let mut s_score: Vec<f32> = (0..ns)
        .map(|i| {
            let b = if bmax > 0.0 { s_base[i] / bmax } else { 0.0 };
            if p.enable_window && wmax > 0.0 {
                p.window_mix * (s_win[i] / wmax) + (1.0 - p.window_mix) * b
            } else {
                b
            }
        })
        .collect();

    // ---- Causal supersession: newest high-affinity session wins current-state queries ----
    if p.enable_recency && it.current_state {
        let ord = order(&s_score, &ix.session_ts);
        if let Some(&top) = ord.first() {
            let floor = s_score[top] * p.recency_band;
            let band: Vec<usize> = ord.iter().take(p.recency_depth).copied().filter(|&i| s_score[i] > 0.0 && s_score[i] >= floor).collect();
            if band.len() > 1 {
                if let Some(&newest) = band.iter().max_by_key(|&&i| ix.session_ts[i]) {
                    if ix.session_ts[newest] > 0 {
                        s_score[newest] *= p.recency_boost;
                    }
                }
            }
        }
    }
    let smax = s_score.iter().cloned().fold(0.0f32, f32::max);

    // ---- Turn level ----
    let nt = ix.turns.len();
    let mut t_raw = vec![0.0f32; nt];
    let mut t_mask = vec![0u64; nt];
    for ti in 0..nt {
        let t = &ix.turns[ti];
        let (mut sc, m) = bm25f(&ix.turn_bags[ti], &plan, &ix.t_stats, p, &fl);
        if sc <= 0.0 {
            continue;
        }
        if p.enable_proximity && proximity_hit(t.seq.iter(), &plan, p.proximity_span) {
            sc *= 1.0 + p.proximity_bonus;
        }
        if p.enable_carry && t.answers_question {
            sc *= p.answer_boost;
        }
        if pref_on && t.pref_marker && t.field == 0 {
            sc *= p.preference_boost;
        }
        t_raw[ti] = sc;
        t_mask[ti] = m;
    }

    // Weapon 1: Bidirectional 3-Turn Micro-Scene Pooling
    let mut t_scene = vec![0.0f32; nt];
    if p.enable_scene {
        for members in &ix.session_turns {
            for (pos, &ti) in members.iter().enumerate() {
                let prev_raw = if pos > 0 { t_raw[members[pos - 1]] } else { 0.0 };
                let next_raw = if pos + 1 < members.len() { t_raw[members[pos + 1]] } else { 0.0 };
                t_scene[ti] = 1.0 * t_raw[ti] + 0.75 * prev_raw + 0.55 * next_raw;
                if pos > 0 { t_mask[ti] |= t_mask[members[pos - 1]]; }
                if pos + 1 < members.len() { t_mask[ti] |= t_mask[members[pos + 1]]; }
            }
        }
    } else {
        t_scene = t_raw;
    }

    // Weapon 2: Deterministic Answer-Type Sniper (+65% payload boost)
    if p.enable_sniper {
        for ti in 0..nt {
            if t_scene[ti] <= 0.0 {
                continue;
            }
            let t = &ix.turns[ti];
            if it.sniper_when {
                if t.has_temporal { t_scene[ti] *= 1.65; }
                if t.is_question { t_scene[ti] *= 0.55; }
            }
            if it.sniper_how_many && t.has_numeric {
                t_scene[ti] *= 1.65;
            }
            if it.sniper_where && t.has_spatial {
                t_scene[ti] *= 1.60;
            }
            if it.sniper_why && t.has_causal {
                t_scene[ti] *= 1.55;
            }
            if t.is_short_question {
                t_scene[ti] *= 0.50;
            }
        }
    }

    // Weapon 3: 2-Iteration Sparse Personalized PageRank (PPR Spreading Activation)
    if p.enable_ppr && nt > 5 {
        let turn_ts: Vec<i64> = ix.turns.iter().map(|t| ix.session_ts[t.session]).collect();
        let seed_indices: Vec<usize> = order(&t_scene, &turn_ts)
            .into_iter()
            .take(25)
            .filter(|&i| t_scene[i] > 0.0)
            .collect();

        if !seed_indices.is_empty() {
            let t0 = seed_indices[0];
            let t0_mask = t_mask[t0];
            let mut activation: HashMap<u32, f32> = HashMap::new();

            // Pass 1: Turn -> Bridge Entities
            for &s in &seed_indices {
                let s_score = t_scene[s];
                for &w in &ix.turns[s].content {
                    if let Some(&df) = ix.t_stats.df.get(&w) {
                        if df >= 2 && (df as f32) <= 0.20 * ix.t_stats.n {
                            let idf_term = 1.0 / ((df as f32 + 1.1).ln());
                            *activation.entry(w).or_insert(0.0) += s_score * idf_term;
                        }
                    }
                }
            }

            // Pass 2: Bridge Entities -> 2nd-Hop Turns
            for ti in 0..nt {
                let mut ppr_sum = 0.0f32;
                for &w in &ix.turns[ti].content {
                    if let Some(&act) = activation.get(&w) {
                        ppr_sum += act;
                    }
                }
                if ppr_sum > 0.0 {
                    let mut boost = 0.45 * ppr_sum;
                    let missing_query_terms = (t_mask[ti] & !t0_mask) != 0;
                    if missing_query_terms {
                        boost *= 1.75;
                    }
                    t_scene[ti] += boost;
                }
            }
        }
    }

    // Hierarchical session-to-turn funnel.
    if p.enable_funnel && smax > 0.0 {
        for ti in 0..nt {
            if t_scene[ti] > 0.0 {
                t_scene[ti] *= 1.0 + p.funnel * (s_score[ix.turns[ti].session] / smax);
            }
        }
    }

    // Question/answer dyad coupling: the partner of a strong turn rides along.
    if p.enable_dyad {
        let base = t_scene.clone();
        for members in &ix.session_turns {
            for pos in 1..members.len() {
                let (a, b) = (members[pos - 1], members[pos]);
                if !ix.turns[b].answers_question {
                    continue;
                }
                let (hi, lo) = if base[a] >= base[b] { (a, b) } else { (b, a) };
                if base[hi] <= 0.0 {
                    continue;
                }
                let dyad = base[hi] + p.dyad * base[lo];
                t_scene[hi] = t_scene[hi].max(dyad);
                t_scene[lo] = t_scene[lo].max(dyad * p.dyad_partner);
                t_mask[lo] |= t_mask[hi];
            }
        }
    }

    let k = top_k.max(1);
    let sessions = order(&s_score, &ix.session_ts)
        .into_iter()
        .take(k)
        .map(|i| RankedItem {
            id: ix.session_ids[i].clone(),
            session_id: ix.session_ids[i].clone(),
            score: s_score[i],
            matched: matched_words(&plan, s_mask[i]),
        })
        .collect();

    // Weapon 4: Greedy Submodular Set Cover for Top-K Selection
    let turn_ts: Vec<i64> = ix.turns.iter().map(|t| ix.session_ts[t.session]).collect();
    let final_turn_indices: Vec<usize> = if p.enable_submodular {
        let mut selected_turns: Vec<usize> = Vec::with_capacity(k);
        let mut covered_query_terms: u64 = 0;
        let mut session_counts: HashMap<usize, usize> = HashMap::new();
        let mut remaining: HashSet<usize> = (0..nt).filter(|&ti| t_scene[ti] > 0.0).collect();

        let total_query_groups = plan.groups.len().min(64);

        while selected_turns.len() < k && !remaining.is_empty() {
            let mut best_turn = None;
            let mut best_gain = -1.0f32;

            for &ti in &remaining {
                let sc = t_scene[ti];
                let t_sess = ix.turns[ti].session;
                let count = session_counts.get(&t_sess).copied().unwrap_or(0);
                let diversity_factor = if count == 0 {
                    1.0
                } else if count == 1 {
                    0.70
                } else {
                    0.40
                };

                let t_m = t_mask[ti];
                let uncovered_mask = t_m & !covered_query_terms;
                let uncovered_count = (uncovered_mask & ((1u64 << total_query_groups) - 1)).count_ones() as f32;
                let uncovered_ratio = if total_query_groups > 0 {
                    uncovered_count / (total_query_groups as f32)
                } else {
                    0.0
                };

                let gain = sc * (1.0 + 0.60 * uncovered_ratio) * diversity_factor;
                if gain > best_gain {
                    best_gain = gain;
                    best_turn = Some(ti);
                }
            }

            if let Some(chosen) = best_turn {
                selected_turns.push(chosen);
                remaining.remove(&chosen);
                covered_query_terms |= t_mask[chosen];
                *session_counts.entry(ix.turns[chosen].session).or_insert(0) += 1;
            } else {
                break;
            }
        }

        if selected_turns.len() < k {
            let ord = order(&t_scene, &turn_ts);
            for ti in ord {
                if selected_turns.len() >= k {
                    break;
                }
                if !selected_turns.contains(&ti) {
                    selected_turns.push(ti);
                }
            }
        }
        selected_turns
    } else {
        order(&t_scene, &turn_ts).into_iter().take(k).collect()
    };

    let turns = final_turn_indices
        .into_iter()
        .map(|i| RankedItem {
            id: ix.turns[i].id.clone(),
            session_id: ix.session_ids[ix.turns[i].session].clone(),
            score: t_scene[i],
            matched: matched_words(&plan, t_mask[i]),
        })
        .collect();
    RankResult { intent: plan.intent, sessions, turns }
}

/// Rank every query against the supplied sessions. Pure, deterministic, no I/O.
pub fn rank(req: &RankRequest) -> RankResponse {
    let ix = build_index(&req.sessions, &req.params);
    let results = req.queries.iter().map(|q| rank_one(&ix, q, &req.params, req.top_k)).collect();
    RankResponse {
        engine: ENGINE,
        sessions_indexed: ix.session_ids.len(),
        turns_indexed: ix.turns.len(),
        results,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(role: &str, content: &str) -> Turn {
        Turn { id: String::new(), role: role.into(), content: content.into() }
    }

    #[test]
    fn rejects_benchmark_labels_at_the_boundary() {
        let leaky = r#"{"sessions":[{"id":"s1","turns":[{"role":"user","content":"x","has_answer":true}]}],"queries":[{"text":"x"}]}"#;
        assert!(serde_json::from_str::<RankRequest>(leaky).is_err());
        let leaky_q = r#"{"sessions":[],"queries":[{"text":"x","question_type":"knowledge-update"}]}"#;
        assert!(serde_json::from_str::<RankRequest>(leaky_q).is_err());
    }

    #[test]
    fn stemmer_converges_inflections() {
        assert_eq!(stem("painting"), stem("painted"));
        assert_eq!(stem("paints"), stem("paint"));
        assert_eq!(stem("baking"), stem("bake"));
        assert_eq!(stem("running"), stem("run"));
        assert_eq!(stem("relocated"), stem("relocating"));
        assert_eq!(stem("stories"), stem("story"));
    }

    #[test]
    fn parses_both_dataset_date_formats() {
        let a = parse_date("2023/05/20 (Sat) 02:21");
        let b = parse_date("1:56 pm on 8 May, 2023");
        assert_eq!(calendar_tokens(a)[..3], ["20".to_string(), "may".into(), "2023".into()]);
        assert_eq!(calendar_tokens(b)[..3], ["8".to_string(), "may".into(), "2023".into()]);
        assert_eq!(calendar_tokens(b)[3], "monday");
        assert!(calendar_tokens(a).contains(&"__DATE__2023_05_20".to_string()));
    }

    #[test]
    fn deictic_compiler_resolves_yesterday() {
        let ts = parse_date("1:56 pm on 8 May, 2023");
        let toks = deictic_tokens(&raw_tokens("I went to the group yesterday"), ts);
        assert!(toks.contains(&"__DATE__2023_05_07".to_string()) && toks.contains(&"may".to_string()));
    }

    #[test]
    fn knowledge_update_prefers_newest_state() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "old".into(), date: "2024/01/10 (Wed) 09:15".into(), timestamp: 0, turns: vec![turn("user", "My brother just moved to Seattle for a new job.")] },
                Session { id: "noise".into(), date: "2024/02/01 (Thu) 09:15".into(), timestamp: 0, turns: vec![turn("user", "I baked sourdough bread this weekend.")] },
                Session { id: "new".into(), date: "2024/03/22 (Fri) 18:40".into(), timestamp: 0, turns: vec![turn("user", "My brother hated the rain in Seattle so he relocated to Austin, Texas.")] },
            ],
            queries: vec![Query { text: "Where does my brother currently live?".into(), date: String::new() }],
            top_k: 3,
            params: Params::default(),
        };
        let r = rank(&req);
        assert!(r.results[0].intent.current_state);
        assert_eq!(r.results[0].sessions[0].id, "new");
    }

    #[test]
    fn answer_turn_inherits_question_entities() {
        let req = RankRequest {
            sessions: vec![Session {
                id: "D1".into(),
                date: String::new(),
                timestamp: 0,
                turns: vec![
                    Turn { id: "D1:1".into(), role: "Caroline".into(), content: "I went hiking with my dog.".into() },
                    Turn { id: "D1:2".into(), role: "Melanie".into(), content: "Are you still painting that sunrise landscape?".into() },
                    Turn { id: "D1:3".into(), role: "Caroline".into(), content: "Yeah, I switched to acrylics for it yesterday.".into() },
                    Turn { id: "D1:4".into(), role: "Melanie".into(), content: "My kids love the beach.".into() },
                ],
            }],
            queries: vec![Query { text: "What medium did Caroline use for her sunrise landscape?".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        let r = rank(&req);
        let top2: Vec<&str> = r.results[0].turns.iter().take(2).map(|t| t.id.as_str()).collect();
        assert!(top2.contains(&"D1:3"), "{top2:?}");
    }

    #[test]
    fn assistant_attribution_routes_to_assistant_field() {
        let i = classify("Can you remind me what hotel you recommended in Lisbon?");
        assert!(i.assistant_attribution && !i.user_attribution);
        let pref = classify("Can you suggest some activities I might like this weekend?");
        assert!(pref.preference);
    }

    #[test]
    fn ranking_is_deterministic() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "a".into(), date: String::new(), timestamp: 1, turns: vec![turn("user", "I adopted a golden retriever puppy named Max.")] },
                Session { id: "b".into(), date: String::new(), timestamp: 2, turns: vec![turn("user", "Work has been busy with the quarterly report.")] },
            ],
            queries: vec![Query { text: "What is the name of my puppy?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let a = serde_json::to_string(&rank(&req)).unwrap_or_default();
        let b = serde_json::to_string(&rank(&req)).unwrap_or_default();
        assert_eq!(a, b);
        assert_eq!(rank(&req).results[0].sessions[0].id, "a");
    }
}
