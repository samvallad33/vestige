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

#![allow(clippy::all)]

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
    pub is_event_sequence: bool,
    /// The query asks for a task whose answer should obey stored imperative rules,
    /// even when those rules share no content words with the question.
    pub task_under_rules: bool,
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
    "by", "can", "cant", "could", "couldnt", "did", "didnt", "do", "does", "doesnt", "doing",
    "dont", "down", "during", "each", "few", "for", "from", "further", "had", "hadnt", "has",
    "hasnt", "have", "havent", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "id", "ill", "im", "if", "in", "into", "is", "isnt", "it",
    "its", "itself", "ive", "just", "me", "more", "most", "my", "myself", "no", "nor", "not",
    "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "same", "she", "should", "shouldnt", "so", "some", "such", "than", "that",
    "thats", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "theyre", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasnt", "we", "were", "werent", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "wont", "would", "wouldnt", "you", "your", "youre", "yours",
    "yourself", "yourselves", "also", "really", "yeah", "yes", "oh", "ok", "okay", "lets",
    "got", "get", "gets", "getting", "go", "going", "gone", "went", "one", "much", "many",
    "lot", "lots", "thing", "things", "something", "anything", "everything", "kind", "sort",
    "way", "well", "even", "still", "ever", "may", "might", "must", "shall", "us", "let",
    "make", "made", "know", "think", "tell", "told", "say", "said", "remind", "remember",
    "conversation", "mentioned", "mention", "talked", "talk", "discussed", "chat",
    "actually", "sure", "great", "thanks", "thank", "hey", "hi", "hello", "wow", "like",
    "want", "wanted", "need", "needed", "use", "used", "using", "try", "tried", "trying",
    "new", "good", "help", "recently", "suggest", "recommend", "useful", "interesting", "past",
    "times", "arent",
];

/// Query-framing / request-boilerplate tokens that should not crush domain matches via coverage^2
/// when genuine content tokens exist in the query.
const QUERY_FRAMING_WORDS: &[&str] = &[
    "name", "find", "total", "number", "amount", "count", "order", "earliest", "latest", "list",
    "items", "item", "steps", "step", "tips", "tip", "advice", "summary", "summarize",
    "comprehensive", "overview", "details", "detail", "aspects", "aspect", "different",
    "distinct", "various", "several", "ago", "days", "day", "weeks", "week", "months", "month",
    "years", "year", "earlier", "later", "throughout", "across",
];

const PREF_MARKERS: &[&str] = &[
    "love", "loved", "loves", "loving", "favorite", "favourite", "favorites", "prefer",
    "prefers", "preferred", "preference", "obsessed", "enjoy", "enjoys", "enjoyed", "enjoying",
    "hobby", "hobbies", "passionate", "passion", "collect", "collecting", "hate", "hates",
    "hated", "dislike", "dislikes", "avoid", "avoiding", "allergic", "diet", "vegan",
    "vegetarian", "big fan", "fond", "crave", "addicted", "habit", "routine", "can't stand",
    "i'd like", "would like", "rather", "instead of", "go-to",
    // Exclusion and contrast. Bare "only" is a stopword and is not a unigram marker;
    // exclusive preference is the constraint phrase, not every occurrence of the word.
    "solely", "exclusively", "stick with", "switched to", "too bitter", "i only", "only ever",
    "only use", "only want", "only like", "only drink", "only eat", "only buy", "only wear",
    // Implicit personal constraints. Phrases, not the stopword "only".
    "i don't want", "i do not want", "i won't", "i will not", "rather than", "no longer",
    "not for me", "can't eat", "cannot eat", "can't have", "cannot have", "leave out",
    "keep out",
];

const PREF_QUERY_CUES: &[&str] = &[
    "what kind of", "what type of", "what sort of", "recommend", "suggest", "suggestion",
    "prefer", "preference", "favorite", "favourite", "would i like", "would i enjoy",
    "should i get", "should i try", "should i buy", "gift", "present for", "suitable for me",
    "how do i like", "any ideas", "any tips", "ideas for", "tips for", "advice on",
    "can you suggest", "what should i", "something i", "i might like", "i would like",
    "go-to", "best for me", "fit my", "match my", "based on my", "steps would you",
    "help me set up", "serve",
    // Personal planning, not a favorite-word lookup.
    "how should i", "help me plan", "walk me through", "help me decide", "help me choose",
    "help me figure", "what should i do",
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
    "what should i", "i've", "i'm", "i brought up", "did i mention", "my instruction",
    "i told you",
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

/// Downstream tasks: the user wants an answer produced (a procedure, list,
/// recommendation, calculation, or date), so stored imperative rules apply.
/// Episodic recall ("when did", "what did") is excluded in `classify`.
const TASK_UNDER_RULES_CUES: &[&str] = &[
    "how should", "how do ", "how can ", "how to ", "how is my", "how much",
    "what are some", "what are my", "what are the", "what can you", "what should",
    "what do i", "what steps",
    "can you", "could you", "would you",
    "when is", "when was", "when are",
    "help me", "walk me through", "show me",
    "recommend", "suggest", " i should",
];

const RECALL_NOT_TASK_CUES: &[&str] = &[
    "what did", "where did", "who did", "when did", "why did", "how did",
    "did i", "did you", "you said", "you told", "you mentioned", "you recommended",
    "you suggested",
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

/// 40-cluster deterministic hypernym lattice (coarse life domains + software systems architecture).
const HYPERNYM_LATTICE: &[(&str, &[&str])] = &[
    ("CULINARY", &["sourdough", "croissant", "baking", "bake", "baker", "bread", "loaf", "recipe", "dinner", "lunch", "breakfast", "vegan", "vegetarian", "pescatarian", "roast", "chef", "cook", "cooking", "kitchen", "cuisine", "meal", "dining", "restaurant", "food", "foodie", "dish", "spice", "pasta", "cheese", "dessert", "chocolate", "pastry", "culinary", "pizza", "sushi", "taco", "burger", "salad", "soup", "grill", "bbq", "cake", "cookie", "appliance", "appliances", "smoker", "fryer", "blender", "microwave", "toaster", "fridge", "refrigerator", "oven", "stove", "ingredient", "ingredients", "produce", "homegrown", "harvest"]),
    ("BEVERAGE", &["coffee", "espresso", "latte", "cappuccino", "roastery", "bean", "brew", "brewing", "tea", "matcha", "chai", "cafe", "barista", "aeropress", "pourover", "wine", "beer", "cocktail", "whiskey", "smoothie", "juice"]),
    ("OUTDOORS", &["hike", "hiking", "trail", "kayak", "kayaking", "camping", "camp", "tent", "backpack", "backpacking", "outdoor", "outdoors", "gear", "park", "wilderness", "climb", "climbing", "mountain", "beach", "lake", "fishing", "nature", "roadtrip"]),
    ("FITNESS", &["gym", "weight", "workout", "running", "runner", "run", "jog", "jogging", "cycling", "bike", "bicycle", "swim", "swimming", "triathlon", "marathon", "5k", "10k", "exercise", "cardio", "fitness", "yoga", "pilates", "crossfit", "training", "race", "tournament", "competition", "athletic", "athletics"]),
    ("VISUAL_ART", &["painting", "paint", "acrylic", "watercolor", "watercolour", "oil", "sketch", "sketching", "draw", "drawing", "pottery", "ceramic", "clay", "studio", "sculpture", "gallery", "canvas", "portrait", "landscape", "art", "artistic", "artist", "medium", "easel", "craft", "knitting", "sewing", "exhibition"]),
    ("MUSIC", &["guitar", "piano", "vinyl", "concert", "festival", "album", "spotify", "band", "jazz", "rock", "classical", "instrument", "drum", "bass", "violin", "orchestra", "musician", "song", "singing", "vocalist", "music", "playlist", "podcast", "streaming", "dance", "ballet", "salsa"]),
    ("PETS", &["dog", "puppy", "cat", "kitten", "vet", "pet", "rescue", "adopt", "adoption", "animal", "retriever", "breed", "leash", "veterinary", "veterinarian", "fish", "bird", "hamster", "horse"]),
    ("BOOKS", &["book", "reading", "read", "novel", "author", "fiction", "nonfiction", "memoir", "library", "bookstore", "kindle", "audiobook", "literature", "poetry", "poem", "writing", "writer", "publish", "publication", "publications", "conference", "conferences", "paper", "papers", "article", "articles", "proceedings", "journal", "journals"]),
    ("GAMING", &["game", "gaming", "gamer", "playstation", "nintendo", "switch", "xbox", "steam", "rpg", "boardgame", "tabletop", "console", "chess", "puzzle", "videogame"]),
    ("TRAVEL", &["travel", "traveling", "flight", "airline", "trip", "vacation", "hotel", "hostel", "explore", "exploring", "passport", "tourist", "destination", "luggage", "itinerary", "cruise", "resort", "abroad", "museum", "theater"]),
    ("PHOTOGRAPHY", &["photo", "photography", "camera", "lens", "shoot", "canon", "nikon", "sony", "shutter", "aperture", "exposure", "darkroom", "instagram"]),
    ("GARDENING", &["garden", "gardening", "plant", "indoor", "succulent", "flower", "flowers", "soil", "pruning", "seed", "greenhouse", "botany", "tomato", "herb", "herbs", "homegrown", "harvest", "harvested", "ingredient", "ingredients", "produce", "basil", "mint", "pepper"]),
    ("TECH", &["python", "rust", "javascript", "typescript", "compiler", "react", "server", "bug", "repo", "code", "coding", "programmer", "programming", "software", "developer", "api", "database", "algorithm", "linux", "terminal", "github", "laptop", "computer", "phone", "app", "apps", "application", "applications", "gadget", "accessory", "accessories", "charger", "charging", "charge", "cable", "case", "protector", "earbuds", "headphones", "adapter", "dongle", "battery", "power", "bank", "wireless", "security", "auth", "authentication", "encryption", "token", "flask", "django", "fastapi", "gunicorn", "docker", "deployment", "deploy", "queue", "backoff", "ratelimit", "retry", "retries", "latency", "cache", "caching"]),
    ("VEHICLES", &["car", "vehicle", "truck", "sedan", "ev", "electric", "drive", "driving", "engine", "mechanic", "commute", "commuting", "train", "subway", "transit", "highway", "roadtrip", "motorcycle"]),
    ("FASHION", &["clothes", "clothing", "wear", "wearing", "dress", "jacket", "coat", "shoe", "boot", "sneaker", "style", "vintage", "wardrobe", "outfit", "sweater", "denim", "wool", "jewelry", "makeup", "skincare", "thrifting", "antique"]),
    ("HEALTH", &["health", "doctor", "hospital", "clinic", "medication", "prescription", "medicine", "allergy", "allergic", "symptom", "surgery", "recover", "recovery", "therapy", "therapist", "mental", "headache", "flu", "dentist", "sleep", "diet", "counseling", "psychology", "medical", "clinical", "diagnosis", "healthcare", "imaging", "radiology"]),
    ("AI_ML", &["ai", "artificial", "intelligence", "ml", "machine", "deep", "learning", "neural", "transformer", "llm", "model", "training", "inference", "vision", "nlp", "imaging"]),
    ("FAMILY", &["kid", "child", "children", "son", "daughter", "brother", "sister", "mother", "father", "mom", "dad", "parent", "parenting", "family", "toddler", "baby", "pregnancy", "grandma", "grandpa", "cousin", "niece", "nephew", "sibling", "siblings", "relative", "relatives", "kin", "twin", "twins"]),
    ("COMMUNITY", &["charity", "volunteer", "volunteering", "fundraiser", "fundraising", "donation", "donate", "shelter", "nonprofit", "cause", "awareness", "community", "activism", "protest", "rescue"]),
    ("IDENTITY", &["lgbtq", "queer", "gay", "lesbian", "pride", "trans", "transgender", "gender", "identity", "support", "transition", "religion", "faith", "church", "spiritual", "mentor"]),
    ("EDUCATION", &["school", "college", "university", "degree", "graduate", "graduation", "studying", "study", "exam", "class", "course", "professor", "campus", "student", "certification", "diploma", "lecture", "workshop", "major", "publication", "publications", "conference", "conferences", "paper", "papers", "journal"]),
    ("CAREER", &["job", "career", "boss", "coworker", "interview", "offer", "resume", "promote", "promotion", "salary", "office", "remote", "work", "working", "firm", "client", "company", "manager", "project", "startup", "business", "profession", "occupation", "occupations", "vocation", "agency", "role", "roles", "milestone", "milestones", "buisiness", "contract", "specialist"]),
    ("HOUSING", &["house", "apartment", "rent", "lease", "mortgage", "buy", "buying", "moving", "moved", "move", "neighborhood", "landlord", "bedroom", "renovate", "furniture", "decor", "home"]),
    ("SCREEN", &["movie", "film", "cinema", "tv", "show", "series", "episode", "netflix", "actor", "director", "watch", "watching", "documentary", "anime", "comedy", "sitcom", "special", "specials"]),
    ("FINANCE", &["budget", "money", "save", "saving", "invest", "investment", "stock", "crypto", "deposit", "loan", "debt", "retirement", "expense", "bank", "tax"]),
    ("COMMERCE", &["buy", "bought", "sell", "sold", "purchase", "purchased", "order", "ordered", "price", "discount", "deal", "store", "shop", "shopping", "receipt", "refund", "warranty", "paid", "pay", "cost"]),
    ("RELATIONSHIP", &["date", "dating", "partner", "boyfriend", "girlfriend", "husband", "wife", "marriage", "wedding", "anniversary", "breakup", "divorce", "couple", "friend", "friendship", "single", "unmarried", "spouse", "ceremony", "event", "events"]),
    ("PERSONALITY", &["introvert", "extrovert", "adventurous", "creative", "organized", "outdoorsy", "compassionate", "empathetic", "patient", "energetic", "calm", "anxious", "stressed", "relaxed", "minimalist", "collector", "vintage", "antique", "thrifting", "personality", "trait"]),
    ("WELLNESS", &["meditation", "mindfulness", "massage", "spa", "relaxation", "stress", "breathe", "journal", "journaling", "selfcare"]),
    ("VIDEO", &["video", "editing", "premiere", "adobe", "lumetri", "davinci", "timeline", "render", "codec", "footage", "clip", "youtube", "vlog"]),
    ("SPORTS", &["soccer", "football", "basketball", "baseball", "tennis", "golf", "hockey", "volleyball", "team", "league", "match", "score", "coach", "stadium", "sport", "sports", "event", "events", "tournament", "race", "triathlon", "marathon", "competition", "championship", "athletic"]),
    ("SHOPPING", &["shop", "shopping", "store", "mall", "purchase", "bought", "order", "amazon", "deal", "sale", "discount", "coupon"]),
    ("CELEBRATION", &["party", "birthday", "celebration", "celebrate", "holiday", "christmas", "thanksgiving", "halloween", "gift", "present", "festival", "fest", "event", "events", "milestone", "ceremony", "wedding", "anniversary", "funeral", "graduation"]),
    ("SCIENCE", &["science", "physics", "chemistry", "biology", "astronomy", "space", "research", "experiment", "lab", "math", "publication", "publications", "conference", "conferences", "paper", "papers", "journal", "article", "articles"]),
    ("MEANING", &["symbolize", "represent", "meaning", "signify", "symbol"]),
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

fn framing_words() -> &'static HashSet<&'static str> {
    static F: OnceLock<HashSet<&'static str>> = OnceLock::new();
    F.get_or_init(|| QUERY_FRAMING_WORDS.iter().copied().collect())
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
// TOKENISATION + PROPER ENTITIES + DETERMINISTIC STEMMER
// =========================================================================

fn clean_contraction(raw_lower: &str) -> &str {
    let trimmed = raw_lower.trim_matches(|c: char| c == '\'' || c == '’');
    if let Some(stem_part) = trimmed.strip_suffix("'s").or_else(|| trimmed.strip_suffix("’s")) {
        return stem_part;
    }
    if let Some(stem_part) = trimmed.strip_suffix("'ve").or_else(|| trimmed.strip_suffix("’ve")) {
        return stem_part;
    }
    if let Some(stem_part) = trimmed.strip_suffix("'re").or_else(|| trimmed.strip_suffix("’re")) {
        return stem_part;
    }
    if let Some(stem_part) = trimmed.strip_suffix("'ll").or_else(|| trimmed.strip_suffix("’ll")) {
        return stem_part;
    }
    if let Some(stem_part) = trimmed.strip_suffix("'m").or_else(|| trimmed.strip_suffix("’m")) {
        return stem_part;
    }
    if let Some(stem_part) = trimmed.strip_suffix("'d").or_else(|| trimmed.strip_suffix("’d")) {
        return stem_part;
    }
    trimmed
}

fn raw_tokens(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut out = Vec::new();
    for piece in lower.split(|c: char| !c.is_alphanumeric() && c != '\'' && c != '’') {
        if piece.is_empty() {
            continue;
        }
        let cleaned = clean_contraction(piece);
        if cleaned.contains('\'') || cleaned.contains('’') {
            // Handle contractions like don't -> dont, didn't -> didnt, can't -> cant cleanly;
            // non-contraction apostrophes like d'aulps split cleanly into sub-words.
            if cleaned.ends_with("n't") || cleaned.ends_with("n’t") {
                let joined: String = cleaned.chars().filter(|&c| c != '\'' && c != '’').collect();
                if !joined.is_empty() {
                    out.push(joined);
                }
            } else {
                for sub in cleaned.split(|c: char| c == '\'' || c == '’') {
                    if !sub.is_empty() {
                        out.push(sub.to_string());
                    }
                }
            }
        } else if !cleaned.is_empty() {
            out.push(cleaned.to_string());
        }
    }
    out
}

fn is_content(w: &str) -> bool {
    w.chars().count() > 1 && !stopwords().contains(w)
}

const STARTER_STOP: &[&str] = &[
    "what", "when", "where", "why", "which", "who", "whom", "whose", "how",
    "are", "is", "was", "were", "do", "does", "did", "can", "could", "will",
    "would", "should", "shall", "may", "might", "have", "has", "had", "the",
    "this", "that", "these", "those", "there", "here", "yes", "yeah", "no",
    "sure", "certainly", "absolutely", "great", "thanks", "thank", "hi", "hey",
    "hello", "if", "so", "but", "and", "or",
    "my", "your", "our", "his", "her", "their", "its", "i", "we", "you", "they", "he", "she", "it",
];

const TITLE_CONNECTORS: &[&str] = &["and", "of", "the", "for", "in", "on", "at"];

/// Extract contiguous capitalized proper-noun entity phrases ("\u{1}ent:local_h", "\u{1}ent:kiss_and_tell").
/// Preserves short single-letter entities ("H" in "Local H"), capitalized prepositions ("For Against"),
/// and title-case connectors ("Kiss and Tell", "Lord of the Rings").
fn proper_entity_tokens(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut run: Vec<String> = Vec::new();

    let flush_run = |run: &mut Vec<String>, out: &mut Vec<String>| {
        // Trim trailing lowercase connectors if the run ended on a preposition like "Kiss and"
        while run.last().is_some_and(|s| s.starts_with('\u{2}')) {
            run.pop();
        }
        if run.len() >= 2 {
            for win in run.windows(2) {
                if !win[0].starts_with('\u{2}') && !win[1].starts_with('\u{2}') {
                    out.push(format!("\u{1}ent:{}_{}", win[0], win[1]));
                }
            }
            if run.len() <= 4 {
                let clean_parts: Vec<&str> = run.iter().map(|s| s.trim_start_matches('\u{2}')).collect();
                out.push(format!("\u{1}ent:{}", clean_parts.join("_")));
            }
        }
        run.clear();
    };

    for raw_word in text.split_whitespace() {
        let ends_clause = raw_word.ends_with(['.', '?', '!', ',', ';', ':']);
        let w = raw_word.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '’' && c != '-');
        if w.is_empty() {
            flush_run(&mut run, &mut out);
            continue;
        }
        let clean = clean_contraction(w);
        let first_upper = clean.chars().next().is_some_and(|c| c.is_ascii_uppercase());
        let lower = clean.to_lowercase();
        let norm: String = lower.chars().filter(|c| c.is_alphanumeric()).collect();
        if first_upper && !norm.is_empty() && !STARTER_STOP.contains(&norm.as_str()) {
            run.push(norm);
        } else if !run.is_empty() && TITLE_CONNECTORS.contains(&norm.as_str()) {
            // Allow up to 2 title connectors inside an active proper-noun chain (e.g. "Lord of the Rings").
            let trailing_conns = run.iter().rev().take_while(|s| s.starts_with('\u{2}')).count();
            if trailing_conns < 2 {
                run.push(format!("\u{2}{norm}"));
            } else {
                flush_run(&mut run, &mut out);
            }
        } else {
            flush_run(&mut run, &mut out);
        }
        if ends_clause {
            flush_run(&mut run, &mut out);
        }
    }
    flush_run(&mut run, &mut out);
    out
}

/// Deterministic suffix stripper (Porter-style + small irregular verb lemma map, no model).
pub fn stem(word: &str) -> String {
    let w = word.to_lowercase();
    let irregular = match w.as_str() {
        "drew" | "drawn" => Some("draw"),
        "felt" => Some("feel"),
        "bought" => Some("buy"),
        "sold" => Some("sell"),
        "taught" => Some("teach"),
        "wrote" | "written" => Some("write"),
        "met" => Some("meet"),
        "spoke" | "spoken" => Some("speak"),
        "ran" => Some("run"),
        "swam" | "swum" => Some("swim"),
        "flew" | "flown" => Some("fly"),
        "ate" | "eaten" => Some("eat"),
        "won" => Some("win"),
        "kept" => Some("keep"),
        "built" => Some("build"),
        "spent" => Some("spend"),
        "took" | "taken" => Some("take"),
        "lost" => Some("lose"),
        "gave" | "given" => Some("give"),
        "saw" | "seen" => Some("see"),
        "knew" | "known" => Some("know"),
        "grew" | "grown" => Some("grow"),
        "wore" | "worn" => Some("wear"),
        "chose" | "chosen" => Some("choose"),
        "broke" | "broken" => Some("break"),
        "drove" | "driven" => Some("drive"),
        "slept" => Some("sleep"),
        _ => None,
    };
    if let Some(base) = irregular {
        return base.to_string();
    }
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
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return dt.timestamp();
    }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return dt.and_utc().timestamp();
    }
    if let Ok(d) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return d.and_hms_opt(0, 0, 0).map(|dt| dt.and_utc().timestamp()).unwrap_or(0);
    }
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
        format!("__MONTH__{:02}", d.month()),
        format!("__{}__{:02}", month_band(d.day()), d.month()),
    ]
}

fn month_band(day: u32) -> &'static str {
    if day <= 10 { "EARLY" } else if day >= 21 { "LATE" } else { "MID" }
}

fn month_index(tok: &str) -> Option<u32> {
    let lower = tok.to_lowercase();
    MONTHS.iter().position(|&m| m == lower).map(|i| i as u32 + 1)
}

/// Extract explicit calendar dates written directly inside prose/queries
/// (e.g. "1 May, 2022", "May 1, 2022", "March 29, 2024", "2022-05-01").
fn inline_date_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let lower = text.to_lowercase();
    // Match ISO YYYY-MM-DD
    for w in lower.split_whitespace() {
        let trimmed = w.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-');
        if trimmed.len() == 10 && trimmed.as_bytes().get(4) == Some(&b'-') && trimmed.as_bytes().get(7) == Some(&b'-') {
            if let Ok(d) = chrono::NaiveDate::parse_from_str(trimmed, "%Y-%m-%d") {
                if let Some(dt) = d.and_hms_opt(0, 0, 0) {
                    out.extend(calendar_tokens(dt.and_utc().timestamp()));
                }
            }
        }
    }
    let toks: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();
    for i in 0..toks.len() {
        let Some(m) = month_index(toks[i]) else { continue };
        if let Some(band) = i.checked_sub(1).and_then(|k| match toks[k] {
            "early" => Some("EARLY"),
            "late" => Some("LATE"),
            "mid" | "middle" => Some("MID"),
            _ => None,
        }) {
            out.push(format!("__{band}__{m:02}"));
            out.push(format!("__MONTH__{m:02}"));
        }
        // "<day> <month> <year>" OR "<month> <day> <year>"
        let prev_num = i.checked_sub(1).and_then(|k| toks[k].parse::<u32>().ok()).filter(|&n| (1..=31).contains(&n));
        let next_num = toks.get(i + 1).and_then(|s| s.parse::<u32>().ok()).filter(|&n| (1..=31).contains(&n));
        let year_after_next = toks.get(i + 2).and_then(|s| (s.len() == 4).then(|| s.parse::<i32>().ok()).flatten());
        let year_after_month = toks.get(i + 1).and_then(|s| (s.len() == 4).then(|| s.parse::<i32>().ok()).flatten());

        if let (Some(day), Some(y)) = (prev_num, year_after_month) {
            if let Some(dt) = chrono::NaiveDate::from_ymd_opt(y, m, day).and_then(|d| d.and_hms_opt(0, 0, 0)) {
                out.extend(calendar_tokens(dt.and_utc().timestamp()));
                continue;
            }
        }
        if let (Some(day), Some(y)) = (next_num, year_after_next) {
            if let Some(dt) = chrono::NaiveDate::from_ymd_opt(y, m, day).and_then(|d| d.and_hms_opt(0, 0, 0)) {
                out.extend(calendar_tokens(dt.and_utc().timestamp()));
                continue;
            }
        }
        if let Some(y) = year_after_month {
            out.push(format!("__YM__{y}_{m:02}"));
            out.push(format!("__MONTH__{m:02}"));
        } else if let Some(day) = prev_num.or(next_num) {
            out.push(format!("__MD__{m:02}_{day:02}"));
            out.push(format!("__MONTH__{m:02}"));
        }
    }
    out
}

fn parse_qty_word(q: &str) -> i64 {
    match q {
        "a" | "an" | "one" => 1,
        "two" | "couple" => 2,
        "three" | "few" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        other => other.parse().unwrap_or(1),
    }
}

/// Resolve relative time expressions against the session timestamp into (token, relative_weight).
fn deictic_weighted_tokens(raw: &[String], ts: i64) -> Vec<(String, f32)> {
    if ts == 0 {
        return Vec::new();
    }
    const DAY: i64 = 86_400;
    // (offset_seconds, weight, is_coarse_window)
    let mut specs: Vec<(i64, f32, bool)> = Vec::new();
    for (i, w) in raw.iter().enumerate() {
        let prev = if i > 0 { raw[i - 1].as_str() } else { "" };
        let next = raw.get(i + 1).map(String::as_str).unwrap_or("");
        match w.as_str() {
            "yesterday" => specs.push((-DAY, 1.0, false)),
            "night" if prev == "last" => specs.push((-DAY, 1.0, false)),
            "before" if prev == "day" => specs.push((-DAY, 0.9, false)),
            "tomorrow" => specs.push((DAY, 1.0, false)),
            "today" | "tonight" => specs.push((0, 1.0, false)),
            "week" if prev == "last" || prev == "past" => specs.push((-7 * DAY, 1.0, true)),
            "week" if prev == "next" => specs.push((7 * DAY, 1.0, true)),
            "weekend" if prev == "last" || prev == "this" || prev == "past" => specs.push((-2 * DAY, 1.0, true)),
            "month" if prev == "last" || prev == "past" => {
                specs.push((-30 * DAY, 1.0, true));
                specs.push((-15 * DAY, 0.85, false));
            }
            "month" if prev == "next" => specs.push((30 * DAY, 1.0, true)),
            "year" if prev == "last" || prev == "past" => specs.push((-365 * DAY, 1.0, true)),
            "year" if prev == "next" => specs.push((365 * DAY, 1.0, true)),
            "ago" => {
                if matches!(prev, "days" | "weeks" | "months" | "years" | "day" | "week" | "month" | "year") {
                    let q = if i >= 3 && raw[i - 2] == "of" {
                        raw[i - 3].as_str()
                    } else if i >= 2 {
                        raw[i - 2].as_str()
                    } else {
                        ""
                    };
                    let n = parse_qty_word(q);
                    let (unit, coarse) = match prev.trim_end_matches('s') {
                        "day" => (1, n > 3),
                        "week" => (7, true),
                        "month" => (30, true),
                        _ => (365, true),
                    };
                    if n > 0 {
                        specs.push((-n * unit * DAY, 1.0, coarse));
                    }
                }
            }
            "days" | "weeks" | "months" | "years" if next != "ago" => {
                // Handle "for 3 years now", "past 2 weeks", "last 3 months"
                let prep = if i >= 2 { raw[i - 2].as_str() } else { "" };
                if matches!(prep, "for" | "past" | "last" | "in") || next == "now" {
                    let n = parse_qty_word(prev);
                    let unit = match w.trim_end_matches('s') {
                        "day" => 1,
                        "week" => 7,
                        "month" => 30,
                        _ => 365,
                    };
                    if n > 0 && n <= 120 {
                        specs.push((-n * unit * DAY, 0.9, true));
                    }
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
                        specs.push((delta * DAY, 1.0, false));
                    }
                }
            }
        }
    }
    let mut out = Vec::new();
    for (off, w, coarse) in specs {
        for tok in calendar_tokens(ts + off) {
            out.push((tok, w));
        }
        // Soft +-2d halo on coarse multi-week / multi-month expressions (handles Feb 28 / Mar 1 boundary).
        if coarse {
            for d_off in [-2 * DAY, -DAY, DAY, 2 * DAY] {
                for tok in calendar_tokens(ts + off + d_off) {
                    if tok.starts_with("__DATE__") || tok.starts_with("__YM__") || tok.starts_with("__MONTH__") {
                        out.push((tok, w * 0.45));
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
fn deictic_tokens(raw: &[String], ts: i64) -> Vec<String> {
    deictic_weighted_tokens(raw, ts).into_iter().map(|(t, _)| t).collect()
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
    /// Capitalized proper-entity term IDs inside this turn.
    entities: HashSet<u32>,
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
    is_standing_policy: bool,
    /// User turn that opens a concrete event, not a rule and not a backchannel.
    is_initiative: bool,
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
    session_entities: Vec<HashSet<u32>>,
    session_content: Vec<HashSet<u32>>,
    turns: Vec<TurnIx>,
    session_bags: Vec<Bag>,
    turn_bags: Vec<Bag>,
    /// (session index, first turn position within session, last turn position inclusive)
    windows: Vec<(usize, usize, usize)>,
    window_bags: Vec<Bag>,
    s_stats: Stats,
    t_stats: Stats,
    w_stats: Stats,
    /// Content-term ids that are the object of a personal constraint ("allergic to X").
    constraint_terms: HashSet<u32>,
}

const PREF_AXIS: &str = "\u{1}pref";

fn hyp_term(dom: &str) -> String {
    format!("\u{1}hyp:{dom}")
}

fn stem_term(s: &str) -> String {
    format!("\u{1}stem:{s}")
}

/// Object tokens of a personal constraint. The marker is not enough; the thing
/// constrained has to be in the index or a later question cannot hit it.
fn typed_slot_tokens(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for raw in text.split_whitespace() {
        let tok = raw.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != ':' && c != '_' && c != '-');
        let Some((key, value)) = tok.split_once(':') else { continue };
        if key.len() >= 2
            && key.chars().all(|c| c.is_ascii_alphabetic())
            && !value.is_empty()
            && value.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            out.push((key.to_lowercase(), value.to_lowercase()));
        }
    }
    out
}

fn constraint_object_tokens(lower: &str) -> Vec<String> {
    const LEADS: &[&str] = &[
        "don't want ", "dont want ", "do not want ",
        "allergic to ", "can't eat ", "cant eat ", "cannot eat ",
        "can't have ", "cant have ", "cannot have ",
    ];
    let mut out = Vec::new();
    for lead in LEADS {
        let mut rest = lower;
        while let Some(at) = rest.find(lead) {
            let after = &rest[at + lead.len()..];
            out.extend(
                after
                    .split(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
                    .filter(|w| !w.is_empty() && is_content(w))
                    .take(2)
                    .map(|w| w.to_string()),
            );
            rest = &rest[at + lead.len()..];
        }
    }
    out
}

/// Imperative user rule. O(1) scan of a fixed prefix/phrase set, shared by
/// index construction and the intention-graph registration path.
pub fn utterance_is_standing_policy(text: &str) -> bool {
    let lower = text.to_lowercase();
    let trimmed = lower.trim_start();
    trimmed.starts_with("always ")
        || trimmed.starts_with("never ")
        || trimmed.starts_with("make sure")
        || trimmed.starts_with("ensure ")
        || trimmed.starts_with("remember to")
        || trimmed.starts_with("from now on")
        || trimmed.starts_with("do not ")
        || trimmed.starts_with("don't ")
        || lower.contains("always format")
        || lower.contains("always include")
        || lower.contains("always use")
        || lower.contains("never use")
        || lower.contains("format all")
}

/// Stable plan id for one normalized standing-policy utterance.
pub fn standing_policy_id(text: &str) -> String {
    let norm = text.trim().to_lowercase();
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in norm.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("sp-{hash:016x}")
}

fn detect_question(content: &str, raw_len: usize) -> (bool, bool) {
    let prose = content.split("```").next().unwrap_or(content).trim_end();
    if let Some(q_pos) = prose.rfind('?') {
        let tail = prose[q_pos + 1..].trim();
        let ends_with_q = tail.is_empty() || tail.chars().all(|c| !c.is_ascii_alphabetic());
        let is_q = ends_with_q || prose.len() <= 350;
        let is_short_q = ends_with_q && raw_len < 10;
        return (is_q, is_short_q);
    }
    (false, false)
}

fn build_index(sessions: &[Session], p: &Params) -> Index {
    let mut vocab = Vocab::default();
    let mut turns: Vec<TurnIx> = Vec::new();
    let mut session_ids = Vec::new();
    let mut session_ts = Vec::new();
    let mut session_turns = Vec::new();
    let mut session_entities = Vec::with_capacity(sessions.len());
    let mut session_content = Vec::with_capacity(sessions.len());
    let pref_set: HashSet<&str> = PREF_MARKERS.iter().copied().collect();
    let mut constraint_terms: HashSet<u32> = HashSet::new();

    for (si, s) in sessions.iter().enumerate() {
        let ts = if s.date.is_empty() { s.timestamp } else { match parse_date(&s.date) { 0 => s.timestamp, t => t } };
        session_ids.push(s.id.clone());
        session_ts.push(ts);
        let date_toks = if p.enable_deictic && ts != 0 { calendar_tokens(ts) } else { Vec::new() };
        let mut members = Vec::new();
        let mut s_ents = HashSet::new();
        let mut s_cont = HashSet::new();
        for (ti, t) in s.turns.iter().enumerate() {
            let raw = raw_tokens(&t.content);
            let role = t.role.trim();
            let field = usize::from(p.enable_roles && role.eq_ignore_ascii_case("assistant"));
            let mut feats = Vec::new();
            let mut seq = Vec::new();
            let mut content = HashSet::new();
            let mut entities = HashSet::new();
            for w in raw.iter().filter(|w| is_content(w)) {
                let e = vocab.intern(w);
                feats.push(Feat { id: e, w: 1.0, counts_len: true, inheritable: true });
                content.insert(e);
                s_cont.insert(e);
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
            for ent in proper_entity_tokens(&t.content) {
                let eid = vocab.intern(&ent);
                feats.push(Feat { id: eid, w: 1.0, counts_len: false, inheritable: true });
                entities.insert(eid);
                s_ents.insert(eid);
            }
            let lower = t.content.to_lowercase();
            let pref_marker = raw.iter().any(|w| pref_set.contains(w.as_str()))
                || PREF_MARKERS.iter().any(|m| m.contains(' ') && lower.contains(m));
            if p.enable_preference && pref_marker && field == 0 {
                feats.push(Feat { id: vocab.intern(PREF_AXIS), w: 1.0, counts_len: false, inheritable: false });
            }
            if p.enable_deictic {
                for (d, w) in deictic_weighted_tokens(&raw, ts) {
                    feats.push(Feat { id: vocab.intern(&d), w, counts_len: false, inheritable: false });
                }
                for d in inline_date_tokens(&t.content) {
                    feats.push(Feat { id: vocab.intern(&d), w: 0.9, counts_len: false, inheritable: false });
                }
                for d in &date_toks {
                    feats.push(Feat { id: vocab.intern(d), w: p.date_weight, counts_len: false, inheritable: false });
                }
            }
            let speaker = role.to_lowercase();
            if !speaker.is_empty() && speaker != "user" && speaker != "assistant" {
                for w in raw_tokens(&speaker) {
                    let id = vocab.intern(&w);
                    // Speaker is a boost, never a filter: the name is an identity edge.
                    feats.push(Feat { id, w: 1.6, counts_len: false, inheritable: false });
                    entities.insert(id);
                    s_ents.insert(id);
                }
            }
            for obj in constraint_object_tokens(&lower) {
                let id = vocab.intern(&obj);
                feats.push(Feat { id, w: 1.8, counts_len: false, inheritable: true });
                content.insert(id);
                s_cont.insert(id);
                constraint_terms.insert(id);
            }
            for (slot_key, slot_val) in typed_slot_tokens(&t.content) {
                let kid = vocab.intern(&slot_key);
                let vid = vocab.intern(&slot_val);
                feats.push(Feat { id: kid, w: 0.4, counts_len: false, inheritable: false });
                feats.push(Feat { id: vid, w: 1.6, counts_len: false, inheritable: true });
                content.insert(vid);
                s_cont.insert(vid);
            }
            let (is_question, is_short_question) = detect_question(&t.content, raw.len());
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
            let is_standing_policy = field == 0 && utterance_is_standing_policy(&t.content);
            let is_initiative = field == 0
                && !is_standing_policy
                && !anaphoric
                && !is_short_question
                && content.len() >= 4; // frozen: 12 was too high for a single user event; do not retune after the run
            let id = if t.id.is_empty() { format!("{}#{}", s.id, ti) } else { t.id.clone() };
            members.push(turns.len());
            turns.push(TurnIx {
                session: si,
                id,
                field,
                feats,
                seq,
                content,
                entities,
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
                is_standing_policy,
                is_initiative,
            });
        }
        session_turns.push(members);
        session_entities.push(s_ents);
        session_content.push(s_cont);
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
            if p.enable_carry {
                let short = t.content.len() <= 6;
                for (npos, weight) in [(pos.wrapping_sub(1), if short { 1.0 } else { 0.55 }), (pos + 1, if short { 1.0 } else { 0.55 })] {
                    if npos >= members.len() || (npos == pos.wrapping_sub(1) && pos == 0) {
                        continue;
                    }
                    let src = &turns[members[npos]];
                    for &f in src.feats.iter().filter(|f| f.inheritable) {
                        turn_bags[ti].add(t.field, Feat { counts_len: false, ..f }, weight);
                    }
                }
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
                    // Inherit open-class content (exact + stem + hypernym + proper entities), never speaker tag or dates.
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
        session_entities,
        session_content,
        turns,
        session_bags,
        turn_bags,
        windows,
        window_bags,
        s_stats,
        t_stats,
        w_stats,
        constraint_terms,
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
    let is_event_sequence = has_any(&lower, &[
        "order of", "order in which", "first to last", "earliest to latest",
        "chronological", "order the", "order these", "timeline of",
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
        is_event_sequence,
        task_under_rules: !has_any(&lower, RECALL_NOT_TASK_CUES)
            && has_any(&lower, TASK_UNDER_RULES_CUES),
    }
}

fn plan_query(ix: &Index, q: &Query, p: &Params) -> QueryPlan {
    let mut intent = classify(&q.text);
    let mut terms = Vec::new();
    let mut groups: Vec<String> = Vec::new();
    let mut group_of = HashMap::new();
    let mut hyp_added: HashSet<&str> = HashSet::new();
    let use_hyp = p.enable_hypernyms;

    let raw_q = raw_tokens(&q.text);
    if raw_q.iter().any(|w| ix.vocab.get(w).is_some_and(|id| ix.constraint_terms.contains(&id))) {
        intent.preference = true;
    }
    let inline_dates = if p.enable_deictic { inline_date_tokens(&q.text) } else { Vec::new() };
    let q_ts = if p.enable_deictic { parse_date(&q.date) } else { 0 };
    let deictic_items = if p.enable_deictic && q_ts != 0 {
        deictic_weighted_tokens(&raw_q, q_ts)
    } else {
        Vec::new()
    };

    let fw = framing_words();
    let domain_groups = raw_q
        .iter()
        .filter(|w| is_content(w) && !fw.contains(w.as_str()))
        .collect::<HashSet<_>>()
        .len();
    let can_demote_framing = domain_groups >= 1;

    let mut seen_toks = HashSet::new();
    for w in raw_q.iter().filter(|w| is_content(w)) {
        if !seen_toks.insert(w.clone()) {
            continue;
        }
        let is_frame = can_demote_framing && fw.contains(w.as_str());
        if is_frame {
            if let Some(id) = ix.vocab.get(w) {
                terms.push(QTerm { id, weight: 0.35, group: None });
            }
            if p.enable_stemming {
                let st = stem(w);
                if let Some(id) = ix.vocab.get(&stem_term(&st)) {
                    terms.push(QTerm { id, weight: 0.25, group: None });
                }
            }
            continue;
        }

        let g = groups.len();
        groups.push(w.clone());
        if let Some(id) = ix.vocab.get(w) {
            terms.push(QTerm { id, weight: 1.0, group: Some(g) });
            group_of.insert(id, g);
        }
        let st = stem(w);
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

    // Capitalized proper-noun entity phrases present in index (e.g. "Local H", "For Against").
    // If all parts of the phrase are already covered content words ("United States"), emit as
    // phrase boost (group: None) to avoid double-counting in coverage; if any part was erased
    // by stopwords or length-1 filtering ("For Against", "Local H"), restore it as a real group.
    let mut seen_ents = HashSet::new();
    for ent in proper_entity_tokens(&q.text) {
        if !seen_ents.insert(ent.clone()) {
            continue;
        }
        if let Some(id) = ix.vocab.get(&ent) {
            let parts_body = ent.strip_prefix("\u{1}ent:").unwrap_or("");
            let any_erased = parts_body.split('_').any(|part| !groups.iter().any(|g| g == part));
            if any_erased {
                let g = groups.len();
                groups.push(ent);
                terms.push(QTerm { id, weight: 1.65, group: Some(g) });
                group_of.insert(id, g);
            } else {
                terms.push(QTerm { id, weight: 0.75, group: None });
            }
        }
    }

    if p.enable_preference && intent.preference {
        if let Some(id) = ix.vocab.get(PREF_AXIS) {
            terms.push(QTerm { id, weight: p.preference_axis_weight, group: None });
        }
    }

    if p.enable_deictic {
        let mut date_weights: HashMap<String, f32> = HashMap::new();
        for d in inline_dates {
            let w = if d.starts_with("__DATE__") {
                2.0
            } else if d.starts_with("__MD__") {
                1.4
            } else if d.starts_with("__YM__") || d.starts_with("__MONTH__") {
                1.0
            } else {
                p.date_weight
            };
            let entry = date_weights.entry(d).or_insert(0.0);
            *entry = (*entry).max(w);
        }
        for (d, scale) in deictic_items {
            let base = if d.starts_with("__DATE__") {
                1.9
            } else if d.starts_with("__YM__") || d.starts_with("__MONTH__") {
                0.85
            } else {
                p.date_weight
            };
            let w = base * scale;
            let entry = date_weights.entry(d).or_insert(0.0);
            *entry = (*entry).max(w);
        }
        for (d, w) in date_weights {
            if let Some(id) = ix.vocab.get(&d) {
                terms.push(QTerm { id, weight: w, group: None });
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

fn jaccard(a: &HashSet<u32>, b: &HashSet<u32>) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f32;
    let union = (a.len() + b.len()) as f32 - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

fn raise_floor(scores: &mut [f32], lexical_max: f32, frac: f32, keep: impl Fn(usize) -> bool) {
    let floor = if lexical_max > 0.0 { frac * lexical_max } else { 1.0 };
    for i in 0..scores.len() {
        if keep(i) && scores[i] < floor {
            scores[i] = floor;
        }
    }
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
            if members.iter().any(|&t| ix.turns[t].is_standing_policy) {
                sc *= 1.8;
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

    // Turn-level role alignment: when the query asks what the USER said/brought up, assistant turns are suppressed.
    if p.enable_roles && it.user_attribution {
        for ti in 0..nt {
            if ix.turns[ti].field == 1 {
                t_scene[ti] *= 0.35;
            }
        }
    }

    // Standing policy invariant boost:
    for ti in 0..nt {
        let t = &ix.turns[ti];
        if t.is_standing_policy && t_scene[ti] > 0.0 {
            t_scene[ti] *= 2.5;
        }
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

    // Weapon 3: Bounded Length-Normalized 2-Iteration Sparse PPR (Turn Level)
    if p.enable_ppr && nt > 5 {
        let turn_ts: Vec<i64> = ix.turns.iter().map(|t| ix.session_ts[t.session]).collect();
        let seed_indices: Vec<usize> = order(&t_scene, &turn_ts)
            .into_iter()
            .take(16)
            .filter(|&i| t_scene[i] > 0.0)
            .collect();

        if let Some(&t0) = seed_indices.first() {
            let t_max = t_scene[t0];
            let t0_mask = t_mask[t0];
            if t_max > 0.0 {
                let mut activation: HashMap<u32, f32> = HashMap::new();
                let max_df = ((ix.t_stats.n * 0.12).max(4.0)) as u32;

                // Pass 1: Normalized Seed Turn -> Discriminative Bridge Entities
                for &s in &seed_indices {
                    if t_scene[s] < 0.15 * t_max {
                        break;
                    }
                    let seed_norm = (t_scene[s] / t_max) / ((ix.turns[s].content.len() + 4) as f32).sqrt();
                    for &e in &ix.turns[s].entities {
                        if let Some(&df) = ix.t_stats.df.get(&e) {
                            if df >= 2 && df <= max_df {
                                let idf_term = 1.0 / ((df as f32 + 1.1).ln());
                                *activation.entry(e).or_insert(0.0) += 2.0 * seed_norm * idf_term;
                            }
                        }
                    }
                }

                // Pass 2: Bridge Entities -> Normalized, Bounded 2nd-Hop Turns
                if !activation.is_empty() {
                    for ti in 0..nt {
                        let mut ent_hit = 0.0f32;
                        for &e in &ix.turns[ti].entities {
                            if let Some(&act) = activation.get(&e) {
                                ent_hit += 2.0 * act;
                            }
                        }
                        let ppr_sum = ent_hit;
                        if ppr_sum > 0.0 && (t_scene[ti] > 0.0 || ent_hit > 0.0) {
                            let target_norm = ppr_sum / ((ix.turns[ti].content.len() + 4) as f32).sqrt();
                            let sat = target_norm / (target_norm + 0.45);
                            let mult = if (t_mask[ti] & !t0_mask) != 0 { 1.65 } else { 1.0 };
                            let boost = t_max * 0.25 * mult * sat;
                            t_scene[ti] += boost;
                        }
                    }
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

    // Frozen floors for this run (do not retune after BEAM): standing 0.68,
    // preference 0.50, event-session 0.42, event-turn 0.38. Lexical max is
    // captured first. 0.68 stays under the 0.72 second-anchor cut so a
    // zero-overlap rule does not become a PPR anchor. The standing floor is
    // applied to the policy turn itself, not only its session.
    let s_lexical = s_score.clone();
    let t_lexical = t_scene.clone();
    let lexical_smax = s_score.iter().cloned().fold(0.0f32, f32::max);
    let lexical_tmax = t_scene.iter().cloned().fold(0.0f32, f32::max);
    if it.task_under_rules {
        raise_floor(&mut s_score, lexical_smax, 0.68, |si| {
            ix.session_turns[si].iter().any(|&ti| ix.turns[ti].is_standing_policy)
        });
        raise_floor(&mut t_scene, lexical_tmax, 0.68, |ti| ix.turns[ti].is_standing_policy);
    }
    if pref_on {
        raise_floor(&mut s_score, lexical_smax, 0.50, |si| {
            ix.session_turns[si].iter().any(|&ti| ix.turns[ti].pref_marker && ix.turns[ti].field == 0)
        });
        raise_floor(&mut t_scene, lexical_tmax, 0.50, |ti| {
            ix.turns[ti].pref_marker && ix.turns[ti].field == 0
        });
    }
    if it.is_event_sequence {
        raise_floor(&mut s_score, lexical_smax, 0.42, |si| {
            ix.session_turns[si].iter().any(|&ti| ix.turns[ti].is_initiative)
        });
        raise_floor(&mut t_scene, lexical_tmax, 0.38, |ti| ix.turns[ti].is_initiative);
    }
    for ti in 0..nt {
        let sp = ix.turns[ti].speaker.as_str();
        if sp.is_empty() || sp == "user" || sp == "assistant" || t_scene[ti] <= 0.0 {
            continue;
        }
        if plan.groups.iter().any(|g| g == sp) {
            t_scene[ti] *= 1.35;
        }
    }
    let initiative_content: Vec<HashSet<u32>> = if it.is_event_sequence {
        ix.session_turns.iter().map(|members| {
            let mut set = HashSet::new();
            for &ti in members {
                if ix.turns[ti].is_initiative {
                    set.extend(ix.turns[ti].content.iter().copied());
                }
            }
            set
        }).collect()
    } else {
        Vec::new()
    };

    // Session-Level Multi-Hop Bridge Activation + Greedy Submodular Set Cover
    let mut bridge_boost = vec![0.0f32; ns];
    let s_ord_base = order(&s_score, &ix.session_ts);
    if p.enable_ppr && ns > 5 && smax > 0.0 {
        if let Some(&top0) = s_ord_base.first() {
            let mut anchors = vec![top0];
            if let Some(&top1) = s_ord_base.get(1) {
                if s_score[top1] >= 0.72 * smax {
                    anchors.push(top1);
                }
            }
            let max_s_df = ((ix.s_stats.n * 0.05).clamp(4.0, 24.0)) as u32;
            let mut s_act: HashMap<u32, f32> = HashMap::new();
            for &anc in &anchors {
                let anc_w = (s_score[anc] / smax).clamp(0.25, 1.0);
                // Prefer highest-scoring matching turns inside anchor session so bridge entities come from hot passage.
                let members = &ix.session_turns[anc];
                let anc_tmax = members.iter().map(|&ti| t_scene[ti]).fold(0.0f32, f32::max);
                let use_hot = anc_tmax > 0.0 && members.len() > 2;
                for &ti in members {
                    if use_hot && t_scene[ti] < 0.25 * anc_tmax {
                        continue;
                    }
                    let turn_len_norm = ((ix.turns[ti].content.len() + 6) as f32).sqrt();
                    for &e in &ix.turns[ti].entities {
                        if let Some(&df) = ix.s_stats.df.get(&e) {
                            if df >= 2 && df <= max_s_df {
                                let idf_s = 1.0 / ((df as f32 + 1.1).ln());
                                *s_act.entry(e).or_insert(0.0) += (anc_w * 2.5 * idf_s) / turn_len_norm;
                            }
                        }
                    }
                }
            }
            if !s_act.is_empty() {
                for si in 0..ns {
                    if si == top0 {
                        continue;
                    }
                    let mut ent_sum = 0.0f32;
                    for &e in &ix.session_entities[si] {
                        if let Some(&a) = s_act.get(&e) {
                            ent_sum += 2.0 * a;
                        }
                    }
                    let sum = ent_sum;
                    // Require direct query overlap (s_score > 0), or an explicit proper-entity bridge
                    // on non-preference/non-user queries, so zero-match distractors never displace personal facts.
                    let allow_zero_hit = !it.preference && !it.user_attribution && ent_sum > 0.0;
                    if sum > 0.0 && (s_score[si] > 0.0 || allow_zero_hit) {
                        let slen = (ix.session_bags[si].len[0] + ix.session_bags[si].len[1] + 8.0).sqrt();
                        let norm = sum / slen;
                        let sat = norm / (norm + 0.05);
                        let coeff = if it.multi_hop || plan.groups.len() >= 3 { 0.55 } else { 0.12 };
                        bridge_boost[si] = smax * coeff * sat;
                    }
                }
            }
        }
    }

    let final_session_indices: Vec<usize> = if p.enable_submodular && ns > 1 {
        let mut selected: Vec<usize> = Vec::with_capacity(k);
        if let Some(&top0) = s_ord_base.first() {
            if s_score[top0] > 0.0 {
                selected.push(top0);
                let mut covered_mask = s_mask[top0];
                let mut covered_ents: HashSet<u32> = ix.session_entities[top0].clone();
                let mut mask_counts: HashMap<u64, usize> = HashMap::new();
                let total_q_groups = plan.groups.len().min(64);
                let mask_limit = if total_q_groups == 64 { u64::MAX } else { (1u64 << total_q_groups) - 1 };
                *mask_counts.entry(s_mask[top0] & mask_limit).or_insert(0) += 1;
                let mut remaining: Vec<usize> = (0..ns)
                    .filter(|&si| si != top0 && (s_score[si] > 0.0 || bridge_boost[si] > 0.0))
                    .collect();

                while selected.len() < k && !remaining.is_empty() {
                    let mut best_pos = 0usize;
                    let mut best_gain = -1.0f32;
                    let mut found = false;
                    for (idx, &si) in remaining.iter().enumerate() {
                        let m_eff = s_mask[si] & mask_limit;
                        let uncovered = ((s_mask[si] & !covered_mask) & mask_limit).count_ones() as f32;
                        let uncovered_ratio = if total_q_groups > 0 {
                            uncovered / (total_q_groups as f32)
                        } else {
                            0.0
                        };
                        let same_mask_seen = mask_counts.get(&m_eff).copied().unwrap_or(0);
                        let policy_prior = s_lexical[si] <= 1e-6
                            && ix.session_turns[si].iter().any(|&t| ix.turns[t].is_standing_policy);
                        let pref_prior = s_lexical[si] <= 1e-6
                            && !policy_prior
                            && ix.session_turns[si].iter().any(|&t| ix.turns[t].pref_marker && ix.turns[t].field == 0);
                        let policy_slots = selected.iter().filter(|&&s| {
                            s_lexical[s] <= 1e-6 && ix.session_turns[s].iter().any(|&t| ix.turns[t].is_standing_policy)
                        }).count();
                        let pref_slots = selected.iter().filter(|&&s| {
                            s_lexical[s] <= 1e-6
                                && ix.session_turns[s].iter().any(|&t| ix.turns[t].pref_marker && ix.turns[t].field == 0)
                        }).count();
                        if (policy_prior && policy_slots >= 1) || (pref_prior && pref_slots >= 1) {
                            continue;
                        }
                        let shared_ent = ix.session_entities[si].iter().any(|e| covered_ents.contains(e));
                        let same_date_seen = it.is_event_sequence && selected.iter().any(|&s| ix.session_ts[s] == ix.session_ts[si] && ix.session_ts[si] > 0);
                        let topic_seen = it.is_event_sequence
                            && !initiative_content[si].is_empty()
                            && selected.iter().any(|&s| jaccard(&initiative_content[s], &initiative_content[si]) >= 0.45);
                        let new_event = it.is_event_sequence && !initiative_content[si].is_empty() && !topic_seen;
                        let redundancy = if same_date_seen {
                            0.25
                        } else if topic_seen {
                            0.35
                        } else if uncovered == 0.0 && bridge_boost[si] < 0.12 * smax && shared_ent && same_mask_seen >= 2 {
                            0.62
                        } else {
                            1.0
                        };
                        // Frozen mix for this run: new-event cover 0.65, already-seen topic 0.35.
                        let event_cover = if new_event { 0.65 } else { 0.0 };
                        let gain = (s_score[si] + bridge_boost[si]) * (1.0 + 0.90 * uncovered_ratio + event_cover) * redundancy;
                        found = true;
                        if gain > best_gain {
                            best_gain = gain;
                            best_pos = idx;
                        }
                    }
                    if !found {
                        break;
                    }
                    let chosen = remaining.swap_remove(best_pos);
                    selected.push(chosen);
                    covered_mask |= s_mask[chosen];
                    covered_ents.extend(ix.session_entities[chosen].iter().copied());
                    *mask_counts.entry(s_mask[chosen] & mask_limit).or_insert(0) += 1;
                }
            }
        }
        for si in &s_ord_base {
            if selected.len() >= k {
                break;
            }
            if !selected.contains(si) {
                selected.push(*si);
            }
        }
        selected
    } else {
        s_ord_base.into_iter().take(k).collect()
    };

    let sessions = final_session_indices
        .into_iter()
        .map(|i| RankedItem {
            id: ix.session_ids[i].clone(),
            session_id: ix.session_ids[i].clone(),
            score: s_score[i] + bridge_boost[i],
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
                let topic_repeat = it.is_event_sequence
                    && ix.turns[ti].is_initiative
                    && selected_turns.iter().any(|&prev| {
                        ix.turns[prev].is_initiative && jaccard(&ix.turns[prev].content, &ix.turns[ti].content) >= 0.45
                    });
                let t_m = t_mask[ti];
                let uncovered_mask = t_m & !covered_query_terms;
                let uncovered_count = (uncovered_mask & ((1u64 << total_query_groups) - 1)).count_ones() as f32;
                let short_reply = ix.turns[ti].content.len() <= 6
                    && selected_turns.iter().any(|&prev| {
                        ix.turns[prev].session == t_sess && (prev as isize - ti as isize).abs() == 1
                    });
                let policy_prior = t_lexical[ti] <= 1e-6 && ix.turns[ti].is_standing_policy;
                let pref_prior = t_lexical[ti] <= 1e-6
                    && ix.turns[ti].pref_marker
                    && ix.turns[ti].field == 0
                    && !policy_prior;
                let policy_slots = selected_turns.iter().filter(|&&t| t_lexical[t] <= 1e-6 && ix.turns[t].is_standing_policy).count();
                let pref_slots = selected_turns.iter().filter(|&&t| {
                    t_lexical[t] <= 1e-6 && ix.turns[t].pref_marker && ix.turns[t].field == 0
                }).count();
                if (policy_prior && policy_slots >= 1) || (pref_prior && pref_slots >= 1) {
                    continue;
                }
                let diversity_factor = if short_reply && !it.is_event_sequence {
                    1.0
                } else if count == 0 && !topic_repeat {
                    1.0
                } else if it.is_event_sequence || topic_repeat {
                    0.15
                } else if it.multi_hop && count >= 1 && uncovered_count == 0.0 {
                    0.15
                } else if count == 1 {
                    0.70
                } else {
                    0.40
                };
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
    fn parses_rfc3339_production_dates() {
        let now = chrono::Utc::now();
        let s = now.to_rfc3339();
        let ts = parse_date(&s);
        assert_eq!(ts, now.timestamp());

        let fixed = "2026-11-09T20:35:54Z";
        let ts2 = parse_date(fixed);
        assert!(ts2 > 0);
        let tokens = calendar_tokens(ts2);
        assert!(tokens.contains(&"__DATE__2026_11_09".to_string()));
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

    #[test]
    fn contraction_fragments_never_leak() {
        let toks = raw_tokens("We've checked Caroline's logs and Don didn't go, I'll stay.");
        assert!(!toks.contains(&"ve".to_string()), "{toks:?}");
        assert!(!toks.contains(&"ll".to_string()), "{toks:?}");
        assert!(toks.contains(&"caroline".to_string()), "{toks:?}");
        assert!(toks.contains(&"don".to_string()), "{toks:?}");
        assert!(is_content("don"));
    }

    #[test]
    fn proper_noun_entities_survive_stopwords() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "band1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Local H is an alternative rock duo from Illinois in the United States.")] },
                Session { id: "band2".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "For Against is an American post-punk band formed in Lincoln, Nebraska.")] },
                Session { id: "noise".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "The United States Senate held hearings against corruption.")] },
            ],
            queries: vec![Query { text: "Are Local H and For Against both from the United States?".into(), date: String::new() }],
            top_k: 3,
            params: Params::default(),
        };
        let top2: Vec<String> = rank(&req).results[0].sessions.iter().take(2).map(|s| s.id.clone()).collect();
        assert!(top2.contains(&"band1".to_string()) && top2.contains(&"band2".to_string()), "{top2:?}");
    }

    #[test]
    fn inline_dates_and_coarse_halo_match() {
        let may1_toks = inline_date_tokens("What movie did Joanna watch on 1 May, 2022?");
        assert!(may1_toks.contains(&"__DATE__2022_05_01".to_string()), "{may1_toks:?}");
        assert!(may1_toks.contains(&"__MONTH__05".to_string()), "{may1_toks:?}");

        let may2_ts = parse_date("2 May, 2022");
        let last_night = deictic_tokens(&raw_tokens("I watched it last night!"), may2_ts);
        assert!(last_night.contains(&"__DATE__2022_05_01".to_string()), "{last_night:?}");

        let mar28_ts = parse_date("2023/03/28 (Tue) 20:35");
        let four_wks = deictic_tokens(&raw_tokens("four weeks ago"), mar28_ts);
        assert!(four_wks.contains(&"__DATE__2023_03_01".to_string()), "{four_wks:?}");
    }

    #[test]
    fn framing_words_dont_crush_domain_facts() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "gold".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "I grew up with 3 sisters in a loud house.")] },
                Session { id: "noise".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "The total number of items on our order list is unknown.")] },
            ],
            queries: vec![Query { text: "What is the total number of siblings I have?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        assert_eq!(rank(&req).results[0].sessions[0].id, "gold");
    }

    #[test]
    fn ppr_turn_scores_remain_bounded_and_length_fair() {
        let filler_essay = "flask app code ".repeat(120);
        let req = RankRequest {
            sessions: vec![
                Session { id: "s1".into(), date: String::new(), timestamp: 0, turns: vec![
                    Turn { id: "gold".into(), role: "user".into(), content: "My first sprint ends on March 29 for login features.".into() },
                    Turn { id: "reply".into(), role: "assistant".into(), content: format!("Here is a sprint plan ending March 29: {filler_essay}") },
                ]},
                Session { id: "s2".into(), date: String::new(), timestamp: 0, turns: vec![
                    Turn { id: "noise".into(), role: "assistant".into(), content: format!("Another unrelated deployment guide: {filler_essay}") },
                ]},
                Session { id: "s3".into(), date: String::new(), timestamp: 0, turns: vec![
                    Turn { id: "noise2".into(), role: "user".into(), content: "Unrelated question about SQLite migrations.".into() },
                    Turn { id: "noise3".into(), role: "assistant".into(), content: "Sure, here are SQLite migration commands.".into() },
                    Turn { id: "noise4".into(), role: "user".into(), content: "Thanks!".into() },
                ]},
            ],
            queries: vec![Query { text: "When does my first sprint end?".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        let top2: Vec<String> = rank(&req).results[0].turns.iter().take(2).map(|t| t.id.clone()).collect();
        assert!(top2.contains(&"gold".to_string()), "{top2:?}");
        assert_ne!(rank(&req).results[0].turns[0].id, "noise");
    }

    #[test]
    fn session_multihop_bridges_second_hop() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "hop1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Kiss and Tell is a 1945 film starring Shirley Temple as Corliss Archer.")] },
                Session { id: "dup1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "A Kiss for Corliss is the sequel comedy film featuring Corliss Archer.")] },
                Session { id: "dup2".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Meet Corliss Archer is a radio sitcom centered on Corliss Archer.")] },
                Session { id: "dup3".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Janet Waldo voiced Corliss Archer in the radio broadcast.")] },
                Session { id: "dup4".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "F. Hugh Herbert created Corliss Archer as a character.")] },
                Session { id: "hop2".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Shirley Temple served as United States Chief of Protocol and ambassador to Ghana.")] },
                Session { id: "noise".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Baking sourdough bread at high altitude requires extra hydration.")] },
            ],
            queries: vec![Query { text: "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        let top5: Vec<String> = rank(&req).results[0].sessions.iter().map(|s| s.id.clone()).collect();
        assert_eq!(top5[0], "hop1");
        assert!(top5.contains(&"hop2".to_string()), "Expected hop2 bridged into top5: {top5:?}");
    }

    #[test]
    fn is_question_survives_trailing_code_or_markers() {
        let (is_q1, short1) = detect_question("Can you fix my Flask route? ->-> 1,14", 6);
        assert!(is_q1 && short1);
        let (is_q2, _) = detect_question("Can you initialize SQLite?\n```python\nimport sqlite3\n```", 5);
        assert!(is_q2);
    }

    #[test]
    fn previous_state_matches_and_zero_hit_bridge_skips_preference() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "gold".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "I worked as a marketing specialist in my previous role at a startup.")] },
                Session { id: "d1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Here is a general list of common occupation titles across industries.")] },
                Session { id: "d2".into(), date: String::new(), timestamp: 0, turns: vec![turn("assistant", "Waiters and teachers are common occupation categories.")] },
            ],
            queries: vec![Query { text: "What was my previous occupation?".into(), date: String::new() }],
            top_k: 3,
            params: Params::default(),
        };
        assert_eq!(rank(&req).results[0].sessions[0].id, "gold");

        let pref_req = RankRequest {
            sessions: vec![
                Session { id: "s1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Can you recommend a great show to watch on Paramount Plus tonight?")] },
                Session { id: "zero".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Paramount Plus runs on Apple TV and Roku devices.")] },
                Session { id: "pref".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "I love comedy specials and standup shows.")] },
            ],
            queries: vec![Query { text: "Can you recommend a show or movie for me to watch tonight?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let top2: Vec<String> = rank(&pref_req).results[0].sessions.iter().map(|s| s.id.clone()).collect();
        assert!(!top2.contains(&"zero".to_string()), "zero-overlap bridge should not enter preference top2: {top2:?}");
    }

    #[test]
    fn standing_policy_classifier_catches_imperatives_and_exclusions() {
        assert!(utterance_is_standing_policy("Always include the receipt hash."));
        assert!(utterance_is_standing_policy("Please format all JSON with two-space indents."));
        assert!(utterance_is_standing_policy("Never use tabs in this repo."));
        assert!(utterance_is_standing_policy("The note says to always include a hash."));
        assert!(!utterance_is_standing_policy("I like two-space indents."));
        assert_eq!(
            standing_policy_id("Always include the receipt hash."),
            standing_policy_id("  always include the receipt hash.  ")
        );
        let seq = classify("What is the chronological order of the three deploys?");
        assert!(seq.is_event_sequence);
        let mine = classify("What instruction did I tell you to always include?");
        assert!(mine.user_attribution && !mine.assistant_attribution);
    }

    #[test]
    fn standing_policy_outranks_a_longer_topic_essay() {
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "essay".into(),
                    date: "2024/06/01 (Sat) 09:00".into(),
                    timestamp: 0,
                    turns: vec![turn(
                        "assistant",
                        "Receipt hashes, formatting, and include paths show up in a long essay about build logs, formatters, and hash functions that never states a rule.",
                    )],
                },
                Session {
                    id: "rule".into(),
                    date: "2024/01/02 (Tue) 09:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "Always include the receipt hash when you format a response.")],
                },
            ],
            queries: vec![Query {
                text: "What formatting rule did I tell you to always include?".into(),
                date: String::new(),
            }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(ranked.results[0].intent.user_attribution);
        assert_eq!(ranked.results[0].turns[0].session_id, "rule");
    }

    #[test]
    fn exclusive_preference_beats_a_neutral_mention() {
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "neutral".into(),
                    date: String::new(),
                    timestamp: 1,
                    turns: vec![turn("user", "The cafe menu lists an oat milk latte and a citrus tonic.")],
                },
                Session {
                    id: "exclusive".into(),
                    date: String::new(),
                    timestamp: 2,
                    turns: vec![turn(
                        "user",
                        "I switched to oat milk only. The citrus tonic is too bitter and I avoid it.",
                    )],
                },
            ],
            queries: vec![Query {
                text: "What kind of milk would I prefer?".into(),
                date: String::new(),
            }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(ranked.results[0].intent.preference);
        assert_eq!(ranked.results[0].sessions[0].id, "exclusive");
    }

    #[test]
    fn event_sequence_spreads_top_turns_across_dates() {
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "jan".into(),
                    date: "2024/01/04 (Thu) 10:00".into(),
                    timestamp: 0,
                    turns: vec![
                        turn("user", "We shipped the ledger format to production."),
                        turn("assistant", "Noted the ledger format shipment."),
                        turn("user", "The ledger format deploy finished before noon."),
                    ],
                },
                Session {
                    id: "jan-same".into(),
                    date: "2024/01/04 (Thu) 10:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "Another note about the ledger format the same day.")],
                },
                Session {
                    id: "mar".into(),
                    date: "2024/03/02 (Sat) 11:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "We rolled the ledger format back after the audit.")],
                },
                Session {
                    id: "jun".into(),
                    date: "2024/06/09 (Sun) 11:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "We restored the ledger format with signed receipts.")],
                },
            ],
            queries: vec![Query {
                text: "What is the chronological order of the ledger format changes?".into(),
                date: String::new(),
            }],
            top_k: 3,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(ranked.results[0].intent.is_event_sequence);
        let turn_sessions: Vec<&str> = ranked.results[0].turns.iter().map(|t| t.session_id.as_str()).collect();
        let unique = turn_sessions.iter().collect::<std::collections::HashSet<_>>();
        assert_eq!(unique.len(), turn_sessions.len(), "event sequence collapsed onto one session: {turn_sessions:?}");
        let picked: Vec<&str> = ranked.results[0].sessions.iter().take(3).map(|s| s.id.as_str()).collect();
        assert!(picked.contains(&"mar") && picked.contains(&"jun"), "same-date session was not penalized: {picked:?}");
        assert!(!picked.contains(&"jan-same"), "{picked:?}");
    }

    #[test]
    fn standing_rule_surfaces_for_a_task_with_no_shared_words() {
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "formula".into(),
                    date: "2024/02/01 (Thu) 09:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "A circle area uses pi times radius squared, and textbooks repeat that formula.")],
                },
                Session {
                    id: "notes".into(),
                    date: "2024/03/01 (Fri) 09:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "Students compute circle areas by squaring the radius and multiplying.")],
                },
                Session {
                    id: "rule".into(),
                    date: "2024/01/04 (Thu) 09:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "Always include the unit symbol when you write a measurement.")],
                },
            ],
            queries: vec![Query { text: "How should I compute the area of a circle?".into(), date: String::new() }],
            top_k: 3,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(ranked.results[0].intent.task_under_rules);
        assert!(classify("What are some ways to store leftover soup?").task_under_rules);
        assert!(classify("Could you outline a backup routine?").task_under_rules);
        assert!(!classify("When did the soup boil?").task_under_rules);
        let rule_turn = ranked.results[0].turns.iter().find(|t| t.session_id == "rule");
        assert!(rule_turn.is_some(), "policy turn itself missed: {:?}", ranked.results[0].turns);
        let ids: Vec<&str> = ranked.results[0].sessions.iter().map(|s| s.id.as_str()).collect();
        assert!(ids.contains(&"rule"), "standing rule stayed at zero overlap: {ids:?}");
    }

    #[test]
    fn event_sequence_covers_distinct_user_initiatives() {
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "echo".into(),
                    date: "2024/05/01 (Wed) 09:00".into(),
                    timestamp: 0,
                    turns: vec![
                        turn("assistant", "Here is the chronological order of the workshop steps you started, as a list of steps."),
                        turn("assistant", "Another chronological order of those workshop steps you started, repeated as more steps."),
                        turn("assistant", "A third chronological order note of the workshop steps you started yesterday."),
                    ],
                },
                Session {
                    id: "sketch".into(),
                    date: "2024/01/08 (Mon) 09:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "I began sketching the cabinet joints with a pencil before cutting walnut.")],
                },
                Session {
                    id: "glue".into(),
                    date: "2024/02/12 (Mon) 09:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "I glued the walnut panels after the joints were dry and clamped.")],
                },
                Session {
                    id: "sand".into(),
                    date: "2024/03/18 (Mon) 09:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "I sanded the drawers and rubbed oil into the faces.")],
                },
            ],
            queries: vec![Query { text: "What is the chronological order of the workshop steps I started?".into(), date: String::new() }],
            top_k: 4,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(ranked.results[0].intent.is_event_sequence);
        let turns: Vec<&str> = ranked.results[0].turns.iter().take(4).map(|t| t.session_id.as_str()).collect();
        for id in ["sketch", "glue", "sand"] {
            assert!(turns.contains(&id), "missing initiative {id} in {turns:?}");
        }
    }

    #[test]
    fn planning_question_binds_an_implicit_constraint() {
        let intent = classify("How should I plan dinner for my guests?");
        assert!(intent.preference && intent.task_under_rules);
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "market".into(),
                    date: String::new(),
                    timestamp: 1,
                    turns: vec![turn("user", "Guests often eat dinner at the market after a long day.")],
                },
                Session {
                    id: "limit".into(),
                    date: String::new(),
                    timestamp: 2,
                    turns: vec![turn("user", "I don't want shellfish when guests come over for dinner.")],
                },
            ],
            queries: vec![Query { text: "How should I plan dinner for my guests?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        assert_eq!(rank(&req).results[0].sessions[0].id, "limit");
    }

    #[test]
    fn short_reply_inherits_the_neighbor_turn() {
        let req = RankRequest {
            sessions: vec![Session {
                id: "chat".into(),
                date: String::new(),
                timestamp: 1,
                turns: vec![
                    turn("alex", "The walnut cabinet needs another coat of oil."),
                    turn("blair", "Done."),
                    turn("alex", "Unrelated note about train tickets and a station platform."),
                ],
            }],
            queries: vec![Query { text: "Who finished the walnut cabinet oil?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        let ids: Vec<&str> = ranked.results[0].turns.iter().map(|t| t.id.as_str()).collect();
        assert!(ids.iter().any(|id| id.ends_with("#1")), "short reply missed the neighbor words: {ids:?}");
    }

    #[test]
    fn constraint_object_turns_preference_on_without_a_planning_cue() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "menu".into(), date: String::new(), timestamp: 1, turns: vec![turn("user", "The diner menu lists walnut pie and citrus tart.")] },
                Session { id: "limit".into(), date: String::new(), timestamp: 2, turns: vec![turn("user", "I am allergic to walnuts and skip that dessert.")] },
            ],
            queries: vec![Query { text: "Does the diner menu include walnuts?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(ranked.results[0].intent.preference);
        let ids: Vec<&str> = ranked.results[0].sessions.iter().map(|s| s.id.as_str()).collect();
        assert!(ids.contains(&"limit"), "constraint object never entered the ranking: {ids:?}");
    }

    #[test]
    fn a_dateless_question_does_not_invent_a_date_window() {
        assert!(inline_date_tokens("How do I sand a drawer?").is_empty());
        let early = inline_date_tokens("We met in early May before the show.");
        assert!(early.iter().any(|t| t == "__EARLY__05"), "{early:?}");
        assert!(early.iter().any(|t| t == "__MONTH__05"), "{early:?}");
    }

    #[test]
    fn speaker_boost_does_not_drop_the_other_speaker() {
        let req = RankRequest {
            sessions: vec![Session {
                id: "talk".into(),
                date: String::new(),
                timestamp: 1,
                turns: vec![
                    turn("alex", "Alex packed the walnut oil for the cabinet."),
                    turn("blair", "Blair packed the walnut oil for the cabinet too."),
                ],
            }],
            queries: vec![Query { text: "What did Alex pack for the cabinet?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        let ids: Vec<&str> = ranked.results[0].turns.iter().map(|t| t.id.as_str()).collect();
        assert_eq!(ids.len(), 2, "{ids:?}");
        assert!(ids[0].ends_with("#0"), "named speaker should lead: {ids:?}");
    }
}
