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

#[path = "steiner_bridge.rs"]
mod steiner_bridge;

// =========================================================================
// INPUT / OUTPUT CONTRACT
// =========================================================================

/// Exact edge. `to` is a turn id. The ranker follows it after exact-token
/// selection and does not read either side's words.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct TurnLink {
    pub kind: String,
    pub to: String,
    /// Turns that share this identity. `0` means one pair.
    #[serde(default)]
    pub degree: u32,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Turn {
    #[serde(default)]
    pub id: String,
    /// "user", "assistant", or a speaker name (LoCoMo-style dialogues).
    #[serde(default)]
    pub role: String,
    pub content: String,
    /// Recorded edges. Empty on raw benchmark text. Reply, order, and session
    /// links are written from position inside the ranker.
    #[serde(default)]
    pub links: Vec<TurnLink>,
    /// FSRS stability in days. Zero means unset. Used only to break an exact tie.
    #[serde(default)]
    pub stability: f32,
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
    /// Stratified governing channel: standing imperative policies that apply to the query
    /// without consuming episodic `sessions` or `turns` top-k slots.
    pub standing_rules: Vec<RankedItem>,
    /// The Steiner bridge moved this query's turn window.
    pub steiner_bridge_fired: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RankResponse {
    pub engine: &'static str,
    pub sessions_indexed: usize,
    pub turns_indexed: usize,
    pub results: Vec<RankResult>,
    /// Queries in this request whose turn window the Steiner bridge changed.
    pub steiner_bridge_fires: u64,
}

/// Process-wide count of queries whose turn window the Steiner bridge changed.
/// One retrieval_bench worker is one benchmark process, so this is that benchmark's fire count.
pub fn steiner_bridge_fire_count() -> u64 {
    steiner_bridge::fire_count()
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
    "got", "get", "gets", "getting", "go", "going", "gone", "much", "many",
    "lot", "lots", "thing", "things", "something", "anything", "everything", "kind", "sort",
    "well", "even", "still", "ever", "may", "might", "must", "shall", "us", "let",
    "know", "think", "tell", "told", "say", "said", "remind", "remember",
    "conversation", "mentioned", "mention", "talked", "talk", "discussed", "chat",
    "actually", "sure", "great", "thanks", "thank", "hey", "hi", "hello", "wow", "like",
    "want", "wanted", "need", "needed",
    "help", "recently", "suggest", "recommend", "useful", "interesting",
    "arent",
];

/// Query-framing / request-boilerplate tokens that should not crush domain matches via coverage^2
/// when genuine content tokens exist in the query.
const QUERY_FRAMING_WORDS: &[&str] = &[
    "name", "find", "total", "number", "amount", "count", "order", "earliest", "latest", "list",
    "items", "item", "steps", "step", "tips", "tip", "advice", "summary", "summarize",
    "comprehensive", "overview", "details", "detail", "aspects", "aspect", "different",
    "distinct", "various", "several", "ago", "days", "day", "weeks", "week", "months", "month",
    "years", "year", "earlier", "later", "throughout", "across",
    "new", "good", "use", "used", "using", "make", "made", "went", "try", "tried", "trying",
    "way", "past", "times",
];

const PREF_MARKERS: &[&str] = &[
    "love", "loved", "loves", "loving", "favorite", "favourite", "favorites", "prefer",
    "prefers", "preferred", "preference", "obsessed", "enjoy", "enjoys", "enjoyed", "enjoying",
    "hobby", "hobbies", "passionate", "passion", "collect", "collecting", "hate", "hates",
    "hated", "dislike", "dislikes", "avoid", "avoiding", "allergic", "allergy", "intolerant",
    "lactose", "celiac", "pescatarian", "gluten-free", "gluten free", "dairy-free", "dairy free",
    "diet", "vegan", "vegetarian", "big fan", "fan of", "fond", "crave", "addicted", "habit",
    "routine", "can't stand", "cannot stand", "rather", "instead of",
    "go-to", "my usual", "i usually", "i tend to", "i'm into", "im into", "i've been into",
    // Exclusion and contrast. Bare "only" is a stopword and is not a unigram marker;
    // exclusive preference is the constraint phrase, not every occurrence of the word.
    "solely", "exclusively", "stick with", "stick to", "switched to", "too bitter", "i only",
    "only ever", "only use", "only want", "only like", "only drink", "only eat", "only buy",
    "only wear",
    // Implicit personal constraints. Phrases, not the stopword "only".
    "i don't want", "i do not want", "i won't", "i will not", "rather than", "no longer",
    "not for me", "can't eat", "cannot eat", "can't have", "cannot have", "can't drink",
    "cannot drink", "don't eat", "do not eat", "don't drink", "do not drink", "never eat",
    "never drink", "stay away from", "steer clear of", "leave out", "keep out",
    "i'd prefer", "so i prefer", "trying to keep", "keep it simple", "keep it lightweight",
    "without compromising", "avoid using",
];

const HARD_CONSTRAINT_MARKERS: &[&str] = &[
    "allergic", "allergy", "intolerant", "lactose", "celiac", "vegan", "vegetarian",
    "pescatarian", "gluten-free", "gluten free", "dairy-free", "dairy free", "kosher", "halal",
    "avoid", "avoiding", "can't stand", "cannot stand", "solely", "exclusively", "stick with",
    "stick to", "switched to", "too bitter", "i only", "only ever", "only use", "only want",
    "only like", "only drink", "only eat", "only buy", "only wear", "i don't want",
    "i do not want", "i won't", "i will not", "rather than", "instead of", "no longer",
    "not for me", "can't eat", "cannot eat", "can't have", "cannot have", "can't drink",
    "cannot drink", "don't eat", "do not eat", "don't drink", "do not drink", "never eat",
    "never drink", "stay away from", "steer clear of", "leave out", "keep out",
];

const PREF_QUERY_CUES: &[&str] = &[
    "what kind of", "what type of", "what sort of", "recommend", "suggest", "suggestion",
    "prefer", "preference", "favorite", "favourite", "would i like", "would i enjoy",
    "should i get", "should i try", "should i buy", "should i choose", "should i pick",
    "should i use", "should i watch", "should i read", "should i cook", "should i make",
    " gift ", " gifts ", "present for", "suitable for me", "suited to", "tailored to",
    "how do i like", "any ideas", "any tips", "ideas for", "tips for", "advice on",
    "can you suggest", "what should i", "something i", "i might like", "i would like",
    "what would be a good", "what are some good", "help me pick", "help me find",
    "go-to", "best for me", "fit my", "match my", "based on my",
    // Personal planning, not a favorite-word lookup.
    "how should i", "help me plan", "walk me through", "help me decide", "help me choose",
    "help me figure", "what should i do",
];

const CURRENT_STATE_CUES: &[&str] = &[
    "currently", "current ", "right now", " now ", "latest", "most recent",
    "most recently", "still ", "anymore", "any more", "end up", "ended up", "these days",
    "at the moment", "where is my", "what is my", "who is my", "where do i", "what do i",
    "do i still", "did i end", "nowadays", " updated ", " new ",
];

const ASSISTANT_ATTRIBUTION_CUES: &[&str] = &[
    "you recommended", "you suggested", "you mentioned", "you told me", "you gave",
    "did you", "your recommendation", "your suggestion", "you wrote", "you said",
    "you provided", "you listed", "you recommend", "you shared", "you came up",
    "you helped", "you explained", "you described", "you named", "you created",
    "remind me", "earlier you",
    "you were telling", "you advised", "you calculated", "you found", "you picked",
];

const COLLECTIVE_CHAT_CUES: &[&str] = &[
    "our previous conversation", "our previous chat", "our conversation",
    "we discussed", "we talked", "previous chat", "last time we",
];

const USER_INITIATIVE_CUES: &[&str] = &[
    "i brought up", "did i mention", "i told you", "my instruction",
    "did i ", "what did i", "when did i", "where did i", "who did i",
    "i mentioned", "i shared", "i asked", "i said",
];

const SUMMARY_REQUEST_CUES: &[&str] = &[
    "summary of", "summarize", "everything we",
];

const TIMELINE_CUES: &[&str] = &[
    "order in which", "chronological", "timeline of", "first to last",
    "earliest to latest", "i brought up",
];

const USER_ATTRIBUTION_CUES: &[&str] = &[
    " i ", " i'm ", " i've ", " i'd ", " i'll ", " my ", " me ", "did i", "do i", "have i",
    "was i", "am i", "what should i", "i brought up", "did i mention", "my instruction",
    "i told you",
];

const MULTI_HOP_CUES: &[&str] = &[
    " both ", " and ", " also ", " in common", " together", " same ", " as well",
    " which of", " between ", " total", " combined", " altogether", " how many", " all the ",
    " each ", " how much", " across ", " compare", " versus", " vs ", " two different",
    " three different", " how many different", " in total",
];

const INFERENCE_CUES: &[&str] = &[
    "likely", "would ", "might ", "probably", "could ", "considered", "personality",
    "traits", "what might", "would be", "be interested", "is it possible", "kind of person",
];

const WHEN_INTERROGATIVE_CUES: &[&str] = &[
    "when did ", "when didn't ", "when didnt ",
    "when was ", "when wasn't ", "when wasnt ",
    "when were ", "when weren't ", "when werent ",
    "when is ", "when isn't ", "when isnt ",
    "when are ", "when aren't ", "when arent ",
    "when will ", "when won't ", "when wont ",
    "when does ", "when doesn't ", "when doesnt ",
    "when do ", "when don't ", "when dont ",
    "when has ", "when hasn't ", "when hasnt ",
    "when have ", "when haven't ", "when havent ",
    "when had ", "when hadn't ", "when hadnt ",
    "when can ", "when can't ", "when cant ", "when cannot ",
];

const TEMPORAL_REST_CUES: &[&str] = &[
    "how long", "how many days", "how many weeks", "how many months",
    "how many years", "before ", "after ", "first ", "last ", " ago ", " date ", " dates ",
    "which year", "what year", "what month", " earlier ", " later ", "since ",
];

/// Downstream tasks: the user wants an answer produced (a procedure, list,
/// recommendation, calculation, or draft), so stored imperative rules apply.
/// Episodic/factual recall ("when did", "when was", "what did") is excluded in `classify`.
const TASK_UNDER_RULES_CUES: &[&str] = &[
    "how should", "how do i", "how can i", "how to ",
    "what are some", "what can you", "what should", "what steps",
    "can you", "could you", "would you",
    "help me", "walk me through", "show me",
    "recommend", "suggest", " i should",
    "should i ", "shouldn't i ", "shouldnt i ",
    "best practices",
    "outline ", "draft ", "write a ", "format ", "calculate ", "compute ",
];

const RECALL_NOT_TASK_CUES: &[&str] = &[
    "what did", "where did", "who did", "when did", "why did", "how did",
    "when was", "when were", "where was", "who was", "how many", "how much did", "how long",
    "did i", "did you", "have i", "had i", "you said", "you told", "you mentioned",
    "you recommended", "you suggested",
    "order in which", "chronological", "timeline of", "first to last", "earliest to latest",
    "summary of", "summarize", "everything we", "throughout our", "across our",
    "has developed", "have developed", "i brought up", "i mentioned",
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
    ("OUTDOORS", &["hike", "hiking", "trail", "kayak", "kayaking", "camping", "camp", "tent", "backpack", "backpacking", "outdoor", "outdoors", "gear", "park", "wilderness", "climb", "climbing", "mountain", "beach", "lake", "fishing", "nature", "roadtrip", "activity", "activities", "hobby", "hobbies"]),
    ("FITNESS", &["gym", "weight", "workout", "running", "runner", "run", "jog", "jogging", "cycling", "bike", "bicycle", "swim", "swimming", "triathlon", "marathon", "5k", "10k", "exercise", "cardio", "fitness", "yoga", "pilates", "crossfit", "training", "race", "tournament", "competition", "athletic", "athletics", "activity", "activities", "hobby", "hobbies"]),
    ("VISUAL_ART", &["painting", "paint", "acrylic", "watercolor", "watercolour", "oil", "sketch", "sketching", "draw", "drawing", "pottery", "ceramic", "clay", "studio", "sculpture", "gallery", "canvas", "portrait", "landscape", "art", "artistic", "artist", "medium", "easel", "craft", "knitting", "sewing", "exhibition", "activity", "activities", "hobby", "hobbies"]),
    ("MUSIC", &["guitar", "piano", "vinyl", "concert", "festival", "album", "spotify", "band", "jazz", "rock", "classical", "instrument", "clarinet", "flute", "cello", "saxophone", "trumpet", "drum", "bass", "violin", "orchestra", "musician", "song", "singing", "vocalist", "music", "playlist", "podcast", "streaming", "dance", "ballet", "salsa", "hobby", "hobbies"]),
    ("PETS", &["dog", "puppy", "cat", "kitten", "vet", "pet", "rescue", "adopt", "adoption", "animal", "retriever", "breed", "leash", "veterinary", "veterinarian", "fish", "bird", "hamster", "horse"]),
    ("BOOKS", &["book", "reading", "read", "novel", "author", "fiction", "nonfiction", "memoir", "library", "bookstore", "kindle", "audiobook", "literature", "poetry", "poem", "writing", "writer", "publish", "publication", "publications", "conference", "conferences", "paper", "papers", "article", "articles", "proceedings", "journal", "journals"]),
    ("GAMING", &["game", "gaming", "gamer", "playstation", "nintendo", "switch", "xbox", "steam", "rpg", "boardgame", "tabletop", "console", "chess", "puzzle", "videogame"]),
    ("TRAVEL", &["travel", "traveling", "flight", "airline", "trip", "vacation", "hotel", "hostel", "explore", "exploring", "passport", "tourist", "destination", "luggage", "itinerary", "cruise", "resort", "abroad", "museum", "theater"]),
    ("PHOTOGRAPHY", &["photo", "photography", "camera", "lens", "shoot", "canon", "nikon", "sony", "shutter", "aperture", "exposure", "darkroom", "instagram", "mirrorless", "tripod"]),
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
    ("CAREER", &["job", "career", "boss", "coworker", "interview", "offer", "resume", "promote", "promotion", "salary", "office", "remote", "work", "working", "firm", "client", "company", "manager", "project", "startup", "business", "profession", "occupation", "occupations", "vocation", "agency", "role", "roles", "milestone", "milestones", "contract", "specialist"]),
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
    " because ", " since ", " so that ", " due to ", " wanted ", " needed ", " decided ", " helped ",
    " inspired ", " to help ", " reason ", " in order ", " purpose ", " aim ", " aims ", " aimed ",
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

fn collapse_thousands_commas(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    for (i, ch) in text.char_indices() {
        if ch == ','
            && i > 0
            && bytes[i - 1].is_ascii_digit()
            && i + 3 < bytes.len()
            && bytes[i + 1].is_ascii_digit()
            && bytes[i + 2].is_ascii_digit()
            && bytes[i + 3].is_ascii_digit()
            && (i + 4 == bytes.len() || !bytes[i + 4].is_ascii_digit())
        {
            continue;
        }
        out.push(ch);
    }
    out
}

/// Strip an ASCII `v` only when it introduces a dotted version (`v5.3.0` -> `5.3.0`,
/// `(v2.1)` -> `(2.1)`). The `v` must be at the start or after a non-alphanumeric,
/// and must be followed by digits, a dot, and a digit. Bare `v8` / `v2` / `v3` stay
/// intact, and `dev5.3` is left alone because its `v` is inside a word.
fn strip_dotted_version_v(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut prev_non_alnum = true;
    for (i, ch) in text.char_indices() {
        if ch == 'v' && prev_non_alnum {
            let rest = &bytes[i + 1..];
            let mut j = 0;
            while j < rest.len() && rest[j].is_ascii_digit() {
                j += 1;
            }
            if j > 0 && j + 1 < rest.len() && rest[j] == b'.' && rest[j + 1].is_ascii_digit() {
                continue;
            }
        }
        out.push(ch);
        prev_non_alnum = !ch.is_alphanumeric();
    }
    out
}

fn emit_token(out: &mut Vec<String>, tok: &str) {
    match tok {
        "2fa" | "twofa" | "twofactor" => {
            out.push("two".to_string());
            out.push("factor".to_string());
        }
        "mfa" | "multifactor" => {
            out.push("multi".to_string());
            out.push("factor".to_string());
        }
        _ => out.push(tok.to_string()),
    }
}

fn fold_two_factor(tokens: &mut [String]) {
    let mut i = 0;
    while i + 1 < tokens.len() {
        if tokens[i] == "2" && (tokens[i + 1] == "factor" || tokens[i + 1] == "factors") {
            tokens[i] = "two".to_string();
        } else if tokens[i] == "2" && tokens[i + 1] == "fa" {
            tokens[i] = "two".to_string();
            tokens[i + 1] = "factor".to_string();
        }
        i += 1;
    }
}

fn raw_tokens(text: &str) -> Vec<String> {
    let lower = strip_dotted_version_v(&collapse_thousands_commas(&text.to_lowercase()));
    let mut out = Vec::new();
    for piece in lower.split(|c: char| !c.is_alphanumeric() && c != '\'' && c != '’') {
        if piece.is_empty() {
            continue;
        }
        let cleaned = clean_contraction(piece);
        if cleaned.contains('\'') || cleaned.contains('’') {
            if cleaned.ends_with("n't") || cleaned.ends_with("n’t") {
                let joined: String = cleaned.chars().filter(|&c| c != '\'' && c != '’').collect();
                if !joined.is_empty() {
                    emit_token(&mut out, &joined);
                }
            } else {
                for sub in cleaned.split(|c: char| c == '\'' || c == '’') {
                    if !sub.is_empty() {
                        emit_token(&mut out, sub);
                    }
                }
            }
        } else if !cleaned.is_empty() {
            emit_token(&mut out, cleaned);
        }
    }
    fold_two_factor(&mut out);
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
    "well", "also", "just", "now", "then", "still", "even", "maybe", "perhaps", "actually",
    "however", "though", "although", "because", "since", "after", "before", "during", "while",
    "today", "yesterday", "tomorrow", "tonight", "always", "never", "please", "let", "lets",
    "not", "only", "very", "really", "wow", "oh", "okay", "ok", "right", "true", "same",
    "another", "other", "more", "some", "many", "most", "every", "each", "both", "all", "any",
    "here", "from", "with", "into", "over", "under", "about", "again", "once", "twice",
];

const TITLE_CONNECTORS: &[&str] = &["and", "of", "the", "for", "in", "on", "at"];

/// Extract leading document title/header line when a multi-line passage starts with a compact
/// header (`"{Title}\n{Body}"`, excluding `[BLIP caption]`), stripping trailing parenthetical
/// disambiguators (`"Armada (novel)"` -> `"Armada"`).
fn prose_without_blip(text: &str) -> &str {
    text.split_once("\n[BLIP caption]").map(|(body, _)| body).unwrap_or(text)
}

fn split_blip_caption(text: &str) -> (&str, Option<&str>) {
    if let Some((body, cap)) = text.split_once("\n[BLIP caption]") {
        let trimmed = cap.trim();
        (body, (!trimmed.is_empty()).then_some(trimmed))
    } else {
        (text, None)
    }
}

fn extract_header_titles(text: &str) -> Option<(&str, &str)> {
    let prose = prose_without_blip(text);
    let (first_line, rest) = prose.split_once('\n')?;
    if rest.trim().is_empty() {
        return None;
    }
    let header = first_line.trim();
    if header.is_empty()
        || header.len() > 96
        || header.starts_with('[')
        || header.starts_with("```")
        || header.ends_with(['.', '?', '!', ';'])
    {
        return None;
    }
    let word_count = header.split_whitespace().count();
    if !(1..=10).contains(&word_count) {
        return None;
    }
    if !header.chars().next().is_some_and(|c| c.is_ascii_uppercase() || c.is_ascii_digit()) {
        return None;
    }
    let core = match header.find(" (") {
        Some(idx) if header.ends_with(')') && idx > 0 => header[..idx].trim(),
        _ => header,
    };
    if core.is_empty() { None } else { Some((header, core)) }
}

fn extract_header_title(text: &str) -> Option<&str> {
    extract_header_titles(text).map(|(_, core)| core)
}

fn norm_span_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

fn hash_u32_slice(tokens: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &tok in tokens {
        h ^= u64::from(tok);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Extract contiguous capitalized proper-noun entity phrases ("\u{1}ent:local_h", "\u{1}ent:kiss_and_tell")
/// plus mid-clause single-token proper names ("\u{1}ent:ghana", "\u{1}ent:melanie") and header titles.
/// Preserves short single-letter entities ("H" in "Local H"), capitalized prepositions ("For Against"),
/// and title-case connectors ("Kiss and Tell", "Lord of the Rings").
fn proper_entity_tokens(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    if let Some(core_title) = extract_header_title(text) {
        let parts: Vec<String> = core_title
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase())
            .collect();
        if !parts.is_empty() && parts.len() <= 6 {
            out.push(format!("\u{1}ent:{}", parts.join("_")));
        }
    }

    let mut run: Vec<String> = Vec::new();
    let mut run_started_clause = true;
    let mut at_clause_start;

    let flush_run = |run: &mut Vec<String>, started_clause: bool, out: &mut Vec<String>| {
        // Trim trailing lowercase connectors if the run ended on a preposition like "Kiss and"
        while run.last().is_some_and(|s| s.starts_with('\u{2}')) {
            run.pop();
        }
        if run.len() >= 2 {
            for part in run.iter() {
                if !part.starts_with('\u{2}')
                    && part.len() >= 3
                    && !STARTER_STOP.contains(&part.as_str())
                    && !stopwords().contains(part.as_str())
                    && !MONTHS.contains(&part.as_str())
                    && !WEEKDAYS.contains(&part.as_str())
                {
                    out.push(format!("\u{1}ent:{part}"));
                }
            }
            for win in run.windows(2) {
                if !win[0].starts_with('\u{2}') && !win[1].starts_with('\u{2}') {
                    out.push(format!("\u{1}ent:{}_{}", win[0], win[1]));
                }
            }
            if run.len() <= 4 {
                let clean_parts: Vec<&str> = run.iter().map(|s| s.trim_start_matches('\u{2}')).collect();
                out.push(format!("\u{1}ent:{}", clean_parts.join("_")));
            }
        } else if run.len() == 1 && !started_clause {
            let tok = &run[0];
            if tok.len() >= 3
                && !STARTER_STOP.contains(&tok.as_str())
                && !stopwords().contains(tok.as_str())
                && !MONTHS.contains(&tok.as_str())
                && !WEEKDAYS.contains(&tok.as_str())
            {
                out.push(format!("\u{1}ent:{tok}"));
            }
        }
        run.clear();
    };

    for line in text.lines() {
        at_clause_start = true;
        for raw_word in line.split_whitespace() {
            let ends_clause = raw_word.ends_with(['.', '?', '!', ',', ';', ':']);
            let starts_quote = raw_word.starts_with(['"', '\'', '“', '‘', '(', '[']);
            let w = raw_word.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '’' && c != '-');
            if w.is_empty() {
                flush_run(&mut run, run_started_clause, &mut out);
                at_clause_start = true;
                continue;
            }
            let clean = clean_contraction(w);
            let first_upper = clean.chars().next().is_some_and(|c| c.is_ascii_uppercase());
            let all_upper_acronym = clean.len() >= 2
                && clean.len() <= 6
                && clean.chars().all(|c| c.is_ascii_uppercase());
            let lower = clean.to_lowercase();
            let norm: String = lower.chars().filter(|c| c.is_alphanumeric()).collect();
            if first_upper && !norm.is_empty() && !STARTER_STOP.contains(&norm.as_str()) {
                if run.is_empty() {
                    run_started_clause = (at_clause_start || starts_quote) && !all_upper_acronym;
                }
                run.push(norm);
            } else if !run.is_empty() && TITLE_CONNECTORS.contains(&norm.as_str()) {
                // Allow up to 2 title connectors inside an active proper-noun chain (e.g. "Lord of the Rings").
                let trailing_conns = run.iter().rev().take_while(|s| s.starts_with('\u{2}')).count();
                if trailing_conns < 2 {
                    run.push(format!("\u{2}{norm}"));
                } else {
                    flush_run(&mut run, run_started_clause, &mut out);
                }
            } else {
                flush_run(&mut run, run_started_clause, &mut out);
            }
            if ends_clause {
                flush_run(&mut run, run_started_clause, &mut out);
                at_clause_start = raw_word.ends_with(['.', '?', '!', ';', ':']);
            } else {
                at_clause_start = false;
            }
        }
        flush_run(&mut run, run_started_clause, &mut out);
    }
    out
}

fn irregular_lemma(w: &str) -> Option<&'static str> {
    match w {
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
        "career" | "careers" => Some("career"),
        _ => None,
    }
}

fn expand_abbrev<'a>(w: &'a str) -> &'a str {
    match w {
        "edu" => "education",
        "fam" => "family",
        "pic" | "pics" => "picture",
        "uni" | "unis" => "university",
        "congrats" => "congratulations",
        "info" => "information",
        "auth" | "authn" => "authentication",
        "authz" => "authorization",
        "config" | "configs" | "cfg" => "configuration",
        "repo" | "repos" => "repository",
        "db" | "dbs" => "database",
        "app" | "apps" => "application",
        "env" | "envs" => "environment",
        "bday" | "bdays" => "birthday",
        _ => w,
    }
}

fn uk_exact(w: &str) -> Option<&'static str> {
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

fn uk_suffix(w: &str) -> Option<String> {
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

fn our_tail(tail: &str) -> bool {
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

fn map_our(w: &str) -> Option<String> {
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
                let head = &root[..root.len() - 3];
                best = Some((tail.len(), format!("{prefix}{head}or{tail}")));
            }
            from = i + 1;
        }
    }
    best.map(|(_, s)| s)
}

fn map_yse(w: &str) -> Option<String> {
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

fn ise_blocked(lemma: &str) -> bool {
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

fn map_ise(w: &str) -> Option<String> {
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

fn uk_to_us(w: &str) -> Option<String> {
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

fn strip_tech_major(w: &str) -> &str {
    let b = w.as_bytes();
    if !(4..=11).contains(&b.len()) {
        return w;
    }
    let last = b[b.len() - 1];
    if !matches!(last, b'1'..=b'6') {
        return w;
    }
    let head = &w[..w.len() - 1];
    if head.bytes().all(|c| c.is_ascii_lowercase())
        && matches!(
            head,
            "html" | "css" | "http" | "oauth" | "sqlite" | "python" | "ipv" | "webgl" | "vue"
        )
    {
        head
    } else {
        w
    }
}

fn classic_stem(w: &str) -> String {
    if w.len() <= 3 || !w.is_ascii() {
        return w.to_string();
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

/// Deterministic suffix stripper (Porter-style + irregular verb lemmas + conversational
/// abbreviations + UK->US orthography + tech major version strip, zero model).
pub fn stem(word: &str) -> String {
    let lower = word.to_lowercase();
    if let Some(base) = irregular_lemma(&lower) {
        return base.to_string();
    }
    let expanded = expand_abbrev(&lower);
    let normalized = uk_to_us(expanded).unwrap_or_else(|| expanded.to_string());
    let surface = strip_tech_major(&normalized);
    let stemmed = classic_stem(surface);
    let again = strip_tech_major(&stemmed);
    if again.len() == stemmed.len() {
        stemmed
    } else {
        classic_stem(again)
    }
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

fn season_code(month: u32) -> &'static str {
    match month {
        3..=5 => "SP",
        6..=8 => "SU",
        9..=11 => "FA",
        _ => "WI",
    }
}

fn calendar_tokens(ts: i64) -> Vec<String> {
    use chrono::Datelike;
    let Some(dt) = chrono::DateTime::from_timestamp(ts, 0) else { return Vec::new() };
    let d = dt.date_naive();
    let sc = season_code(d.month());
    let sy = if d.month() == 12 { d.year() + 1 } else { d.year() };
    vec![
        d.day().to_string(),
        MONTHS[d.month0() as usize].to_string(),
        d.year().to_string(),
        WEEKDAYS[d.weekday().num_days_from_monday() as usize].to_string(),
        format!("__DATE__{}_{:02}_{:02}", d.year(), d.month(), d.day()),
        format!("__YM__{}_{:02}", d.year(), d.month()),
        format!("__MONTH__{:02}", d.month()),
        format!("__MD__{:02}_{:02}", d.month(), d.day()),
        format!("__{}__{:02}", month_band(d.day()), d.month()),
        format!("__YEAR__{}", d.year()),
        format!("__SEASON__{sc}"),
        format!("__SY__{}_{sc}", d.year()),
        format!("__SY__{sy}_{sc}"),
    ]
}

fn month_band(day: u32) -> &'static str {
    if day <= 10 { "EARLY" } else if day >= 21 { "LATE" } else { "MID" }
}

fn month_index(tok: &str) -> Option<(u32, bool)> {
    match tok {
        "january" | "jan" => Some((1, false)),
        "february" | "feb" => Some((2, false)),
        "march" => Some((3, true)),
        "mar" => Some((3, false)),
        "april" | "apr" => Some((4, false)),
        "may" => Some((5, true)),
        "june" | "jun" => Some((6, false)),
        "july" | "jul" => Some((7, false)),
        "august" | "aug" => Some((8, false)),
        "september" | "sep" | "sept" => Some((9, false)),
        "october" | "oct" => Some((10, false)),
        "november" | "nov" => Some((11, false)),
        "december" | "dec" => Some((12, false)),
        _ => None,
    }
}

fn parse_day_token(tok: &str) -> Option<u32> {
    let stripped = tok
        .strip_suffix("st")
        .or_else(|| tok.strip_suffix("nd"))
        .or_else(|| tok.strip_suffix("rd"))
        .or_else(|| tok.strip_suffix("th"))
        .unwrap_or(tok);
    if let Ok(n) = stripped.parse::<u32>() {
        if (1..=31).contains(&n) {
            return Some(n);
        }
    }
    match tok {
        "first" => Some(1),
        "second" => Some(2),
        "third" => Some(3),
        "fourth" | "forth" => Some(4),
        "fifth" => Some(5),
        "sixth" => Some(6),
        "seventh" => Some(7),
        "eighth" => Some(8),
        "ninth" => Some(9),
        "tenth" => Some(10),
        "eleventh" => Some(11),
        "twelfth" => Some(12),
        "thirteenth" => Some(13),
        "fourteenth" => Some(14),
        "fifteenth" => Some(15),
        "sixteenth" => Some(16),
        "seventeenth" => Some(17),
        "eighteenth" => Some(18),
        "nineteenth" => Some(19),
        "twentieth" => Some(20),
        "thirtieth" => Some(30),
        _ => None,
    }
}

/// Extract explicit calendar dates written directly inside prose/queries
/// (e.g. "1 May, 2022", "May 1st, 2022", "the 15th of March", "in May", "summer of 2023", "2022-05-01").
fn inline_date_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let lower = text.to_lowercase();
    // Match ISO YYYY-MM-DD or YYYY/MM/DD
    for w in lower.split_whitespace() {
        let trimmed = w.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '/');
        if trimmed.len() == 10 {
            let b = trimmed.as_bytes();
            if (b.get(4) == Some(&b'-') && b.get(7) == Some(&b'-'))
                || (b.get(4) == Some(&b'/') && b.get(7) == Some(&b'/'))
            {
                let iso = trimmed.replace('/', "-");
                if let Ok(d) = chrono::NaiveDate::parse_from_str(&iso, "%Y-%m-%d") {
                    if let Some(dt) = d.and_hms_opt(0, 0, 0) {
                        out.extend(calendar_tokens(dt.and_utc().timestamp()));
                    }
                }
            }
        }
    }
    let orig_toks: Vec<&str> = text
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();
    let toks: Vec<String> = orig_toks.iter().map(|s| s.to_lowercase()).collect();

    const TEMP_PREPS: &[&str] = &[
        "in", "on", "of", "during", "since", "before", "after", "until", "by", "from", "through",
        "around", "this", "last", "next", "past", "every", "each", "late", "early", "mid", "middle",
    ];
    const NON_YEAR_PREV: &[&str] = &[
        "room", "suite", "apt", "unit", "section", "port", "error", "code", "rfc", "iso", "line",
        "page", "usd", "eur", "model", "version",
    ];
    const NON_YEAR_NEXT: &[&str] = &[
        "km", "kilometers", "mi", "miles", "ft", "feet", "meters", "px", "gb", "mb", "kb", "tb",
        "dollars", "euros", "percent", "percents", "points", "hours", "minutes", "seconds",
    ];

    for i in 0..toks.len() {
        // Explicit 4-digit calendar year ("in 2022", "back in 2019", "since 2020") with HeidelTime guards
        if toks[i].len() == 4 && toks[i].chars().all(|c| c.is_ascii_digit()) {
            if let Ok(y) = toks[i].parse::<i32>() {
                if (1900..=2099).contains(&y) {
                    let prev_bad = i
                        .checked_sub(1)
                        .is_some_and(|k| NON_YEAR_PREV.contains(&toks[k].as_str()));
                    let next_bad = toks
                        .get(i + 1)
                        .is_some_and(|nx| NON_YEAR_NEXT.contains(&nx.as_str()));
                    if !prev_bad && !next_bad {
                        out.push(format!("__YEAR__{y}"));
                    }
                }
            }
        }

        // Seasonal anchors ("summer of 2023", "in the winter", "last spring")
        let season = match toks[i].as_str() {
            "summer" => Some(("SU", false)),
            "winter" => Some(("WI", false)),
            "autumn" => Some(("FA", false)),
            "spring" => Some(("SP", true)),
            "fall" => Some(("FA", true)),
            _ => None,
        };
        if let Some((sc, ambig_season)) = season {
            let prev1 = i.checked_sub(1).map(|k| toks[k].as_str()).unwrap_or("");
            let prev2 = i.checked_sub(2).map(|k| toks[k].as_str()).unwrap_or("");
            let y_after = if toks.get(i + 1).map(String::as_str) == Some("of") {
                toks.get(i + 2)
            } else {
                toks.get(i + 1)
            }
            .and_then(|s| (s.len() == 4).then(|| s.parse::<i32>().ok()).flatten())
            .filter(|&y| (1900..=2099).contains(&y));

            if let Some(y) = y_after {
                out.push(format!("__SY__{y}_{sc}"));
                out.push(format!("__SEASON__{sc}"));
                out.push(format!("__YEAR__{y}"));
            } else if !ambig_season
                || matches!(prev1, "this" | "last" | "next" | "past" | "every" | "early" | "late" | "mid")
                || (prev1 == "the" && matches!(prev2, "in" | "during" | "since" | "before" | "after"))
            {
                out.push(format!("__SEASON__{sc}"));
            }
        }

        let Some((m, ambig_month)) = month_index(&toks[i]) else { continue };
        let band = i
            .checked_sub(1)
            .and_then(|k| match toks[k].as_str() {
                "early" | "earlier" => Some("EARLY"),
                "late" | "later" => Some("LATE"),
                "mid" | "middle" => Some("MID"),
                "of" if k >= 1 => match toks[k - 1].as_str() {
                    "beginning" | "start" => Some("EARLY"),
                    "middle" => Some("MID"),
                    "end" => Some("LATE"),
                    _ => None,
                },
                _ => None,
            });
        if let Some(b) = band {
            out.push(format!("__{b}__{m:02}"));
            out.push(format!("__MONTH__{m:02}"));
        }

        // "<day> [of] <month> [of] <year>" OR "<month> [the] <day> [of] <year>"
        let prev_num = i.checked_sub(1).and_then(|k| {
            if toks[k] == "of" && k >= 1 {
                parse_day_token(&toks[k - 1])
            } else {
                parse_day_token(&toks[k])
            }
        });
        let (next_num, day_end_idx) = if toks.get(i + 1).map(String::as_str) == Some("the") {
            (toks.get(i + 2).and_then(|s| parse_day_token(s)), i + 2)
        } else {
            (toks.get(i + 1).and_then(|s| parse_day_token(s)), i + 1)
        };
        let year_at = |idx: usize| -> Option<i32> {
            let cand = if toks.get(idx).map(String::as_str) == Some("of") {
                toks.get(idx + 1)
            } else {
                toks.get(idx)
            };
            cand.and_then(|s| (s.len() == 4).then(|| s.parse::<i32>().ok()).flatten())
                .filter(|&y| (1900..=2099).contains(&y))
        };
        let year_after_month = year_at(i + 1);
        let year_after_next = year_at(day_end_idx + 1);

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
            out.push(format!("__YEAR__{y}"));
        } else if let Some(day) = prev_num.or(next_num) {
            out.push(format!("__MD__{m:02}_{day:02}"));
            out.push(format!("__MONTH__{m:02}"));
        } else if band.is_none() {
            // Bare month in prose/query ("in May", "during July", "last October").
            // Guard ambiguous "may"/"march" with preposition or capitalization + non-modal context.
            let prev1 = i.checked_sub(1).map(|k| toks[k].as_str()).unwrap_or("");
            let is_cap = orig_toks
                .get(i)
                .and_then(|s| s.chars().next())
                .is_some_and(|c| c.is_ascii_uppercase());
            if !ambig_month || TEMP_PREPS.contains(&prev1) || (is_cap && i > 0) {
                out.push(format!("__MONTH__{m:02}"));
            }
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
            "before" | "earlier" if prev == "day" => specs.push((-DAY, 0.9, false)),
            "before" if prev == "week" => specs.push((-7 * DAY, 0.9, true)),
            "before" if prev == "month" => specs.push((-30 * DAY, 0.9, true)),
            "before" if prev == "year" => specs.push((-365 * DAY, 0.9, true)),
            "after" | "later" if prev == "day" && (i < 2 || raw[i - 2] == "the" || raw[i - 2] == "a") => {
                specs.push((DAY, 0.9, false));
            }
            "after" | "following" if prev == "week" && (i < 2 || raw[i - 2] == "the" || raw[i - 2] == "a") => {
                specs.push((7 * DAY, 0.9, true));
            }
            "tomorrow" => specs.push((DAY, 1.0, false)),
            "today" | "tonight" => specs.push((0, 1.0, false)),
            "week" if matches!(prev, "last" | "past" | "previous") => specs.push((-7 * DAY, 1.0, true)),
            "week" if matches!(prev, "next" | "coming" | "following") => specs.push((7 * DAY, 1.0, true)),
            "week" if prev == "this" && i >= 2 && matches!(raw[i - 2].as_str(), "earlier" | "early") => {
                specs.push((-2 * DAY, 0.9, true));
            }
            "weekend" if matches!(prev, "last" | "this" | "past" | "previous") => specs.push((-2 * DAY, 1.0, true)),
            "weekend" if matches!(prev, "next" | "coming") => specs.push((5 * DAY, 0.9, true)),
            "month" if matches!(prev, "last" | "past" | "previous") => {
                specs.push((-30 * DAY, 1.0, true));
                specs.push((-15 * DAY, 0.85, false));
            }
            "month" if prev == "this" && i >= 2 && matches!(raw[i - 2].as_str(), "earlier" | "early") => {
                specs.push((-7 * DAY, 0.9, true));
            }
            "month" if matches!(prev, "next" | "coming" | "following") => specs.push((30 * DAY, 1.0, true)),
            "year" if matches!(prev, "last" | "past" | "previous") => specs.push((-365 * DAY, 1.0, true)),
            "year" if matches!(prev, "next" | "coming" | "following") => specs.push((365 * DAY, 1.0, true)),
            "ago" | "earlier" | "later" => {
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
                    let dir = if w == "later" { 1 } else { -1 };
                    if n > 0 {
                        specs.push((dir * n * unit * DAY, 1.0, coarse));
                        if q == "few" && unit >= 7 {
                            specs.push((dir * (n - 1) * unit * DAY, 0.65, true));
                            specs.push((dir * (n + 1) * unit * DAY, 0.65, true));
                        }
                    }
                }
            }
            "days" | "weeks" | "months" | "years" if !matches!(next, "ago" | "earlier" | "later") => {
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
                if matches!(prev, "last" | "this" | "next" | "past" | "previous" | "coming" | "following")
                    && WEEKDAYS.contains(&w.as_str())
                {
                    use chrono::Datelike;
                    if let Some(dt) = chrono::DateTime::from_timestamp(ts, 0) {
                        let today = dt.weekday().num_days_from_monday() as i64;
                        let target = WEEKDAYS.iter().position(|d| d == w).unwrap_or(0) as i64;
                        let is_future = matches!(prev, "next" | "coming" | "following");
                        let is_strict_past = matches!(prev, "last" | "past" | "previous");
                        let back = ((today - target + 7) % 7).max(if is_strict_past { 1 } else { 0 });
                        let fwd = ((target - today + 7) % 7).max(if is_future { 1 } else { 0 });
                        let delta = if is_future { fwd } else { -back };
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
                    if tok.starts_with("__DATE__")
                        || tok.starts_with("__YM__")
                        || tok.starts_with("__MONTH__")
                        || tok.starts_with("__MD__")
                    {
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
    tf: [HashMap<u32, f32>; 3],
    len: [f32; 3],
}

impl Bag {
    fn add(&mut self, field: usize, f: Feat, scale: f32) {
        *self.tf[field].entry(f.id).or_insert(0.0) += f.w * scale;
        if f.counts_len {
            self.len[field] += f.w * scale;
        }
    }

    fn add_inherited(&mut self, field: usize, f: Feat, scale: f32) {
        *self.tf[field].entry(f.id).or_insert(0.0) += f.w * scale;
        if f.counts_len {
            self.len[field] += 0.50 * f.w * scale;
        }
    }
}

struct StandingRule {
    policy_key: String,
    turn: usize,
    session: usize,
    ts: i64,
    has_explicit_scope: bool,
    scope: HashSet<u32>,
    domains: Vec<&'static str>,
    assistant_nouns: HashSet<u32>,
}

struct TurnIx {
    session: usize,
    id: String,
    field: usize,
    feats: Vec<Feat>,
    blip_feats: Vec<Feat>,
    /// Exact/stem term ids of content tokens in order, for span proximity.
    seq: Vec<(u32, u32)>,
    content: HashSet<u32>,
    /// Capitalized proper-entity term IDs inside this turn.
    entities: HashSet<u32>,
    /// Body text (header line removed) for lazy exact title-span linking on hot anchors.
    body_for_headers: String,
    span_ids_cache: OnceLock<Vec<u32>>,
    speaker: String,
    is_question: bool,
    is_pure_info_question: bool,
    pre_question_ids: HashSet<u32>,
    question_clause_ids: HashSet<u32>,
    answers_question: bool,
    is_short_reply: bool,
    is_backchannel: bool,
    anaphoric: bool,
    pref_marker: bool,
    is_personal_constraint: bool,
    is_user_schema: bool,
    schema_domains: Vec<&'static str>,
    has_temporal: bool,
    has_numeric: bool,
    has_spatial: bool,
    has_causal: bool,
    is_short_question: bool,
    is_standing_policy: bool,
    /// User turn that opens a concrete event, not a rule and not a backchannel.
    is_initiative: bool,
    /// Previous turn in this session with a different role. Position and role only.
    /// Backward only. A later turn points at the earlier one. There is no forward edge.
    answers: Option<usize>,
    /// Caller & ingest-time RSB edges resolved to `(degree, target)` — strictly backward.
    exact_edges: Vec<(u32, usize)>,
    /// SalienceJoin v2: earlier standing-policy turns this later turn points at.
    /// Promoted only when that policy is a standing rule that fired for the query.
    policy_backlinks: Vec<usize>,
    /// Backward state-revision / contradiction edges (`t_new -> t_old`).
    supersedes_turns: Vec<usize>,
    stability: f32,
    /// ISO date or HH:MM in the turn text. Character form, not a month name.
    form_clock: bool,
    /// An ASCII digit in the turn text. Number words do not count.
    form_digit: bool,
    /// An @handle, a URL, or a path token.
    form_handle: bool,
    /// A capitalized word of two or more letters that is not sentence-initial.
    form_midcap: bool,
    raw_links: Vec<TurnLink>,
}

struct Stats {
    n: f32,
    df: HashMap<u32, u32>,
    avg: [f32; 3],
}

fn stats_of(bags: &[Bag]) -> Stats {
    let mut df: HashMap<u32, u32> = HashMap::new();
    let mut sum = [0.0f32; 3];
    let mut cnt = [0.0f32; 3];
    for b in bags {
        let mut seen: HashSet<u32> = HashSet::new();
        for f in 0..3 {
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
            if cnt[2] > 0.0 { sum[2] / cnt[2] } else { 1.0 },
        ],
    }
}

struct HeaderKey {
    tokens: Vec<u32>,
    sessions: Vec<(usize, f32)>,
}

struct Index {
    vocab: Vocab,
    session_ids: Vec<String>,
    session_ts: Vec<i64>,
    session_turns: Vec<Vec<usize>>,
    session_entities: Vec<HashSet<u32>>,
    session_content: Vec<HashSet<u32>>,
    /// Rarest non-hub content token IDs per session (sorted, up to 32) for GhostLink Ochiai divergence.
    session_rare_nt: Vec<Vec<u32>>,
    /// Backward-only session RSB edges (`later_session -> (degree, earlier_session)`).
    session_rsb_back: Vec<Vec<(u32, usize)>>,
    /// Backward-only session contradiction/supersession edges (`s_new -> s_old`).
    session_supersedes_back: Vec<Vec<usize>>,
    /// Backward-only Temporal Context Episode neighbor (`si -> si - 1` when sharing episode/date anchor).
    session_episode_prev: Vec<Option<usize>>,
    turns: Vec<TurnIx>,
    session_bags: Vec<Bag>,
    turn_own_bags: Vec<Bag>,
    turn_bags: Vec<Bag>,
    /// (session index, first turn position within session, last turn position inclusive)
    windows: Vec<(usize, usize, usize)>,
    window_bags: Vec<Bag>,
    s_stats: Stats,
    t_own_stats: Stats,
    t_stats: Stats,
    w_stats: Stats,
    /// Content-term ids that are the object of a personal constraint ("allergic to X").
    constraint_terms: HashSet<u32>,
    /// Compiled SOAR/Rete standing rules for the stratified governing channel (`RankResult.standing_rules`).
    standing_rules: Vec<StandingRule>,
    /// Non-hub hypernym domains (`session_df <= 0.12 * N` when `N >= 16`).
    non_hub_domains: HashSet<&'static str>,
    /// Exact `{Header}\\n{Body}` titles, hashed, for the session bridge.
    header_dict: HashMap<u64, Vec<HeaderKey>>,
    header_max_words: usize,
    /// Unique title-case speaker nickname bindings (`Mel` -> `melanie`).
    nicks: SpeakerNickIndex,
    /// Recall cues: per turn, the sorted ids of the kernel tokens (`raw_tokens` + `stem`) of its
    /// full content, in a private id space (`cue_vocab`) so the ranker vocabulary is untouched.
    turn_cues: Vec<Vec<u32>>,
    cue_vocab: HashMap<String, u32>,
    /// Cue ids that occur in any turn's role / speaker field (structural metadata, not a list).
    role_cues: HashSet<u32>,
    /// Exact bridge tokens per turn: capitalized names, technical handles, and
    /// digit strings that survived the Steiner guard. Sorted, deduplicated.
    bridge_tokens: Vec<Vec<u32>>,
}

/// Kernel tokens of `text` for recall-cue matching: the same tokenizer and stemmer as ranking.
fn cue_tokens(text: &str) -> Vec<String> {
    raw_tokens(text).into_iter().map(|w| stem(&w)).collect()
}

fn intern_cues(text: &str, vocab: &mut HashMap<String, u32>) -> Vec<u32> {
    let mut ids: Vec<u32> = cue_tokens(text)
        .into_iter()
        .map(|w| {
            let next = vocab.len() as u32;
            *vocab.entry(w).or_insert(next)
        })
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
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
/// Parse both compact `key:value` tokens and structured `Key: Value` clauses
/// (excluding URL schemes `http:`/`https:` and `[BLIP caption]`), emitting exact
/// `(slot_key, slot_value)` pairs so typed attributes affect ranking and supersession.
fn typed_slot_tokens(text: &str) -> Vec<(String, String)> {
    const SKIP_KEYS: &[&str] = &["http", "https", "mailto", "file", "note", "example", "output", "input"];
    let mut out = Vec::new();
    for line in text.lines() {
        let trimmed_line = line.trim();
        if trimmed_line.starts_with('[') || trimmed_line.starts_with("```") {
            continue;
        }
        // Check structured `Key: Value` at line/clause head (up to 2 words for key, 1-2 words for value).
        if let Some((lhs, rhs)) = trimmed_line.split_once(':') {
            let key_words: Vec<&str> = lhs.split_whitespace().collect();
            if (1..=2).contains(&key_words.len())
                && key_words.iter().all(|w| w.len() >= 2 && w.chars().all(|c| c.is_ascii_alphabetic() || c == '_' || c == '-'))
            {
                let norm_key = key_words.join("_").to_lowercase();
                if !SKIP_KEYS.contains(&norm_key.as_str()) {
                    for val in rhs
                        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-' && c != '$')
                        .filter(|w| !w.is_empty())
                        .take(2)
                    {
                        let clean_val = val.trim_matches('$').to_lowercase();
                        if clean_val.len() >= 2 && !stopwords().contains(clean_val.as_str()) {
                            out.push((norm_key.clone(), clean_val));
                        }
                    }
                }
            }
        }
        // Also check inline `key:value` tokens without spaces.
        for raw in trimmed_line.split_whitespace() {
            let tok = raw.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != ':' && c != '_' && c != '-');
            let Some((key, value)) = tok.split_once(':') else { continue };
            let kl = key.to_lowercase();
            if kl.len() >= 2
                && kl.chars().all(|c| c.is_ascii_alphabetic())
                && !SKIP_KEYS.contains(&kl.as_str())
                && !value.is_empty()
                && value.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            {
                out.push((kl, value.to_lowercase()));
            }
        }
    }
    out
}

fn constraint_object_tokens(lower: &str) -> Vec<String> {
    const LEADS: &[&str] = &[
        "don't want ", "dont want ", "do not want ",
        "allergic to ", "intolerant to ",
        "can't eat ", "cant eat ", "cannot eat ",
        "can't have ", "cant have ", "cannot have ",
        "can't drink ", "cant drink ", "cannot drink ",
        "don't eat ", "dont eat ", "do not eat ",
        "don't drink ", "dont drink ", "do not drink ",
        "never eat ", "never drink ",
        "stay away from ", "steer clear of ",
    ];
    const PSEUDO_VERBS: &[&str] = &[
        "bother", "intrude", "waste", "bore", "jinx", "derail", "worry", "know", "think",
    ];
    let mut out = Vec::new();
    for lead in LEADS {
        let mut rest = lower;
        while let Some(at) = rest.find(lead) {
            let after = &rest[at + lead.len()..];
            // Bound ConText scope at clause/exception boundaries ("but", "except", punctuation).
            let clause = after
                .split(['.', ',', ';', '!', '?', '\n'])
                .next()
                .unwrap_or(after);
            let scoped = clause
                .split(" but ")
                .next()
                .unwrap_or(clause)
                .split(" except ")
                .next()
                .unwrap_or(clause);
            let cands: Vec<&str> = scoped
                .split(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
                .filter(|w| !w.is_empty() && is_content(w))
                .take(2)
                .collect();
            if !cands.first().is_some_and(|w| PSEUDO_VERBS.contains(w)) {
                out.extend(cands.into_iter().map(str::to_string));
            }
            rest = &rest[at + lead.len()..];
        }
    }
    out
}

/// Imperative user rule. Grounded in Searle directives + SWBD-DAMSL / ISO 24617-2 + NegEx
/// pseudo-trigger exclusion: rejects questions, epistemic hedges ("don't think/know"),
/// downplayers ("don't worry", "never mind"), auto-feedback checks ("make sure I understand"),
/// and elliptical perfects ("never been/seen/heard", "always wanted/been" via VBN/VBD check).
pub fn utterance_is_standing_policy(text: &str) -> bool {
    const IRREGULAR_PAST_PARTICIPLES: &[&str] = &[
        "been", "was", "were", "had", "done", "did", "seen", "saw", "heard", "known", "knew",
        "thought", "gone", "went", "got", "gotten", "met", "found", "felt", "meant", "lost",
        "paid", "said", "sold", "sent", "spent", "built", "caught", "taught", "bought", "brought",
        "told", "took", "taken", "gave", "given", "ate", "eaten", "drank", "drunk", "wrote",
        "written", "spoke", "spoken", "broke", "broken", "chose", "chosen", "drove", "driven",
        "fell", "fallen", "flew", "flown", "forgot", "forgotten", "grew", "grown", "understood",
        "won",
    ];
    const DONT_PSEUDO_NEXT: &[&str] = &[
        "worry", "sweat", "mention", "fret", "bother", "know", "remember", "recall", "recollect",
        "understand", "think", "believe", "suppose", "imagine", "reckon", "guess", "expect",
        "feel", "doubt", "mind", "care", "get", "take", "have", "kid", "ask", "want", "you",
    ];

    let prose = text.split("```").next().unwrap_or(text).trim();
    if prose.is_empty() || prose.ends_with('?') {
        return false;
    }
    let lower = prose.to_lowercase();
    let trimmed = lower
        .trim_start()
        .strip_prefix("please ")
        .map(str::trim_start)
        .unwrap_or_else(|| lower.trim_start());

    if trimmed.starts_with("never mind") || trimmed.starts_with("nevermind") {
        return false;
    }

    let is_non_imperative_past = |verb: &str| -> bool {
        if IRREGULAR_PAST_PARTICIPLES.contains(&verb) {
            return true;
        }
        verb.len() > 4
            && verb.ends_with("ed")
            && !matches!(verb, "need" | "feed" | "seed" | "proceed" | "succeed" | "exceed" | "embed" | "shred")
    };

    if let Some(rest) = trimmed.strip_prefix("always ").or_else(|| trimmed.strip_prefix("never ")) {
        let next = rest
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
            .find(|s| !s.is_empty())
            .unwrap_or("");
        return !next.is_empty() && !is_non_imperative_past(next);
    }

    if let Some(rest) = trimmed.strip_prefix("don't ").or_else(|| trimmed.strip_prefix("do not ")) {
        let next = rest
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
            .find(|s| !s.is_empty())
            .unwrap_or("");
        if next.is_empty() || DONT_PSEUDO_NEXT.contains(&next) {
            return false;
        }
        return true;
    }

    if let Some(rest) = trimmed.strip_prefix("make sure").or_else(|| trimmed.strip_prefix("ensure ")) {
        let after = rest.trim_start();
        let after_that = after.strip_prefix("that ").map(str::trim_start).unwrap_or(after);
        if after_that.starts_with("i ")
            || after_that.starts_with("i'm ")
            || after_that.starts_with("we're on the same")
            || after_that.starts_with("we are on the same")
        {
            return false;
        }
        return true;
    }

    trimmed.starts_with("remember to ")
        || trimmed.starts_with("from now on")
        || trimmed.starts_with("going forward")
        || lower.contains("always format")
        || lower.contains("always include")
        || lower.contains("always use")
        || lower.contains("never use")
        || lower.contains("never include")
        || lower.contains("format all")
}

/// Extract `(has_explicit_scope, scope_words)` of a standing imperative rule for Rete specificity.
/// Conditional subordinators (`when`, `whenever`, `if`, `for all/every/any`, `in all`) restrict
/// firing to matching queries; unscoped universal directives remain eligible on `task_under_rules`.
fn extract_rule_scope(text: &str) -> (bool, Vec<String>) {
    const CONDITIONAL_CLAUSES: &[&str] = &[
        "when ",
        "whenever ",
        "if ",
        "for all ",
        "for every ",
        "for any ",
        "in all ",
    ];
    const DIRECTIVE_VERBS: &[&str] = &[
        "always", "never", "please", "make", "sure", "ensure", "remember", "format", "include",
        "provide", "give", "write", "use", "using", "keep", "response", "responses", "output",
        "outputs", "answer", "answers",
    ];
    let (prose, _) = split_blip_caption(text);
    let lower = format!(" {} ", prose.to_lowercase());
    for marker in CONDITIONAL_CLAUSES {
        let pat = format!(" {marker}");
        if let Some(idx) = lower.find(&pat) {
            let after = &lower[idx + pat.len()..];
            let clause = after.split(['.', '!', '?', ';', '\n']).next().unwrap_or(after);
            let toks: Vec<String> = raw_tokens(clause)
                .into_iter()
                .filter(|w| is_content(w) && !DIRECTIVE_VERBS.contains(&w.as_str()))
                .collect();
            if !toks.is_empty() {
                return (true, toks);
            }
        }
    }
    let toks = raw_tokens(prose)
        .into_iter()
        .filter(|w| is_content(w) && !DIRECTIVE_VERBS.contains(&w.as_str()))
        .collect();
    (false, toks)
}

/// Pure byte-exact technical handle extractor for Retroactive Salience Backfill (RSB):
/// URLs, emails, commit SHAs (7..=40 hex with >=1 alpha + >=1 digit), issue refs (`PROJ-123`),
/// version tags (`v1.2.3`), and file paths (`dir/file.ext`). Zero word similarity.
fn extract_exact_technical_handles(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for raw_tok in text.split_whitespace() {
        let tok = raw_tok.trim_matches(|c: char| matches!(c, '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>' | ',' | ';' | '"' | '\'' | '`'));
        let tok_clean = tok.trim_end_matches('.');
        if tok_clean.len() < 3 {
            continue;
        }
        // 1. URL
        if tok_clean.starts_with("http://") || tok_clean.starts_with("https://") {
            let h = format!("\u{1}h:url:{}", tok_clean.to_lowercase());
            if seen.insert(h.clone()) {
                out.push(h);
            }
            continue;
        }
        // 2. Email
        if let Some((local, dom)) = tok_clean.split_once('@') {
            if !local.is_empty()
                && dom.contains('.')
                && !dom.starts_with('.')
                && !dom.ends_with('.')
                && local.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '%' | '+' | '-'))
                && dom.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-'))
            {
                let h = format!("\u{1}h:email:{}", tok_clean.to_lowercase());
                if seen.insert(h.clone()) {
                    out.push(h);
                }
                continue;
            }
        }
        // 3. IssueRef (`ABC-123`)
        if let Some((proj, num)) = tok_clean.split_once('-') {
            if (2..=10).contains(&proj.len())
                && proj.chars().next().is_some_and(|c| c.is_ascii_uppercase())
                && proj.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
                && !num.is_empty()
                && num.len() <= 7
                && num.chars().all(|c| c.is_ascii_digit())
            {
                let h = format!("\u{1}h:issue:{}", tok_clean.to_lowercase());
                if seen.insert(h.clone()) {
                    out.push(h);
                }
                continue;
            }
        }
        // 4. CommitSha (7..=40 hex with >=1 hex letter and >=1 digit)
        if (7..=40).contains(&tok_clean.len())
            && tok_clean.chars().all(|c| c.is_ascii_hexdigit())
            && tok_clean.chars().any(|c| c.is_ascii_digit())
            && tok_clean.chars().any(|c| matches!(c.to_ascii_lowercase(), 'a'..='f'))
        {
            let h = format!("\u{1}h:sha:{}", &tok_clean.to_lowercase()[..7.min(tok_clean.len())]);
            if seen.insert(h.clone()) {
                out.push(h);
            }
            continue;
        }
        // 5. Version (`v1.2` or `1.2.3`)
        let ver_cand = tok_clean.strip_prefix('v').or_else(|| tok_clean.strip_prefix('V')).unwrap_or(tok_clean);
        let ver_parts: Vec<&str> = ver_cand.split('.').collect();
        if ver_parts.len() >= 2
            && (tok_clean.starts_with('v') || tok_clean.starts_with('V') || ver_parts.len() >= 3)
            && ver_parts.iter().all(|p| !p.is_empty() && p.len() <= 5 && p.chars().all(|c| c.is_ascii_digit()))
        {
            let h = format!("\u{1}h:ver:{ver_cand}");
            if seen.insert(h.clone()) {
                out.push(h);
            }
            continue;
        }
        // 6. FilePath (`dir/file.ext` or `src\file.rs`)
        let norm_path = tok_clean.replace('\\', "/");
        if norm_path.contains('/')
            && !norm_path.contains("://")
            && norm_path.chars().any(|c| c.is_ascii_alphabetic())
            && norm_path.rsplit('/').next().is_some_and(|leaf| leaf.contains('.') && leaf.len() >= 3)
        {
            let h = format!("\u{1}h:path:{}", norm_path.to_lowercase());
            if seen.insert(h.clone()) {
                out.push(h);
            }
        }
    }
    out
}

/// Technical-handle term ids (`\u{1}h:…`) are linked only by backward RSB edges.
/// Undirected PPR must not reactivate them, or a later zero-overlap session is pulled forward.
fn is_technical_handle(vocab: &Vocab, id: u32) -> bool {
    vocab.names.get(id as usize).is_some_and(|name| name.starts_with("\u{1}h:"))
}

/// Episode prefix before `_s{digits}` when sub-sessions belong to one contiguous batch,
/// else empty string (never inspects benchmark labels).
fn session_episode_key<'a>(session_id: &'a str, date: &'a str) -> String {
    if let Some((prefix, suffix)) = session_id.rsplit_once("_s") {
        if !prefix.is_empty() && !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
            return format!("ep:{prefix}");
        }
    }
    let trimmed = date.trim();
    if !trimmed.is_empty() {
        return format!("dt:{trimmed}");
    }
    String::new()
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

const INFO_Q_WORDS: &[&str] = &[
    "who", "whom", "whose", "what", "where", "when", "why", "which", "how",
    "did", "do", "does", "is", "are", "was", "were",
    "have", "has", "had", "can", "could", "would", "should", "will",
    "anything", "anyone", "anybody", "anywhere", "anytime",
    "any", "ever", "still",
    "hey", "hi",
];

const Q_LEAD_WORDS: &[&str] = &[
    "wow", "oh", "aw", "aww", "ooh", "whoa", "woah", "hmm", "hm", "omg",
    "haha", "hahaha", "lol", "lmao",
    "nice", "cool", "awesome", "congrats", "congratulations",
    "yay", "sweet", "amazing", "wonderful", "fantastic",
    "thanks", "thank", "sorry", "glad", "ouch", "yikes",
    "hey", "hi", "hello", "yo",
    "so", "and", "but", "well",
    "okay", "ok", "yeah", "yes", "yep", "nope",
    "right", "great", "good",
];

const Q_REACTION_HEADS: &[&str] = &[
    "oh", "ah", "wow", "haha", "hahaha", "sounds", "looks", "seems",
];
const Q_REACTION_TAILS: &[&str] = &[
    "fun", "cool", "nice", "great", "good", "awesome", "amazing",
    "wonderful", "cute", "sweet", "lovely",
];

fn take_word(text: &str) -> Option<(&str, &str)> {
    let text = text.trim_start();
    let end = text
        .char_indices()
        .find(|(_, c)| !c.is_ascii_alphanumeric())
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    if end == 0 { None } else { Some((&text[..end], &text[end..])) }
}

fn skip_comma(text: &str) -> &str {
    let s = text.trim_start();
    s.strip_prefix(',').unwrap_or(s).trim_start()
}

fn is_speaker_alias(tok: &str, names: &HashSet<String>) -> bool {
    if names.contains(tok) {
        return true;
    }
    if tok.len() < 3 || stopwords().contains(tok) {
        return false;
    }
    let st = stem(tok);
    if hypernyms().contains_key(st.as_str()) {
        return false;
    }
    names.iter().any(|n| n.len() >= tok.len() + 3 && n.starts_with(tok))
}

fn strip_one_question_lead<'a>(text: &'a str, names: &HashSet<String>) -> Option<&'a str> {
    let s = text.trim_start();
    let (tok, rest) = take_word(s)?;
    let low = tok.to_ascii_lowercase();
    if is_speaker_alias(&low, names) && rest.trim_start().starts_with(',') {
        return Some(skip_comma(rest));
    }
    if matches!(low.as_str(), "hey" | "hi" | "hello") {
        let after = skip_comma(rest);
        if let Some((name, nrest)) = take_word(after) {
            if is_speaker_alias(&name.to_ascii_lowercase(), names) {
                return Some(skip_comma(nrest));
            }
        }
        return Some(skip_comma(rest));
    }
    if Q_REACTION_HEADS.contains(&low.as_str()) {
        if let Some((tail, trest)) = take_word(skip_comma(rest)) {
            let tail_l = tail.to_ascii_lowercase();
            if Q_REACTION_TAILS.contains(&tail_l.as_str()) {
                return Some(skip_comma(trest));
            }
        }
    }
    if Q_LEAD_WORDS.contains(&low.as_str()) {
        return Some(skip_comma(rest));
    }
    None
}

fn info_question_start(clause: &str) -> bool {
    let s = clause.trim_start();
    let Some((tok, rest)) = take_word(s) else { return false };
    let low = tok.to_ascii_lowercase();
    if low == "what" {
        if let Some((next, _)) = take_word(skip_comma(rest)) {
            let n = next.to_ascii_lowercase();
            if n == "a" || n == "an" {
                return false;
            }
        }
    }
    INFO_Q_WORDS.contains(&low.as_str())
}

fn stripped_question_clause<'a>(clause: &'a str, names: &HashSet<String>) -> &'a str {
    let mut s = clause.trim();
    for _ in 0..8 {
        if info_question_start(s) {
            return s;
        }
        match strip_one_question_lead(s, names) {
            Some(next) => {
                let next = next.trim_start();
                if next.is_empty() || next.len() == s.len() {
                    return s;
                }
                s = next;
            }
            None => return s,
        }
    }
    s
}

fn last_sentence_boundary(head: &str) -> Option<usize> {
    let bytes = head.as_bytes();
    let mut end = head.len();
    while end > 0 {
        let rel = head[..end].rfind(|c: char| matches!(c, '.' | '!' | ';' | '\n'))?;
        let ok = match bytes[rel] {
            b'.' => {
                let prev_digit = rel > 0 && bytes[rel - 1].is_ascii_digit();
                let next_ok = head[rel + 1..].chars().next().is_none_or(|c| c.is_whitespace());
                !prev_digit && next_ok
            }
            _ => true,
        };
        if ok {
            return Some(rel);
        }
        end = rel;
    }
    None
}

fn terminal_question_spans(prose: &str) -> Option<(&str, &str)> {
    let prose = prose.split("```").next().unwrap_or(prose);
    let q_pos = prose.rfind('?')?;
    let tail = prose[q_pos + 1..].trim();
    if !(tail.is_empty() || tail.chars().all(|c| !c.is_ascii_alphabetic())) {
        return None;
    }
    let head = &prose[..q_pos];
    match last_sentence_boundary(head) {
        Some(i) => Some((&head[..i], head[i + 1..].trim())),
        None => Some(("", head.trim())),
    }
}

fn token_is_temporal_word(w: &str) -> bool {
    MONTHS.contains(&w)
        || WEEKDAYS.contains(&w)
        || [
            "yesterday", "tomorrow", "ago", "today", "tonight",
            "week", "weeks", "weekend", "month", "months", "year", "years",
            "days", "spring", "summer", "autumn", "winter",
            "mon", "tue", "wed", "thu", "fri", "sat", "sun",
        ]
        .contains(&w)
        || (w.len() == 4 && w.chars().all(|c| c.is_ascii_digit()))
}

fn pre_has_answer_payload(pre: &str) -> bool {
    if pre.chars().any(|c| c.is_ascii_digit()) {
        return true;
    }
    raw_tokens(pre).iter().any(|w| NUMBER_WORDS.contains(&w.as_str()) || token_is_temporal_word(w))
}

fn content_term_ids(vocab: &mut Vocab, text: &str, stemming: bool) -> HashSet<u32> {
    let mut ids = HashSet::new();
    for w in raw_tokens(text).into_iter().filter(|w| is_content(w)) {
        ids.insert(vocab.intern(&w));
        if stemming {
            ids.insert(vocab.intern(&stem_term(&stem(&w))));
        }
    }
    ids
}

struct QuestionShape {
    pre_ids: HashSet<u32>,
    clause_ids: HashSet<u32>,
    is_pure: bool,
}

fn analyze_question(
    vocab: &mut Vocab,
    prose: &str,
    is_question: bool,
    stemming: bool,
    names: &HashSet<String>,
) -> QuestionShape {
    let lower = prose.to_ascii_lowercase();
    let legacy = is_question && info_question_start(lower.trim_start());
    let tag = {
        let t = lower.trim();
        t.ends_with("right?") || t.ends_with("yeah?") || t.ends_with("huh?")
    };
    let Some((pre, clause)) = terminal_question_spans(prose) else {
        return QuestionShape {
            pre_ids: HashSet::new(),
            clause_ids: HashSet::new(),
            is_pure: is_question && legacy && !tag,
        };
    };
    let stripped = stripped_question_clause(clause, names);
    let terminal_info = info_question_start(stripped);
    let payload = pre_has_answer_payload(pre);
    QuestionShape {
        pre_ids: content_term_ids(vocab, pre, stemming),
        clause_ids: content_term_ids(vocab, clause, stemming),
        is_pure: is_question && !tag && !payload && (legacy || terminal_info),
    }
}

fn habitual_perfect(lower: &str) -> bool {
    const LEADS: &[&str] = &["i've been ", "i’ve been ", "ive been "];
    const PROGRESS: &[&str] = &[
        "running into", "running out", "trying", "working", "debugging", "struggling",
        "thinking", "feeling", "looking", "wondering", "dealing", "waiting", "fixing",
        "testing", "starting", "planning", "considering", "hoping", "attempting", "stuck",
    ];
    for lead in LEADS {
        let mut rest = lower;
        while let Some(at) = rest.find(lead) {
            let after = rest[at + lead.len()..].trim_start();
            let progress = PROGRESS.iter().any(|p| after.starts_with(p));
            let next = after
                .split(|c: char| !c.is_ascii_alphanumeric())
                .find(|s| !s.is_empty())
                .unwrap_or("");
            if !next.is_empty() && !progress {
                return true;
            }
            rest = &rest[at + lead.len()..];
        }
    }
    false
}

fn clause_has_manner(clause: &str) -> bool {
    const PHRASES: &[&str] = &["step-by-step", "step by step", "without compromising", "avoid using"];
    if PHRASES.iter().any(|p| clause.contains(p)) {
        return true;
    }
    const WORDS: &[&str] = &[
        "keep", "simple", "lightweight", "pragmatic", "minimal", "avoid", "rather", "prefer",
    ];
    clause
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '\'')
        .any(|w| WORDS.contains(&w))
}

fn hedged_like_is_preference(lower: &str) -> bool {
    const LEADS: &[&str] = &["i would like", "i'd like", "i’d like", "id like"];
    const TASK: &[&str] = &[
        "add", "implement", "build", "create", "write", "make", "fix", "set", "help",
        "include", "integrate", "show", "calculate", "organize", "update", "install",
        "configure", "deploy", "code", "debug", "change", "start", "finish", "complete",
        "do", "get", "find", "use", "put", "give", "send", "run",
    ];
    for lead in LEADS {
        let mut rest = lower;
        while let Some(at) = rest.find(lead) {
            let after = rest[at + lead.len()..].trim_start();
            let clause = after.split(['.', '!', '?', '\n']).next().unwrap_or(after);
            let end = clause.char_indices().nth(96).map(|(i, _)| i).unwrap_or(clause.len());
            let window = &clause[..end];
            if clause_has_manner(window) {
                return true;
            }
            let head = window.trim_start().strip_prefix("to ").unwrap_or(window.trim_start());
            let verb_end = head.find(|c: char| !c.is_ascii_alphanumeric()).unwrap_or(head.len());
            let verb = &head[..verb_end];
            if !verb.is_empty() && !TASK.contains(&verb) {
                return true;
            }
            rest = &rest[at + lead.len()..];
        }
    }
    false
}

fn commitment_manner(lower: &str) -> bool {
    const MANNERS: &[&str] = &["lightweight", "step-by-step", "step by step", "pragmatic"];
    const COMMITS: &[&str] = &[
        "i prefer", "i'd prefer", "id prefer", "trying to keep", "keep it", "keep my",
        "i'd like", "i’d like", "i would like", "without compromising", "avoid using", "i avoid",
    ];
    lower.split(['.', '!', '?', '\n']).any(|sentence| {
        MANNERS.iter().any(|m| sentence.contains(m)) && COMMITS.iter().any(|c| sentence.contains(c))
    })
}

fn user_preference_speech_act(lower: &str, raw: &[String], pref_set: &HashSet<&str>) -> bool {
    raw.iter().any(|w| pref_set.contains(w.as_str()))
        || PREF_MARKERS.iter().any(|m| m.contains(' ') && lower.contains(m))
        || hedged_like_is_preference(lower)
        || commitment_manner(lower)
}

fn camel_case_parts(ident: &str) -> Vec<String> {
    let chars: Vec<char> = ident.chars().collect();
    let mut seen_lower = false;
    let mut gated = false;
    for &c in &chars {
        if c.is_ascii_lowercase() {
            seen_lower = true;
        } else if seen_lower && c.is_ascii_uppercase() {
            gated = true;
            break;
        }
    }
    if !gated {
        return Vec::new();
    }
    let mut raw_parts: Vec<String> = Vec::new();
    let mut start = 0usize;
    for i in 1..chars.len() {
        let prev = chars[i - 1];
        let cur = chars[i];
        let hump = (prev.is_ascii_lowercase() || prev.is_ascii_digit()) && cur.is_ascii_uppercase();
        let acronym_tail = prev.is_ascii_uppercase()
            && cur.is_ascii_uppercase()
            && chars.get(i + 1).is_some_and(|n| n.is_ascii_lowercase());
        if hump || acronym_tail {
            raw_parts.push(chars[start..i].iter().collect());
            start = i;
        }
    }
    raw_parts.push(chars[start..].iter().collect());
    if raw_parts.len() < 2 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for part in raw_parts {
        let lower = part.to_lowercase();
        if lower.chars().count() < 3 || !is_content(&lower) || out.contains(&lower) {
            continue;
        }
        out.push(lower);
    }
    out
}

fn surface_aux_tokens(word: &str) -> Vec<(String, f32)> {
    let mut out = Vec::new();
    if word.contains('_') || word.contains('-') {
        let joined: String = word
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();
        if (4..=48).contains(&joined.len()) && joined.chars().any(|c| c.is_ascii_alphabetic()) {
            out.push((joined, 1.0));
        }
    }
    for piece in word.split(|c: char| !c.is_ascii_alphanumeric()) {
        if piece.is_empty() {
            continue;
        }
        for part in camel_case_parts(piece) {
            if !out.iter().any(|(p, _)| p == &part) {
                out.push((part, 0.85));
            }
        }
    }
    out
}

const NICK_TAIL: usize = 3;

struct SpeakerNickIndex {
    prefix_of: HashMap<String, String>,
    witnessed: HashMap<String, HashSet<String>>,
    lower_attested: HashSet<String>,
}

fn common_speaker_tokens() -> &'static HashSet<String> {
    static S: OnceLock<HashSet<String>> = OnceLock::new();
    S.get_or_init(|| {
        let mut s = HashSet::new();
        for w in STOPWORDS
            .iter()
            .chain(STARTER_STOP)
            .chain(QUERY_FRAMING_WORDS)
            .chain(MONTHS.iter())
            .chain(WEEKDAYS.iter())
        {
            s.insert((*w).to_string());
        }
        for (_, words) in HYPERNYM_LATTICE {
            for w in *words {
                s.insert((*w).to_string());
            }
        }
        for word in MONTHS.iter().chain(WEEKDAYS.iter()) {
            for len in 2..word.len() {
                s.insert(word[..len].to_string());
            }
        }
        s
    })
}

fn common_speaker_token(w: &str) -> bool {
    common_speaker_tokens().contains(w)
}

fn inflectional_tail(nick: &str, full: &str) -> bool {
    matches!(
        full.strip_prefix(nick),
        Some("s" | "es" | "ed" | "ing" | "er" | "ers" | "ly")
    )
}

struct SurfaceTok {
    lower: String,
    title_case: bool,
    sentence_initial: bool,
}

fn surface_tokens(text: &str) -> Vec<SurfaceTok> {
    let mut out = Vec::new();
    let mut sentence_initial = true;
    let mut current = String::new();
    let flush = |current: &mut String, sentence_initial: bool, out: &mut Vec<SurfaceTok>| {
        if current.is_empty() {
            return;
        }
        let piece = std::mem::take(current);
        let owned = piece.clone();
        let lower_piece = owned.to_lowercase();
        let cleaned_lower = clean_contraction(&lower_piece);
        let cleaned_raw = if cleaned_lower.len() == lower_piece.len() {
            owned
        } else {
            let keep = cleaned_lower.chars().count();
            owned.chars().take(keep).collect()
        };
        if cleaned_raw.is_empty() {
            return;
        }
        let mut chars = cleaned_raw.chars();
        let first_upper = chars.next().is_some_and(|c| c.is_ascii_uppercase());
        let rest_lower = chars.all(|c| !c.is_ascii_uppercase());
        let lower: String = cleaned_raw.chars().flat_map(|c| c.to_lowercase()).collect();
        if lower.is_empty() {
            return;
        }
        out.push(SurfaceTok {
            lower,
            title_case: first_upper && rest_lower,
            sentence_initial,
        });
    };
    for ch in text.chars() {
        let keep = ch.is_alphanumeric() || ch == '\'' || ch == '’';
        if keep {
            current.push(ch);
            continue;
        }
        let was_empty = current.is_empty();
        flush(&mut current, sentence_initial, &mut out);
        if matches!(ch, '.' | '?' | '!' | '\n') {
            sentence_initial = true;
        } else if !was_empty {
            sentence_initial = false;
        }
    }
    flush(&mut current, sentence_initial, &mut out);
    out
}

impl SpeakerNickIndex {
    fn build(sessions: &[Session]) -> Self {
        let mut exact_roles: HashSet<String> = HashSet::new();
        let mut targets: HashSet<String> = HashSet::new();
        let mut lower_attested: HashSet<String> = HashSet::new();
        let mut texts: Vec<&str> = Vec::new();
        for s in sessions {
            for t in &s.turns {
                texts.push(t.content.as_str());
                let role = t.role.trim();
                if role.is_empty()
                    || role.eq_ignore_ascii_case("user")
                    || role.eq_ignore_ascii_case("assistant")
                {
                    continue;
                }
                for w in raw_tokens(role) {
                    exact_roles.insert(w.clone());
                    if w.len() >= 4
                        && !STARTER_STOP.contains(&w.as_str())
                        && !stopwords().contains(w.as_str())
                    {
                        targets.insert(w);
                    }
                }
            }
        }
        for text in &texts {
            for tok in surface_tokens(text) {
                if !tok.title_case {
                    lower_attested.insert(tok.lower);
                }
            }
        }
        let mut hits: HashMap<String, Vec<String>> = HashMap::new();
        let mut ordered: Vec<&String> = targets.iter().collect();
        ordered.sort();
        for name in ordered {
            for len in 2..=5 {
                if name.len() < len + NICK_TAIL {
                    continue;
                }
                let p = &name[..len];
                if exact_roles.contains(p) || common_speaker_token(p) || inflectional_tail(p, name) {
                    continue;
                }
                hits.entry(p.to_string()).or_default().push(name.clone());
            }
        }
        let mut prefix_of: HashMap<String, String> = HashMap::new();
        for (p, mut names) in hits {
            names.sort();
            names.dedup();
            if names.len() == 1 {
                prefix_of.insert(p, names.pop().unwrap());
            }
        }
        let mut witnessed: HashMap<String, HashSet<String>> = HashMap::new();
        for text in &texts {
            for tok in surface_tokens(text) {
                if !tok.title_case {
                    continue;
                }
                let Some(full) = prefix_of.get(&tok.lower) else { continue };
                if tok.sentence_initial && lower_attested.contains(&tok.lower) {
                    continue;
                }
                witnessed.entry(full.clone()).or_default().insert(tok.lower);
            }
        }
        Self { prefix_of, witnessed, lower_attested }
    }

    fn bind_token(&self, tok: &SurfaceTok, sentence_initial: bool) -> Option<&str> {
        if !tok.title_case || common_speaker_token(&tok.lower) {
            return None;
        }
        let full = self.prefix_of.get(&tok.lower)?;
        if sentence_initial && self.lower_attested.contains(&tok.lower) {
            return None;
        }
        let seen = self.witnessed.get(full);
        let witnessed_this = seen.is_some_and(|s| s.contains(&tok.lower));
        if !(3..=4).contains(&tok.lower.len()) && !witnessed_this {
            return None;
        }
        if seen.is_some_and(|s| !s.is_empty() && !s.contains(&tok.lower)) {
            return None;
        }
        Some(full.as_str())
    }

    fn bindings_in(&self, text: &str) -> HashMap<String, String> {
        let mut out = HashMap::new();
        for tok in surface_tokens(text) {
            if let Some(full) = self.bind_token(&tok, tok.sentence_initial) {
                out.insert(tok.lower, full.to_string());
            }
        }
        out
    }
}

fn build_index(sessions: &[Session], p: &Params) -> Index {
    const BACKCHANNEL_LEADS: &[&str] = &[
        "wow", "oh", "aw", "aww", "haha", "hahaha", "lol", "nice", "cool", "awesome",
        "congrats", "congratulations", "yay", "sweet", "amazing", "wonderful", "fantastic",
        "thanks", "thank", "sorry", "glad", "ouch", "yikes",
    ];
    const BLIP_STOPLIST: &[&str] = &[
        "photo", "picture", "image", "man", "woman", "person", "people", "group", "standing",
        "sitting", "holding", "wearing", "smiling", "table", "room", "background", "white",
        "black", "young", "two", "close", "up", "front", "next", "looking", "camera",
    ];
    const USER_SCHEMA_CUES: &[&str] = &[
        "i always ", "i usually ", "i tend to ", "my go-to", "my usual",
        "i switched to ", "i stick to ", "i stick with ", "fan of ",
        "i'm into ", "im into ", "i love ", "i prefer ", "i enjoy ", "my favorite", "my favourite",
    ];
    const STATE_REVISION_CUES: &[&str] = &[
        "actually", "switched to", "moved to", "relocated to", "changed to", "now use",
        "now live", "no longer", "instead of", "used to ", "anymore", "upgraded to", "replaced",
    ];

    let nicks = SpeakerNickIndex::build(sessions);
    let mut vocab = Vocab::default();
    let mut turns: Vec<TurnIx> = Vec::new();
    let mut session_ids = Vec::new();
    let mut session_ts = Vec::new();
    let mut session_ep_keys = Vec::with_capacity(sessions.len());
    let mut session_turns = Vec::new();
    let mut session_entities = Vec::with_capacity(sessions.len());
    let mut session_content = Vec::with_capacity(sessions.len());
    let mut domain_session_df: HashMap<&'static str, u32> = HashMap::new();
    let mut raw_standing_rules: Vec<(String, usize, usize, i64, bool, Vec<String>)> = Vec::new();
    let mut turn_tech_handles: Vec<Vec<u32>> = Vec::new();
    let mut turn_prose_cache: Vec<String> = Vec::new();
    let mut cue_vocab: HashMap<String, u32> = HashMap::new();
    let mut turn_cues: Vec<Vec<u32>> = Vec::new();
    let mut role_cues: HashSet<u32> = HashSet::new();
    let pref_set: HashSet<&str> = PREF_MARKERS.iter().copied().collect();
    let hard_constraint_set: HashSet<&str> = HARD_CONSTRAINT_MARKERS.iter().copied().collect();
    let mut constraint_terms: HashSet<u32> = HashSet::new();
    let mut header_dict: HashMap<u64, Vec<HeaderKey>> = HashMap::new();
    let mut header_max_words = 0usize;
    for (si, s) in sessions.iter().enumerate() {
        let Some(first) = s.turns.first() else { continue };
        let Some((full_h, core_h)) = extract_header_titles(&first.content) else { continue };
        for (span_str, mult) in [(full_h, 1.0f32), (core_h, 0.88f32)] {
            let htoks = norm_span_tokens(span_str);
            if htoks.is_empty() || htoks.len() > 10 {
                continue;
            }
            if htoks.len() == 1 && (htoks[0].len() < 3 || stopwords().contains(htoks[0].as_str())) {
                continue;
            }
            header_max_words = header_max_words.max(htoks.len());
            let hids: Vec<u32> = htoks.iter().map(|w| vocab.intern(w)).collect();
            let hh = hash_u32_slice(&hids);
            let bucket = header_dict.entry(hh).or_default();
            if let Some(existing) = bucket.iter_mut().find(|k| k.tokens == hids) {
                if let Some(pair) = existing.sessions.iter_mut().find(|(id, _)| *id == si) {
                    pair.1 = pair.1.max(mult);
                } else {
                    existing.sessions.push((si, mult));
                }
            } else {
                bucket.push(HeaderKey { tokens: hids, sessions: vec![(si, mult)] });
            }
        }
    }

    // Pre-pass: collect exact single-token proper names evidenced by speakers, mid-clause
    // capitalization, multi-word entity parts, or document header titles so sentence-initial
    // occurrences of the same byte-equal name bind to the same identity node.
    let mut known_proper: HashSet<String> = HashSet::new();
    for s in sessions {
        for t in &s.turns {
            let sp = t.role.trim().to_lowercase();
            if !sp.is_empty() && sp != "user" && sp != "assistant" {
                for w in raw_tokens(&sp) {
                    if w.len() >= 3 && !STARTER_STOP.contains(&w.as_str()) && !stopwords().contains(w.as_str()) {
                        known_proper.insert(w);
                    }
                }
            }
            for ent in proper_entity_tokens(&t.content) {
                if let Some(body) = ent.strip_prefix("\u{1}ent:") {
                    if !body.contains('_') && body.len() >= 3 {
                        known_proper.insert(body.to_string());
                    }
                }
            }
        }
    }

    for (si, s) in sessions.iter().enumerate() {
        let ts = if s.date.is_empty() { s.timestamp } else { match parse_date(&s.date) { 0 => s.timestamp, t => t } };
        session_ids.push(s.id.clone());
        session_ts.push(ts);
        session_ep_keys.push(session_episode_key(&s.id, &s.date));
        let date_toks = if p.enable_deictic && ts != 0 { calendar_tokens(ts) } else { Vec::new() };
        let mut members = Vec::new();
        let mut s_ents = HashSet::new();
        let mut s_cont = HashSet::new();
        let mut s_doms: HashSet<&'static str> = HashSet::new();
        let mut speaker_names: HashSet<String> = HashSet::new();
        for t in &s.turns {
            let sp = t.role.trim().to_lowercase();
            if !sp.is_empty() && sp != "user" && sp != "assistant" {
                for w in raw_tokens(&sp) {
                    speaker_names.insert(w);
                }
            }
        }
        // Collect named speakers in this session so vocative address ("Thanks, Caroline!")
        // is capped at speaker identity parity (0.72) and never outranks Caroline speaking.
        let mut sess_other_speakers: HashMap<usize, HashSet<String>> = HashMap::new();
        for (idx_a, ta) in s.turns.iter().enumerate() {
            let spa = ta.role.trim().to_lowercase();
            let mut others = HashSet::new();
            for (idx_b, tb) in s.turns.iter().enumerate() {
                if idx_a == idx_b {
                    continue;
                }
                let spb = tb.role.trim().to_lowercase();
                if !spb.is_empty() && spb != "user" && spb != "assistant" && spb != spa {
                    for sw in raw_tokens(&spb) {
                        if let Some(set) = nicks.witnessed.get(&sw) {
                            for nick in set {
                                others.insert(nick.clone());
                            }
                        }
                        others.insert(sw);
                    }
                }
            }
            sess_other_speakers.insert(idx_a, others);
        }

        for (ti, t) in s.turns.iter().enumerate() {
            let (prose_body, blip_opt) = split_blip_caption(&t.content);
            let raw = raw_tokens(prose_body);
            let nick_bound = nicks.bindings_in(prose_body);
            let role = t.role.trim();
            let speaker = role.to_lowercase();
            let vocative_partners = sess_other_speakers.get(&ti);
            let field = usize::from(p.enable_roles && role.eq_ignore_ascii_case("assistant"));
            let mut feats = Vec::new();
            let mut blip_feats = Vec::new();
            let mut seq = Vec::new();
            let mut content = HashSet::new();
            let mut entities = HashSet::new();
            let mut t_doms: Vec<&'static str> = Vec::new();
            let mut tech_handles: Vec<u32> = Vec::new();
            for w in raw.iter().filter(|w| is_content(w)) {
                let is_vocative_partner = vocative_partners.is_some_and(|vp| vp.contains(w.as_str()));
                let e = vocab.intern(w);
                let tok_w = if is_vocative_partner { 0.72 } else { 1.0 };
                feats.push(Feat { id: e, w: tok_w, counts_len: true, inheritable: !is_vocative_partner });
                content.insert(e);
                s_cont.insert(e);
                if let Some(full) = nick_bound.get(w.as_str()) {
                    if !raw.iter().any(|t_tok| t_tok == full) {
                        let voc = vocative_partners.is_some_and(|vp| vp.contains(full.as_str()) || vp.contains(w.as_str()));
                        let fid = vocab.intern(full);
                        feats.push(Feat {
                            id: fid,
                            w: if voc { 0.72 } else { 1.0 },
                            counts_len: false,
                            inheritable: false,
                        });
                    }
                }
                let st = stem(w);
                let sid = if p.enable_stemming { vocab.intern(&stem_term(&st)) } else { e };
                if p.enable_stemming && !is_vocative_partner {
                    feats.push(Feat { id: sid, w: 1.0, counts_len: false, inheritable: true });
                }
                seq.push((e, sid));
                if let Some(doms) = hypernyms().get(&st) {
                    for &d in doms {
                        s_doms.insert(d);
                        if !t_doms.contains(&d) {
                            t_doms.push(d);
                        }
                        if p.enable_hypernyms {
                            feats.push(Feat { id: vocab.intern(&hyp_term(d)), w: 1.0, counts_len: false, inheritable: true });
                        }
                    }
                }
                if known_proper.contains(w) && !is_vocative_partner {
                    let eid = vocab.intern(&format!("\u{1}ent:{w}"));
                    entities.insert(eid);
                    s_ents.insert(eid);
                }
            }
            // Stream 2: Auxiliary BLIP caption field (w=0.75, b_blip=0, counts_len:false, inheritable:false)
            if let Some(cap) = blip_opt {
                for bw in raw_tokens(cap).into_iter().filter(|w| is_content(w) && !BLIP_STOPLIST.contains(&w.as_str())) {
                    let bid = vocab.intern(&bw);
                    blip_feats.push(Feat { id: bid, w: 1.0, counts_len: false, inheritable: false });
                    if p.enable_stemming {
                        let bst = stem(&bw);
                        let bsid = vocab.intern(&stem_term(&bst));
                        blip_feats.push(Feat { id: bsid, w: 0.75, counts_len: false, inheritable: false });
                    }
                }
                for bent in proper_entity_tokens(cap) {
                    let eid = vocab.intern(&bent);
                    blip_feats.push(Feat { id: eid, w: 0.85, counts_len: false, inheritable: false });
                    entities.insert(eid);
                    s_ents.insert(eid);
                }
            }
            // Pre-saturation BM25F title boost for structured `{Title}\n{Body}` documents.
            if let Some(core_title) = extract_header_title(prose_body) {
                for hw in raw_tokens(core_title).into_iter().filter(|w| is_content(w)) {
                    let hid = vocab.intern(&hw);
                    feats.push(Feat { id: hid, w: 1.5, counts_len: true, inheritable: false });
                    if p.enable_stemming {
                        let hst = stem(&hw);
                        let hsid = vocab.intern(&stem_term(&hst));
                        feats.push(Feat { id: hsid, w: 1.0, counts_len: false, inheritable: false });
                    }
                }
            }
            for ent in proper_entity_tokens(prose_body) {
                if let Some(body) = ent.strip_prefix("\u{1}ent:") {
                    if !body.contains('_') && vocative_partners.is_some_and(|vp| vp.contains(body)) {
                        continue;
                    }
                }
                let eid = vocab.intern(&ent);
                feats.push(Feat { id: eid, w: 1.0, counts_len: false, inheritable: true });
                entities.insert(eid);
                s_ents.insert(eid);
            }
            for raw_word in prose_body.split_whitespace() {
                let word = raw_word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-');
                if word.is_empty() {
                    continue;
                }
                for (part, aux_w) in surface_aux_tokens(word) {
                    if raw.iter().any(|t_tok| t_tok == &part) {
                        continue;
                    }
                    let id = vocab.intern(&part);
                    feats.push(Feat { id, w: aux_w, counts_len: false, inheritable: false });
                }
            }
            let lower = prose_body.to_lowercase();
            let pref_marker = user_preference_speech_act(&lower, &raw, &pref_set);
            let is_personal_constraint = field == 0
                && (raw.iter().any(|w| hard_constraint_set.contains(w.as_str()))
                    || HARD_CONSTRAINT_MARKERS.iter().any(|m| m.contains(' ') && lower.contains(m)));
            let is_user_schema = field == 0
                && (pref_marker
                    || habitual_perfect(&lower)
                    || USER_SCHEMA_CUES.iter().any(|c| lower.contains(c)));
            if p.enable_preference && pref_marker && field == 0 {
                feats.push(Feat { id: vocab.intern(PREF_AXIS), w: 1.0, counts_len: false, inheritable: false });
            }
            if p.enable_deictic {
                for (d, w) in deictic_weighted_tokens(&raw, ts) {
                    feats.push(Feat { id: vocab.intern(&d), w, counts_len: false, inheritable: false });
                }
                for d in inline_date_tokens(prose_body) {
                    feats.push(Feat { id: vocab.intern(&d), w: 0.9, counts_len: false, inheritable: false });
                }
                for d in &date_toks {
                    feats.push(Feat { id: vocab.intern(d), w: p.date_weight, counts_len: false, inheritable: false });
                }
            }
            if !speaker.is_empty() && speaker != "user" && speaker != "assistant" {
                for w in raw_tokens(&speaker) {
                    let id = vocab.intern(&w);
                    // Speaker identity edge: own speech carries full parity (1.05) over vocative address.
                    feats.push(Feat { id, w: 1.05, counts_len: false, inheritable: false });
                    let eid = vocab.intern(&format!("\u{1}ent:{w}"));
                    entities.insert(eid);
                    s_ents.insert(eid);
                }
            }
            for obj in constraint_object_tokens(&lower) {
                let id = vocab.intern(&obj);
                feats.push(Feat { id, w: 1.8, counts_len: false, inheritable: true });
                content.insert(id);
                s_cont.insert(id);
                constraint_terms.insert(id);
            }
            for (slot_key, slot_val) in typed_slot_tokens(prose_body) {
                let slot_ent = vocab.intern(&format!("\u{1}slot:{slot_key}"));
                let kid = vocab.intern(&slot_key);
                let vid = vocab.intern(&slot_val);
                feats.push(Feat { id: kid, w: 0.5, counts_len: false, inheritable: false });
                feats.push(Feat { id: vid, w: 1.6, counts_len: false, inheritable: true });
                feats.push(Feat { id: slot_ent, w: 0.8, counts_len: false, inheritable: false });
                content.insert(vid);
                s_cont.insert(vid);
                entities.insert(slot_ent);
                s_ents.insert(slot_ent);
                tech_handles.push(slot_ent);
            }
            for h_str in extract_exact_technical_handles(prose_body) {
                let hid = vocab.intern(&h_str);
                feats.push(Feat { id: hid, w: 1.25, counts_len: false, inheritable: true });
                entities.insert(hid);
                s_ents.insert(hid);
                tech_handles.push(hid);
            }
            let (is_question, is_short_question) = detect_question(prose_body, raw.len());
            let qshape = analyze_question(&mut vocab, prose_body, is_question, p.enable_stemming, &speaker_names);
            let is_pure_info_question = qshape.is_pure;
            let anaphoric = raw.first().is_some_and(|w| CONTINUATIONS.contains(&w.as_str()))
                || raw.iter().take(12).any(|w| ANAPHORS.contains(&w.as_str()));
            let has_temporal = raw.iter().any(|w| {
                token_is_temporal_word(w.as_str())
                    || ["am", "pm"].contains(&w.as_str())
                    || (w.len() >= 3 && (w.ends_with("st") || w.ends_with("nd") || w.ends_with("rd") || w.ends_with("th")) && w[..w.len()-2].chars().all(|c| c.is_ascii_digit()))
            });
            let has_numeric = prose_body.chars().any(|c| c.is_ascii_digit())
                || raw.iter().any(|w| NUMBER_WORDS.contains(&w.as_str()));
            let has_spatial = raw.iter().any(|w| SPATIAL_WORDS.contains(&w.as_str()));
            let padded_lower = word_padded(&lower);
            let has_causal = has_any(&padded_lower, CAUSAL_WORDS);
            let is_standing_policy = field == 0 && utterance_is_standing_policy(prose_body);
            let is_backchannel = !is_question
                && content.len() <= 4
                && !has_temporal
                && !has_numeric
                && entities.is_empty()
                && !pref_marker
                && !is_standing_policy
                && raw.first().is_some_and(|w| BACKCHANNEL_LEADS.contains(&w.as_str()));
            let is_initiative = field == 0
                && !is_standing_policy
                && !anaphoric
                && !is_short_question
                && content.len() >= 4;
            let turn_idx = turns.len();
            if is_standing_policy {
                let (has_explicit_scope, scope_words) = extract_rule_scope(prose_body);
                raw_standing_rules.push((
                    standing_policy_id(prose_body),
                    turn_idx,
                    si,
                    ts,
                    has_explicit_scope,
                    scope_words,
                ));
            }
            let id = if t.id.is_empty() { format!("{}#{}", s.id, ti) } else { t.id.clone() };
            let body_for_headers = if header_max_words > 0 {
                let body = if extract_header_titles(prose_body).is_some() {
                    prose_body.split_once('\n').map(|(_, rest)| rest).unwrap_or(prose_body)
                } else {
                    prose_body
                };
                body.to_string()
            } else {
                String::new()
            };
            turn_tech_handles.push(tech_handles);
            turn_prose_cache.push(prose_body.to_string());
            turn_cues.push(intern_cues(&t.content, &mut cue_vocab));
            role_cues.extend(intern_cues(&t.role, &mut cue_vocab));
            members.push(turn_idx);
            turns.push(TurnIx {
                session: si,
                id,
                field,
                feats,
                blip_feats,
                seq,
                content,
                entities,
                body_for_headers,
                span_ids_cache: OnceLock::new(),
                speaker,
                is_question,
                is_pure_info_question,
                pre_question_ids: qshape.pre_ids,
                question_clause_ids: qshape.clause_ids,
                answers_question: false,
                is_short_reply: false,
                is_backchannel,
                anaphoric,
                pref_marker,
                is_personal_constraint,
                is_user_schema,
                schema_domains: t_doms,
                has_temporal,
                has_numeric,
                has_spatial,
                has_causal,
                is_short_question,
                is_standing_policy,
                is_initiative,
                answers: None,
                exact_edges: Vec::new(),
                policy_backlinks: Vec::new(),
                supersedes_turns: Vec::new(),
                stability: t.stability,
                form_clock: has_iso_or_hhmm(prose_body),
                form_digit: has_ascii_digit(prose_body),
                form_handle: has_handle_or_path(prose_body),
                form_midcap: has_mid_sentence_name(prose_body),
                raw_links: t.links.clone(),
            });
        }
        for d in s_doms {
            *domain_session_df.entry(d).or_insert(0) += 1;
        }
        session_turns.push(members);
        session_entities.push(s_ents);
        session_content.push(s_cont);
    }
    let _ = STATE_REVISION_CUES;

    // Mark DAMSL adjacency pairs before bag construction: question/answer pairs AND
    // short value/commitment replies to preceding statements ("Done.", "Last Tuesday.", "Me too!").
    for members in &session_turns {
        for pos in 1..members.len() {
            let (a, b) = (members[pos - 1], members[pos]);
            if turns[a].speaker != turns[b].speaker {
                turns[b].answers = Some(a);
                if turns[a].is_question {
                    turns[b].answers_question = true;
                }
                if !turns[b].is_question
                    && !turns[b].is_backchannel
                    && turns[b].content.len() <= 5
                    && (turns[a].is_question
                        || turns[b].anaphoric
                        || turns[b].content.len() <= 2
                        || turns[b].has_temporal
                        || turns[b].has_numeric)
                {
                    turns[b].is_short_reply = true;
                }
            }
        }
    }

    let mut id_at: HashMap<String, usize> = HashMap::new();
    for (i, t) in turns.iter().enumerate() {
        if !t.id.is_empty() {
            id_at.entry(t.id.clone()).or_insert(i);
        }
    }
    for ti in 0..turns.len() {
        let pending = std::mem::take(&mut turns[ti].raw_links);
        for link in pending {
            let Some(&target) = id_at.get(&link.to) else { continue };
            if target == ti {
                continue;
            }
            turns[ti].exact_edges.push((link.degree.max(2), target));
        }
    }

    // ---- Ingest-Time Retroactive Salience Backfill (RSB), Contradiction Supersession,
    //      Synaptic Tagging 9h CaptureWindow, and Temporal Context Episode Contiguity ----
    // Strictly BACKWARD-ONLY (`later -> earlier`). Zero forward edges, zero word similarity.
    let ns_count = session_turns.len();
    let mut session_rsb_back: Vec<Vec<(u32, usize)>> = vec![Vec::new(); ns_count];
    let mut session_supersedes_back: Vec<Vec<usize>> = vec![Vec::new(); ns_count];
    let mut session_episode_prev: Vec<Option<usize>> = vec![None; ns_count];
    for si in 1..ns_count {
        let same_ep = !session_ep_keys[si].is_empty() && session_ep_keys[si] == session_ep_keys[si - 1];
        let same_nonzero_ts = session_ts[si] > 0 && session_ts[si] == session_ts[si - 1];
        if same_ep || same_nonzero_ts {
            session_episode_prev[si] = Some(si - 1);
        }
    }

    // 1. Exact Technical Handle & Typed Slot RSB (`df <= 6`): later turn/session -> earlier turn/session.
    let mut handle_postings: HashMap<u32, Vec<usize>> = HashMap::new();
    for (ti, hs) in turn_tech_handles.iter().enumerate() {
        for &h in hs {
            let list = handle_postings.entry(h).or_default();
            if list.last().copied() != Some(ti) {
                list.push(ti);
            }
        }
    }
    for (_, list) in handle_postings {
        let deg = list.len() as u32;
        if !(2..=6).contains(&deg) {
            continue;
        }
        for idx in 1..list.len() {
            let later = list[idx];
            let earlier = list[idx - 1];
            if !turns[later].exact_edges.iter().any(|&(_, t)| t == earlier) {
                turns[later].exact_edges.push((deg, earlier));
            }
            let s_lat = turns[later].session;
            let s_ear = turns[earlier].session;
            if s_lat > s_ear && !session_rsb_back[s_lat].iter().any(|&(_, s)| s == s_ear) {
                session_rsb_back[s_lat].push((deg, s_ear));
            }
        }
    }

    // 2. State-Update & Structural Contradiction Supersession Chain (`appears_contradictory`):
    //    When two user turns share a rare proper entity (`2 <= df <= 8`) or constraint object
    //    and either `advanced::contradiction::appears_contradictory` fires or the later turn
    //    asserts an explicit revision/constraint, record a backward `supersedes` edge (`later -> earlier`).
    let mut ent_user_postings: HashMap<u32, Vec<usize>> = HashMap::new();
    for (ti, t) in turns.iter().enumerate() {
        if t.field != 0 || t.is_question || t.is_backchannel {
            continue;
        }
        for &eid in &t.entities {
            ent_user_postings.entry(eid).or_default().push(ti);
        }
        for &cid in &t.content {
            if constraint_terms.contains(&cid) {
                ent_user_postings.entry(cid).or_default().push(ti);
            }
        }
    }
    for (_, mut list) in ent_user_postings {
        list.dedup();
        if !(2..=8).contains(&list.len()) {
            continue;
        }
        for idx in 1..list.len() {
            let later = list[idx];
            let prev = list[idx - 1];
            let s_lat = turns[later].session;
            let s_prev = turns[prev].session;
            if s_lat <= s_prev {
                continue;
            }
            let lat_prose = &turn_prose_cache[later];
            let prev_prose = &turn_prose_cache[prev];
            let lat_low = lat_prose.to_lowercase();
            let has_rev_cue = turns[later].is_personal_constraint
                || STATE_REVISION_CUES.iter().any(|c| lat_low.contains(c));
            let contradicts = crate::advanced::contradiction::appears_contradictory(
                prev_prose,
                lat_prose,
                crate::advanced::contradiction::SubjectIdentity::AlreadyEstablished,
            );
            if contradicts || has_rev_cue {
                if !turns[later].supersedes_turns.contains(&prev) {
                    turns[later].supersedes_turns.push(prev);
                }
                if !session_supersedes_back[s_lat].contains(&s_prev) {
                    session_supersedes_back[s_lat].push(s_prev);
                }
            }
        }
    }

    // 3. Synaptic Tagging `CaptureWindow` (Frey & Morris 1997; Redondo & Morris 2011; 9-hour window):
    //    A high-salience turn (`is_standing_policy`, `is_personal_constraint`, or `FAILURE_MARKERS`)
    //    captures the immediately preceding tagged user initiative turn within `CaptureWindow::backward_hours` (9h)
    //    or the immediate preceding same-episode sub-session (`session_episode_prev`).
    let capture_window = crate::neuroscience::CaptureWindow::default();
    let max_back_secs = (capture_window.backward_hours * 3600.0) as i64;
    for ti in 1..turns.len() {
        let lower_p = turn_prose_cache[ti].to_lowercase();
        let is_failure_prp = turns[ti].field == 0
            && !turns[ti].is_question
            && crate::advanced::retroactive_backfill::FAILURE_MARKERS
                .iter()
                .any(|m| lower_p.contains(m));
        let is_prp = turns[ti].is_standing_policy || turns[ti].is_personal_constraint || is_failure_prp;
        if !is_prp {
            continue;
        }
        let s_cur = turns[ti].session;
        let ts_cur = session_ts[s_cur];
        // Search backward for the closest preceding user initiative turn inside the 9h capture window (or same episode).
        for back_s in (0..s_cur).rev() {
            let ts_prev = session_ts[back_s];
            let in_clock_window = ts_cur > 0 && ts_prev > 0 && ts_cur >= ts_prev && (ts_cur - ts_prev) <= max_back_secs;
            let is_ep_prev = session_episode_prev[s_cur] == Some(back_s);
            if !in_clock_window && !is_ep_prev {
                if ts_cur > 0 && ts_prev > 0 && (ts_cur - ts_prev) > max_back_secs {
                    break;
                }
                continue;
            }
            if let Some(&init_ti) = session_turns[back_s].iter().rev().find(|&&cand| turns[cand].is_initiative) {
                // Require at least 1 shared entity or schema domain so unrelated adjacent topics don't capture blindly.
                let shares_anchor = turns[ti].entities.iter().any(|e| turns[init_ti].entities.contains(e))
                    || turns[ti].schema_domains.iter().any(|d| turns[init_ti].schema_domains.contains(d));
                if shares_anchor {
                    if !turns[ti].exact_edges.iter().any(|&(_, t)| t == init_ti) {
                        turns[ti].exact_edges.push((3, init_ti));
                    }
                    if !session_rsb_back[s_cur].iter().any(|&(_, s)| s == back_s) {
                        session_rsb_back[s_cur].push((3, back_s));
                    }
                }
                break;
            }
        }
    }

    // 4. SalienceJoin. A later turn records a backward link to an earlier
    //    standing policy in the same session, or in an earlier session that shares
    //    the episode key. The link is session or episode identity, not a word match.
    //    It is not a general exact edge and it does not write session_rsb_back.
    //    v1's global winner seat raised a rule from a different batch. v3 does not
    //    require the policy to match the query. Rank time seats it behind the
    //    retrieved turn that points at it.
    for si in 0..ns_count {
        let members = &session_turns[si];
        for (pos, &p_ti) in members.iter().enumerate() {
            if !turns[p_ti].is_standing_policy {
                continue;
            }
            for &t_ti in &members[pos + 1..] {
                if !turns[t_ti].policy_backlinks.contains(&p_ti) {
                    turns[t_ti].policy_backlinks.push(p_ti);
                }
            }
        }
    }
    for si in 0..ns_count {
        if session_ep_keys[si].is_empty() {
            continue;
        }
        for sj in 0..si {
            if session_ep_keys[sj] != session_ep_keys[si] {
                continue;
            }
            for &p_ti in &session_turns[sj] {
                if !turns[p_ti].is_standing_policy {
                    continue;
                }
                for &t_ti in &session_turns[si] {
                    if !turns[t_ti].policy_backlinks.contains(&p_ti) {
                        turns[t_ti].policy_backlinks.push(p_ti);
                    }
                }
            }
        }
    }

    // Session bags (own text + field-2 BLIP), uncarried turn bags (`turn_own_bags`),
    // and carried turn bags (`turn_bags` — never carrying into questions!).
    let mut session_bags = Vec::with_capacity(session_turns.len());
    let mut turn_own_bags: Vec<Bag> = (0..turns.len()).map(|_| Bag::default()).collect();
    let mut turn_bags: Vec<Bag> = (0..turns.len()).map(|_| Bag::default()).collect();
    for members in &session_turns {
        let mut sb = Bag::default();
        for (pos, &ti) in members.iter().enumerate() {
            let t = &turns[ti];
            for &f in &t.feats {
                sb.add(t.field, f, 1.0);
                turn_own_bags[ti].add(t.field, f, 1.0);
                turn_bags[ti].add(t.field, f, 1.0);
            }
            for &bf in &t.blip_feats {
                sb.add(2, bf, 1.0);
                turn_own_bags[ti].add(2, bf, 1.0);
                turn_bags[ti].add(2, bf, 1.0);
            }
            // In named-speaker dialogues (`LoCoMo`), a follow-up question does NOT state the
            // previous speaker's facts; in `user`/`assistant` dialogues, user follow-up questions
            // still inherit conversational context (`pos - k`).
            let is_named_speaker_q = t.is_question
                && !t.speaker.is_empty()
                && t.speaker != "user"
                && t.speaker != "assistant";
            if p.enable_carry && pos > 0 && !is_named_speaker_q {
                let answer = t.answers_question;
                let factor = if t.is_backchannel {
                    0.15
                } else if answer || t.anaphoric {
                    0.85
                } else {
                    p.carry_plain_factor
                };
                let field = t.field;
                let mut inherited: Vec<(Feat, f32)> = Vec::new();
                for k in 1..=3usize {
                    if pos < k {
                        break;
                    }
                    let src = &turns[members[pos - k]];
                    let w = if k == 1 && answer {
                        p.carry_answer
                    } else if k == 1 && t.is_short_reply {
                        0.75
                    } else {
                        p.carry[k - 1] * factor
                    };
                    if w <= 0.0 {
                        continue;
                    }
                    // Inherit open-class content (exact + stem + hypernym + proper entities), never speaker tag or dates.
                    for &f in src.feats.iter().filter(|f| f.inheritable) {
                        inherited.push((f, w));
                    }
                }
                for (f, w) in inherited {
                    turn_bags[ti].add_inherited(field, f, w);
                }
            }
            // Third-position uptake: a concise value-bearing turn inherits light context from pos + 1.
            if p.enable_carry && !t.is_backchannel && !t.is_question && t.content.len() <= 4 && pos + 1 < members.len() {
                let nxt = &turns[members[pos + 1]];
                for &f in nxt.feats.iter().filter(|f| f.inheritable) {
                    turn_bags[ti].add_inherited(t.field, f, 0.25);
                }
            }
        }
        session_bags.push(sb);
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
                for &bf in &turns[ti].blip_feats {
                    b.add(2, bf, 1.0);
                }
            }
            windows.push((si, start, end));
            window_bags.push(b);
        }
    }

    let s_stats = stats_of(&session_bags);
    let t_own_stats = stats_of(&turn_own_bags);
    let t_stats = stats_of(&turn_bags);
    let w_stats = stats_of(&window_bags);

    // Precompute rarest non-hub tokens per session (`df >= 2 && df <= 0.20 * N`, up to 32 sorted)
    // for GhostLink Ochiai divergence (`1 - |Nt(a) & Nt(b)| / sqrt(|Nt(a)| * |Nt(b)|)`).
    let ns_total = session_bags.len();
    let max_rare_nt_df = ((s_stats.n * 0.20).max(4.0)) as u32;
    let session_rare_nt: Vec<Vec<u32>> = session_content
        .iter()
        .map(|sc| {
            let mut v: Vec<(u32, u32)> = sc
                .iter()
                .filter_map(|&id| {
                    let df = s_stats.df.get(&id).copied().unwrap_or(0);
                    (df >= 2 && df <= max_rare_nt_df).then_some((df, id))
                })
                .collect();
            v.sort_unstable();
            let mut ids: Vec<u32> = v.into_iter().take(32).map(|(_, id)| id).collect();
            ids.sort_unstable();
            ids
        })
        .collect();

    let dom_hub_cut = if ns_total >= 16 {
        ((ns_total as f32 * 0.12).ceil() as u32).max(3)
    } else {
        u32::MAX
    };
    let non_hub_domains: HashSet<&'static str> = HYPERNYM_LATTICE
        .iter()
        .map(|(d, _)| *d)
        .filter(|d| domain_session_df.get(d).copied().unwrap_or(0) <= dom_hub_cut)
        .collect();

    let fan_cut = if ns_total >= 16 {
        ((s_stats.n * 0.12).max(4.0)) as u32
    } else {
        u32::MAX
    };
    let fw_set = framing_words();
    let mut standing_rules = Vec::with_capacity(raw_standing_rules.len());
    for (policy_key, turn_idx, sess_idx, ts, has_explicit_scope, scope_words) in raw_standing_rules {
        let mut scope = HashSet::new();
        let mut dom_vec: Vec<&'static str> = Vec::new();
        for w in &scope_words {
            if fw_set.contains(w.as_str()) {
                continue;
            }
            if let Some(id) = vocab.get(w) {
                if s_stats.df.get(&id).copied().unwrap_or(0) <= fan_cut {
                    scope.insert(id);
                }
            }
            let st = stem(w);
            if let Some(sid) = vocab.get(&stem_term(&st)) {
                if s_stats.df.get(&sid).copied().unwrap_or(0) <= fan_cut {
                    scope.insert(sid);
                }
            }
            if let Some(doms) = hypernyms().get(&st) {
                for &d in doms {
                    if non_hub_domains.contains(d) && !dom_vec.contains(&d) {
                        dom_vec.push(d);
                    }
                }
            }
        }
        let mut assistant_nouns = HashSet::new();
        let members = &session_turns[sess_idx];
        if let Some(pos) = members.iter().position(|&ti| ti == turn_idx) {
            if pos + 1 < members.len() {
                let adj = &turns[members[pos + 1]];
                if adj.field == 1 {
                    for &cid in &adj.content {
                        if s_stats.df.get(&cid).copied().unwrap_or(0) <= fan_cut {
                            assistant_nouns.insert(cid);
                        }
                    }
                }
            }
        }
        standing_rules.push(StandingRule {
            policy_key,
            turn: turn_idx,
            session: sess_idx,
            ts,
            has_explicit_scope,
            scope,
            domains: dom_vec,
            assistant_nouns,
        });
    }

    let bridge_tokens = steiner_bridge::collect_bridge_tokens(&vocab, &turns, &nicks);

    Index {
        vocab,
        session_ids,
        session_ts,
        session_turns,
        session_entities,
        session_content,
        session_rare_nt,
        session_rsb_back,
        session_supersedes_back,
        session_episode_prev,
        turns,
        session_bags,
        turn_own_bags,
        turn_bags,
        windows,
        window_bags,
        s_stats,
        t_own_stats,
        t_stats,
        w_stats,
        constraint_terms,
        standing_rules,
        non_hub_domains,
        header_dict,
        header_max_words,
        nicks,
        turn_cues,
        cue_vocab,
        role_cues,
        bridge_tokens,
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
    exact_hit: bool,
}

struct QueryPlan {
    intent: QueryIntent,
    terms: Vec<QTerm>,
    groups: Vec<String>,
    /// term id -> group, for proximity scanning
    group_of: HashMap<u32, usize>,
    focal_ids: HashSet<u32>,
    focal_domains: HashSet<&'static str>,
}

/// Normalize punctuation (preserving internal apostrophes in contractions like `i'm` / `don't`)
/// into single-space-padded lowercase tokens so word-boundary cue matching never misfires on
/// substrings ("Army" != "my", "server" != "serve", "Miami's" != "i'", "claim" != "aim")
/// and never strips literal "$0".
fn word_padded(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 4);
    out.push(' ');
    let chars: Vec<char> = text.chars().collect();
    let mut prev_space = true;
    for (idx, &ch) in chars.iter().enumerate() {
        let lower = ch.to_ascii_lowercase();
        let is_apos = (lower == '\'' || lower == '’')
            && idx > 0
            && idx + 1 < chars.len()
            && chars[idx - 1].is_alphanumeric()
            && chars[idx + 1].is_alphanumeric();
        // Glue a currency marker to the amount so "$0" stays the literal token "$0".
        let is_currency = ch == '$' && idx + 1 < chars.len() && chars[idx + 1].is_ascii_digit();
        if lower.is_alphanumeric() || is_currency {
            out.push(if is_currency { '$' } else { lower });
            prev_space = false;
        } else if is_apos {
            out.push('\'');
            prev_space = false;
        } else if !prev_space {
            out.push(' ');
            prev_space = true;
        }
    }
    if !prev_space {
        out.push(' ');
    }
    out
}

fn has_any(padded_hay: &str, cues: &[&str]) -> bool {
    cues.iter().any(|c| {
        if c.starts_with(' ') && c.ends_with(' ') {
            padded_hay.contains(c)
        } else if c.starts_with(' ') {
            // Leading word boundary; allow suffix inflection on phrases like " recommend" / " suggest"
            padded_hay.contains(c)
        } else if c.ends_with(' ') {
            let pat = format!(" {c}");
            padded_hay.contains(&pat)
        } else if c.contains(' ') {
            // Multi-word phrase: require leading word boundary so "did i" matches " did i " or " did i'm"
            let pat = format!(" {c}");
            padded_hay.contains(&pat)
        } else {
            // Single un-spaced stem cue ("recommend", "suggest", "prefer", "chronological"):
            // require leading word boundary (` {c}`) so "server" never matches "serve" and "knew" never matches "new",
            // while "recommended" / "recommendation" still match ` recommend`.
            let pat = format!(" {c}");
            padded_hay.contains(&pat)
        }
    })
}

fn interrogative_when(padded: &str) -> bool {
    if has_any(padded, WHEN_INTERROGATIVE_CUES) {
        return true;
    }
    for cue in [" when should ", " when shouldn't ", " when shouldnt "] {
        let mut start = 0;
        while let Some(rel) = padded[start..].find(cue) {
            let abs = start + rel;
            let after = &padded[abs + cue.len()..];
            if !(after.starts_with("i ") || after.starts_with("i'")) {
                return true;
            }
            start = abs + cue.len();
        }
    }
    false
}

pub fn classify(text: &str) -> QueryIntent {
    let lower = word_padded(text);
    let content_groups = raw_tokens(text).into_iter().filter(|w| is_content(w)).collect::<HashSet<_>>().len();
    let user_initiative = has_any(&lower, USER_INITIATIVE_CUES);
    let retrospective_user = user_initiative || has_any(&lower, TIMELINE_CUES);
    let direct_assistant = has_any(&lower, ASSISTANT_ATTRIBUTION_CUES);
    let collective_chat = has_any(&lower, COLLECTIVE_CHAT_CUES);
    let assistant_attribution = direct_assistant || (collective_chat && !user_initiative);
    let when_q = interrogative_when(&lower);
    let sniper_when = when_q || has_any(&lower, &[
        "what date", "what time", "what year", "what month", "what day",
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
        "chronological", "order the", "order these", "timeline of", "i brought up",
    ]);
    QueryIntent {
        preference: has_any(&lower, PREF_QUERY_CUES) && !retrospective_user,
        current_state: has_any(&lower, CURRENT_STATE_CUES),
        assistant_attribution,
        user_attribution: !assistant_attribution && has_any(&lower, USER_ATTRIBUTION_CUES),
        multi_hop: content_groups >= 3 && has_any(&lower, MULTI_HOP_CUES),
        inference: has_any(&lower, INFERENCE_CUES),
        temporal: when_q || has_any(&lower, TEMPORAL_REST_CUES),
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
    let mut focal_ids: HashSet<u32> = HashSet::new();
    let mut focal_domains: HashSet<&'static str> = HashSet::new();
    let use_hyp = p.enable_hypernyms;

    let raw_q = raw_tokens(&q.text);
    let q_lower = word_padded(&q.text);
    let episodic_past_recall = has_any(&q_lower, RECALL_NOT_TASK_CUES)
        || intent.sniper_how_many
        || intent.multi_hop
        || intent.temporal;
    if !episodic_past_recall
        && raw_q.iter().any(|w| ix.vocab.get(w).is_some_and(|id| ix.constraint_terms.contains(&id)))
    {
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

    let nick_bind = ix.nicks.bindings_in(&q.text);
    let mut label_group: HashMap<String, usize> = HashMap::new();
    let mut seen_toks = HashSet::new();
    for w in raw_q.iter().filter(|w| is_content(w)) {
        if !seen_toks.insert(w.clone()) {
            continue;
        }
        let is_frame = can_demote_framing && fw.contains(w.as_str());
        if is_frame {
            if let Some(id) = ix.vocab.get(w) {
                terms.push(QTerm { id, weight: 0.30, group: None, exact_hit: false });
            }
            if p.enable_stemming {
                let st = stem(w);
                if let Some(id) = ix.vocab.get(&stem_term(&st)) {
                    terms.push(QTerm { id, weight: 0.20, group: None, exact_hit: false });
                }
            }
            continue;
        }

        let label = nick_bind.get(w).cloned().unwrap_or_else(|| w.clone());
        let g = if let Some(&g) = label_group.get(&label) {
            g
        } else {
            let g = groups.len();
            groups.push(label.clone());
            label_group.insert(label, g);
            g
        };
        if let Some(id) = ix.vocab.get(w) {
            // Full name owns the person group. A vocative nick ("Mel") that also
            // expands to "melanie" must not score 1.0 on top of that expansion.
            let weight = if nick_bind.get(w).is_some_and(|full| ix.vocab.get(full).is_some()) {
                0.35
            } else {
                1.0
            };
            terms.push(QTerm { id, weight, group: Some(g), exact_hit: true });
            group_of.insert(id, g);
            focal_ids.insert(id);
        }
        if let Some(full) = nick_bind.get(w) {
            if let Some(id) = ix.vocab.get(full) {
                terms.push(QTerm { id, weight: 1.0, group: Some(g), exact_hit: true });
                group_of.insert(id, g);
                focal_ids.insert(id);
            }
            if p.enable_stemming {
                let fst = stem(full);
                if let Some(id) = ix.vocab.get(&stem_term(&fst)) {
                    terms.push(QTerm { id, weight: p.stem_weight, group: Some(g), exact_hit: true });
                    group_of.insert(id, g);
                    focal_ids.insert(id);
                }
            }
        }
        let st = stem(w);
        if p.enable_stemming {
            if let Some(id) = ix.vocab.get(&stem_term(&st)) {
                terms.push(QTerm { id, weight: p.stem_weight, group: Some(g), exact_hit: true });
                group_of.insert(id, g);
                focal_ids.insert(id);
            }
        }
        if let Some(doms) = hypernyms().get(&st) {
            for &d in doms {
                if ix.non_hub_domains.contains(d) {
                    focal_domains.insert(d);
                }
                if use_hyp && hyp_added.insert(d) {
                    if let Some(id) = ix.vocab.get(&hyp_term(d)) {
                        terms.push(QTerm { id, weight: p.hypernym_weight, group: Some(g), exact_hit: false });
                        group_of.insert(id, g);
                    }
                }
            }
        }
    }

    // CamelCase & snake/hyphen reverse query expansion (`localStorage` -> `local`, `storage` at 0.55).
    for raw_word in q.text.split_whitespace() {
        let word = raw_word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-');
        if word.is_empty() {
            continue;
        }
        for (part, _) in surface_aux_tokens(word) {
            if seen_toks.contains(&part) {
                continue;
            }
            if let Some(id) = ix.vocab.get(&part) {
                terms.push(QTerm { id, weight: 0.55, group: None, exact_hit: false });
            }
        }
    }

    // Capitalized proper-noun entity phrases present in index (e.g. "Local H", "For Against").
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
                terms.push(QTerm { id, weight: 1.65, group: Some(g), exact_hit: true });
                group_of.insert(id, g);
            } else {
                terms.push(QTerm { id, weight: 0.75, group: None, exact_hit: false });
            }
        }
    }

    // Exact technical handles in the query (`Url`, `Email`, `CommitSha`, `IssueRef`, `Version`, `FilePath`).
    for h_str in extract_exact_technical_handles(&q.text) {
        if let Some(id) = ix.vocab.get(&h_str) {
            terms.push(QTerm { id, weight: 1.5, group: None, exact_hit: true });
            focal_ids.insert(id);
        }
    }

    if p.enable_preference && intent.preference {
        if let Some(id) = ix.vocab.get(PREF_AXIS) {
            terms.push(QTerm { id, weight: p.preference_axis_weight, group: None, exact_hit: false });
        }
    }

    if p.enable_deictic {
        let mut date_weights: HashMap<String, f32> = HashMap::new();
        for d in inline_dates {
            let w = if d.starts_with("__DATE__") {
                2.0
            } else if d.starts_with("__MD__") {
                1.4
            } else if d.starts_with("__YM__") {
                1.2
            } else if d.starts_with("__SY__") {
                1.1
            } else if d.starts_with("__MONTH__")
                || d.starts_with("__EARLY__")
                || d.starts_with("__MID__")
                || d.starts_with("__LATE__")
            {
                1.0
            } else if d.starts_with("__YEAR__") || d.starts_with("__SEASON__") {
                0.85
            } else {
                p.date_weight
            };
            let entry = date_weights.entry(d).or_insert(0.0);
            *entry = (*entry).max(w);
        }
        for (d, scale) in deictic_items {
            let base = if d.starts_with("__DATE__") {
                1.9
            } else if d.starts_with("__MD__") {
                1.1
            } else if d.starts_with("__YM__") || d.starts_with("__MONTH__") {
                0.85
            } else if d.starts_with("__YEAR__") || d.starts_with("__SEASON__") || d.starts_with("__SY__") {
                0.55
            } else {
                p.date_weight
            };
            let w = base * scale;
            let entry = date_weights.entry(d).or_insert(0.0);
            *entry = (*entry).max(w);
        }
        for (d, w) in date_weights {
            if let Some(id) = ix.vocab.get(&d) {
                terms.push(QTerm { id, weight: w, group: None, exact_hit: false });
            }
        }
    }

    QueryPlan { intent, terms, groups, group_of, focal_ids, focal_domains }
}

// =========================================================================
// SCORING
// =========================================================================

struct Fields {
    w: [f32; 3],
    b: [f32; 3],
}

/// 3-Field BM25F (`user`, `assistant`, `blip_caption`) with IDF power law, IDF-weighted
/// distinct-term coverage, and separate `(any_mask, exact_mask)` so 0.45 hypernym hits
/// never poison submodular coverage masks (`covered_mask` / `covered_query_terms`).
fn bm25f(bag: &Bag, plan: &QueryPlan, st: &Stats, p: &Params, fl: &Fields) -> (f32, u64, u64) {
    let mut score = 0.0f32;
    let mut mask = 0u64;
    let mut exact_mask = 0u64;
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
        for f in 0..3 {
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
                    if qt.exact_hit {
                        exact_mask |= 1 << g;
                    }
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
    (score, mask, exact_mask)
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

fn raise_floor(scores: &mut [f32], lexical_max: f32, frac: f32, keep: impl Fn(usize) -> bool) -> Vec<bool> {
    let floor = if lexical_max > 0.0 { frac * lexical_max } else { 1.0 };
    let mut lifted = vec![false; scores.len()];
    for i in 0..scores.len() {
        if keep(i) && scores[i] < floor {
            scores[i] = floor;
            lifted[i] = true;
        }
    }
    lifted
}

fn ghostlink_hub_damp(deg: u32, own: u32) -> f32 {
    let extra = deg.saturating_sub(own);
    1.0 / (2.0 + extra as f32).log2()
}

fn ochiai_divergence(a: &[u32], b: &[u32]) -> Option<f32> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let (mut i, mut j, mut inter) = (0usize, 0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                inter += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    let denom = ((a.len() * b.len()) as f32).sqrt();
    if denom <= 0.0 {
        None
    } else {
        Some((1.0 - (inter as f32) / denom).clamp(0.0, 1.0))
    }
}

fn named_partner(a: &str, b: &str) -> bool {
    let role = |s: &str| s.is_empty() || s == "user" || s == "assistant";
    !role(a) && !role(b) && a != b
}

/// Stratified Governing Channel (Satchel SCR 2026; Laird SOAR 2012; Forgy Rete 1982):
/// Ranks active standing imperative policies into `RankResult.standing_rules` without
/// consuming or distorting episodic `sessions` / `turns` slots.
fn rank_standing_rules(
    ix: &Index,
    plan: &QueryPlan,
    episodic_user_quote: bool,
    t_mask: &[u64],
    top_k: usize,
) -> Vec<RankedItem> {
    if ix.standing_rules.is_empty() || episodic_user_quote {
        return Vec::new();
    }
    let it = &plan.intent;
    // Multi-evidence recall keeps every standing policy on the stratified channel,
    // including zero-overlap rules. The episodic floor stays off for these intents,
    // so the channel can fire without taking one of the five episodic slots.
    let multi_evidence = it.multi_hop || it.sniper_how_many || it.temporal || it.is_event_sequence;
    let mut latest_by_key: HashMap<&str, (usize, i32, i32, i64)> = HashMap::new();
    for (ri, r) in ix.standing_rules.iter().enumerate() {
        let scope_hits = r.scope.iter().filter(|id| plan.focal_ids.contains(id)).count() as i32;
        let dom_hits = r.domains.iter().filter(|d| plan.focal_domains.contains(*d)).count() as i32;
        let asst_hits = r.assistant_nouns.iter().filter(|id| plan.focal_ids.contains(id)).count() as i32;
        let spec = 5 * scope_hits + 2 * dom_hits + asst_hits;
        let fires = if r.has_explicit_scope {
            spec > 0 || it.task_under_rules || multi_evidence
        } else {
            it.task_under_rules || spec >= 3 || multi_evidence
        };
        if !fires {
            continue;
        }
        let scope_neg_len = -(r.scope.len() as i32);
        let cand = (ri, spec, scope_neg_len, r.ts);
        match latest_by_key.get(r.policy_key.as_str()) {
            Some(&cur) if (cur.3, cur.1, cur.2) >= (cand.3, cand.1, cand.2) => {}
            _ => {
                latest_by_key.insert(r.policy_key.as_str(), cand);
            }
        }
    }
    let max_spec = latest_by_key.values().map(|c| c.1).max().unwrap_or(0);
    let mut cands: Vec<(usize, i32, i32, i64)> = latest_by_key
        .into_values()
        .filter(|c| if max_spec > 0 { c.1 > 0 } else { true })
        .collect();
    cands.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then(b.2.cmp(&a.2))
            .then(b.3.cmp(&a.3))
            .then(a.0.cmp(&b.0))
    });
    cands
        .into_iter()
        .take(top_k.max(1))
        .map(|(ri, spec, _, _)| {
            let r = &ix.standing_rules[ri];
            RankedItem {
                id: ix.turns[r.turn].id.clone(),
                session_id: ix.session_ids[r.session].clone(),
                score: 1.0 + spec.max(0) as f32,
                matched: matched_words(plan, t_mask[r.turn]),
            }
        })
        .collect()
}

fn policy_spec_winner_floor(lexical_max: f32) -> f32 {
    if lexical_max > 0.0 {
        0.715 * lexical_max
    } else {
        0.715 / 0.68
    }
}

fn exact_query_group_ids(plan: &QueryPlan) -> HashSet<u32> {
    plan.terms
        .iter()
        .filter(|t| t.exact_hit && t.group.is_some())
        .map(|t| t.id)
        .collect()
}

fn session_speaker_names(ix: &Index, members: &[usize]) -> HashSet<String> {
    let mut names = HashSet::new();
    for &m in members {
        let sp = ix.turns[m].speaker.as_str();
        if sp.is_empty() || sp == "user" || sp == "assistant" {
            continue;
        }
        for w in raw_tokens(sp) {
            names.insert(w);
        }
    }
    names
}

fn is_speaker_term(ix: &Index, id: u32, names: &HashSet<String>) -> bool {
    let Some(name) = ix.vocab.names.get(id as usize) else { return false };
    let bare = name
        .strip_prefix("\u{1}ent:")
        .or_else(|| name.strip_prefix("\u{1}stem:"))
        .unwrap_or(name.as_str());
    if bare.contains('_') {
        return bare.split('_').any(|p| is_speaker_alias(p, names));
    }
    is_speaker_alias(bare, names)
}

fn clause_hits_query(ids: &HashSet<u32>, g: &HashSet<u32>, ix: &Index, names: &HashSet<String>) -> bool {
    ids.iter().any(|id| g.contains(id) && !is_speaker_term(ix, *id, names))
}

fn turn_echoes_query(ix: &Index, t: &TurnIx, g: &HashSet<u32>, names: &HashSet<String>) -> bool {
    if !t.is_pure_info_question {
        return false;
    }
    if clause_hits_query(&t.pre_question_ids, g, ix, names) {
        return false;
    }
    t.pre_question_ids.is_empty() || clause_hits_query(&t.question_clause_ids, g, ix, names)
}

fn rank_one(ix: &Index, q: &Query, p: &Params, top_k: usize) -> RankResult {
    let plan = plan_query(ix, q, p);
    let it = &plan.intent;
    let pref_on = p.enable_preference && it.preference;
    let q_padded = word_padded(&q.text);
    let persona_infer_on = it.inference
        && !pref_on
        && !it.assistant_attribution
        && !it.temporal
        && !it.sniper_how_many
        && has_any(
            &q_padded,
            &["likely", "would ", "might ", "probably", "considered", "personality", "traits", "what might", "would be", "be interested", "is it possible", "kind of person"],
        );
    let episodic_user_quote = it.user_attribution
        && !pref_on
        && !it.task_under_rules
        && !it.sniper_how_many
        && has_any(
            &q_padded,
            &[
                "did i ", "what did i ", "where did i ", "when did i ", "who did i ",
                "i told you", "i brought up", "did i mention", "my instruction",
                "what was my ", "where is my ", "what is my ",
            ],
        );
    let summary_hold = has_any(&q_padded, SUMMARY_REQUEST_CUES)
        && has_any(&q_padded, TASK_UNDER_RULES_CUES)
        && !has_any(&q_padded, USER_INITIATIVE_CUES);
    let user_role = p.enable_roles
        && it.user_attribution
        && !pref_on
        && !it.task_under_rules
        && !summary_hold;
    let fl = if p.enable_roles && it.assistant_attribution {
        Fields { w: [1.0, p.assistant_attribution_weight, 0.75], b: [p.b_user, p.b_assistant_attributed, 0.0] }
    } else if user_role {
        Fields { w: [p.user_attribution_weight, 0.35, 0.75], b: [p.b_user, p.b_assistant, 0.0] }
    } else if p.enable_roles && pref_on {
        Fields { w: [1.45, 0.55, 0.75], b: [p.b_user, p.b_assistant, 0.0] }
    } else {
        Fields { w: [1.0, 1.0, 0.75], b: [p.b_user, p.b_assistant, 0.0] }
    };
    let ns = ix.session_bags.len();

    // ---- Session level: whole-session BM25F + max-passage window pooling ----
    let mut s_base = vec![0.0f32; ns];
    let mut s_mask = vec![0u64; ns];
    let mut s_exact_mask = vec![0u64; ns];
    for si in 0..ns {
        let (sc, m, em) = bm25f(&ix.session_bags[si], &plan, &ix.s_stats, p, &fl);
        s_base[si] = sc;
        s_mask[si] = m;
        s_exact_mask[si] = em;
    }
    let mut s_win = vec![0.0f32; ns];
    if p.enable_window {
        for (wi, &(si, a, b)) in ix.windows.iter().enumerate() {
            let (mut sc, m, em) = bm25f(&ix.window_bags[wi], &plan, &ix.w_stats, p, &fl);
            if sc <= 0.0 {
                continue;
            }
            let members = &ix.session_turns[si][a..=b];
            if p.enable_proximity
                && proximity_hit(members.iter().flat_map(|&t| ix.turns[t].seq.iter()), &plan, p.proximity_span)
            {
                sc *= 1.0 + p.proximity_bonus;
            }
            if pref_on {
                let mut best_pref = 1.0f32;
                for &t in members {
                    let tr = &ix.turns[t];
                    if tr.field == 0 {
                        if tr.pref_marker {
                            best_pref = best_pref.max(p.preference_boost);
                        } else if tr.is_user_schema
                            && tr.schema_domains.iter().any(|d| plan.focal_domains.contains(d))
                        {
                            // Tse et al. (Science 2007) domain-gated user schema activation:
                            // habitual/behavioral declarations boost only when matching query domain.
                            best_pref = best_pref.max(1.45);
                        }
                    }
                }
                sc *= best_pref;
            } else if persona_infer_on && members.iter().any(|&t| ix.turns[t].pref_marker || ix.turns[t].is_user_schema) {
                sc *= 1.38;
            }
            if members.iter().any(|&t| ix.turns[t].is_standing_policy) {
                sc *= 1.8;
            }
            if sc > s_win[si] {
                s_win[si] = sc;
            }
            s_mask[si] |= m;
            s_exact_mask[si] |= em;
        }
    }
    if pref_on {
        for si in 0..ns {
            let mut best = 1.0f32;
            for &ti in &ix.session_turns[si] {
                let tr = &ix.turns[ti];
                if tr.field != 0 {
                    continue;
                }
                let domain_hit = tr.schema_domains.iter().any(|d| plan.focal_domains.contains(d));
                let focal_hit = tr.content.iter().any(|id| plan.focal_ids.contains(id))
                    || tr.feats.iter().any(|f| plan.focal_ids.contains(&f.id));
                if tr.pref_marker && (focal_hit || domain_hit) {
                    best = best.max(p.preference_boost);
                } else if tr.is_user_schema && domain_hit {
                    best = best.max(1.45);
                }
            }
            if best > 1.0 {
                s_base[si] *= best;
            }
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

    // ---- Turn level: score BOTH uncarried `t_own` AND carried `t_raw` ----
    let nt = ix.turns.len();
    let mut t_own = vec![0.0f32; nt];
    let mut t_raw = vec![0.0f32; nt];
    let mut t_mask = vec![0u64; nt];
    let mut t_exact_mask = vec![0u64; nt];
    for ti in 0..nt {
        let t = &ix.turns[ti];
        let (mut sc_own, _, em_own) = bm25f(&ix.turn_own_bags[ti], &plan, &ix.t_own_stats, p, &fl);
        let (mut sc, m, _) = bm25f(&ix.turn_bags[ti], &plan, &ix.t_stats, p, &fl);
        if sc <= 0.0 && sc_own <= 0.0 {
            continue;
        }
        let prox = p.enable_proximity && proximity_hit(t.seq.iter(), &plan, p.proximity_span);
        if prox {
            sc *= 1.0 + p.proximity_bonus;
            sc_own *= 1.0 + p.proximity_bonus;
        }
        if p.enable_carry && t.answers_question {
            sc *= p.answer_boost;
            sc_own *= p.answer_boost;
        } else if p.enable_carry && t.is_short_reply {
            sc *= 1.05;
            sc_own *= 1.05;
        }
        if pref_on && t.field == 0 {
            if t.pref_marker {
                sc *= p.preference_boost;
                sc_own *= p.preference_boost;
            } else if t.is_user_schema && t.schema_domains.iter().any(|d| plan.focal_domains.contains(d)) {
                sc *= 1.45;
                sc_own *= 1.45;
            }
        } else if persona_infer_on && (t.pref_marker || t.is_user_schema) {
            sc *= 1.40;
            sc_own *= 1.40;
        }
        t_own[ti] = sc_own;
        t_raw[ti] = sc;
        t_mask[ti] = m;
        t_exact_mask[ti] = em_own;
    }

    // Weapon 1: Bidirectional 3-Turn Scene Pooling.
    // - On named-speaker dialogues (`LoCoMo` / multi-person chats), pool over uncarried `t_own` +
    //   Center-Peak Monotonicity so follow-up questions & reactions (`gold + 1`) never steal Rank 1!
    // - On `user`/`assistant` dialogues (`BEAM` / `LME`), preserve symmetric `t_raw` pooling so
    //   concise user turns ending in `?` are never inverted beneath verbose assistant expansions!
    let mut t_scene = vec![0.0f32; nt];
    if p.enable_scene {
        for members in &ix.session_turns {
            let has_named_dialogue = members.iter().any(|&m| {
                let sp = ix.turns[m].speaker.as_str();
                !sp.is_empty() && sp != "user" && sp != "assistant"
            });
            if has_named_dialogue {
                for (pos, &ti) in members.iter().enumerate() {
                    let prev_own = if pos > 0 { t_own[members[pos - 1]] } else { 0.0 };
                    let next_own = if pos + 1 < members.len() { t_own[members[pos + 1]] } else { 0.0 };
                    let prev_w = if ix.turns[ti].is_backchannel {
                        0.22
                    } else if pos > 0 && ix.turns[members[pos - 1]].is_question {
                        0.75
                    } else {
                        0.48
                    };
                    let self_score = if ix.turns[ti].answers_question || ix.turns[ti].is_short_reply || ix.turns[ti].anaphoric {
                        t_raw[ti]
                    } else {
                        t_own[ti] + 0.18 * (t_raw[ti] - t_own[ti]).max(0.0)
                    };
                    t_scene[ti] = self_score + prev_w * prev_own + 0.42 * next_own;
                    if pos > 0 && !ix.turns[ti].is_backchannel {
                        t_mask[ti] |= t_mask[members[pos - 1]];
                    }
                    if pos + 1 < members.len() && !ix.turns[ti].is_question {
                        t_mask[ti] |= t_mask[members[pos + 1]];
                    }
                }
                for pos in 1..members.len() {
                    let prev = members[pos - 1];
                    let ti = members[pos];
                    if !ix.turns[prev].is_question
                        && !ix.turns[prev].is_backchannel
                        && t_own[prev] > 0.0
                        && (t_exact_mask[ti] & !t_exact_mask[prev]) == 0
                        && t_own[ti] <= t_own[prev] * 1.05
                        && t_scene[ti] > 0.92 * t_scene[prev]
                    {
                        t_scene[ti] = 0.92 * t_scene[prev];
                    }
                }
            } else {
                for (pos, &ti) in members.iter().enumerate() {
                    let prev_raw = if pos > 0 { t_raw[members[pos - 1]] } else { 0.0 };
                    let next_raw = if pos + 1 < members.len() { t_raw[members[pos + 1]] } else { 0.0 };
                    let prev_w = if ix.turns[ti].is_backchannel {
                        0.25
                    } else if pos > 0 && ix.turns[members[pos - 1]].is_question {
                        0.75
                    } else {
                        0.55
                    };
                    t_scene[ti] = 1.0 * t_raw[ti] + prev_w * prev_raw + 0.55 * next_raw;
                    if pos > 0 && !ix.turns[ti].is_backchannel {
                        t_mask[ti] |= t_mask[members[pos - 1]];
                    }
                    if pos + 1 < members.len() {
                        t_mask[ti] |= t_mask[members[pos + 1]];
                    }
                }
            }
        }
    } else {
        t_scene = t_raw.clone();
    }

    // Named-speaker boost (never a filter): 1.06x tie-breaker gated on t_own[ti] > 0 or t_raw[ti] > 0.
    for ti in 0..nt {
        if t_raw[ti] <= 0.0 && t_own[ti] <= 0.0 {
            continue;
        }
        let sp = ix.turns[ti].speaker.as_str();
        if !sp.is_empty() && sp != "user" && sp != "assistant" && plan.groups.iter().any(|g| g == sp) {
            t_scene[ti] *= 1.06;
        }
    }

    // Turn-level role alignment: suppress assistant turns on explicit user-role queries,
    // except when no user turn in the same session covers that assistant turn's exact query bits!
    if user_role {
        for ti in 0..nt {
            if ix.turns[ti].field == 1 {
                let s_idx = ix.turns[ti].session;
                let a_bits = t_exact_mask[ti];
                let user_covers = a_bits == 0
                    || ix.session_turns[s_idx]
                        .iter()
                        .any(|&u| ix.turns[u].field == 0 && (t_exact_mask[u] & a_bits) == a_bits);
                if user_covers {
                    t_scene[ti] *= 0.35;
                } else {
                    t_scene[ti] *= 0.72;
                }
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

    let exact_groups = exact_query_group_ids(&plan);

    // Weapon 2: Deterministic Answer-Type Sniper + Query-Local Question-Echo Demotion
    if p.enable_sniper {
        for members in &ix.session_turns {
            let has_named_dialogue = members.iter().any(|&m| {
                let sp = ix.turns[m].speaker.as_str();
                !sp.is_empty() && sp != "user" && sp != "assistant"
            });
            let speaker_names = if has_named_dialogue {
                session_speaker_names(ix, members)
            } else {
                HashSet::new()
            };
            let has_declarative_peer = members.iter().any(|&m| !ix.turns[m].is_question && !ix.turns[m].is_backchannel);
            for &ti in members {
                if t_scene[ti] <= 0.0 {
                    continue;
                }
                let t = &ix.turns[ti];
                if it.sniper_when {
                    if t.has_temporal {
                        t_scene[ti] *= 1.65;
                    }
                    if t.is_question {
                        t_scene[ti] *= 0.55;
                    }
                } else if has_named_dialogue
                    && has_declarative_peer
                    && !episodic_user_quote
                    && turn_echoes_query(ix, t, &exact_groups, &speaker_names)
                {
                    t_scene[ti] *= 0.62;
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
                if t.is_backchannel {
                    t_scene[ti] *= 0.55;
                }
            }
        }
    }

    // Named-Partner cQA Question-Echo Swap (strictly `named_partner` ONLY — never `user`/`assistant`!):
    let mut q_ents_and_vocatives: HashSet<u32> = HashSet::new();
    for ent in proper_entity_tokens(&q.text) {
        if let Some(id) = ix.vocab.get(&ent) {
            q_ents_and_vocatives.insert(id);
        }
    }
    for w in raw_tokens(&q.text) {
        if let Some(id) = ix.vocab.get(&format!("\u{1}ent:{w}")) {
            q_ents_and_vocatives.insert(id);
        }
    }
    let query_has_temp = it.temporal || it.sniper_when || !inline_date_tokens(&q.text).is_empty();
    let query_has_num = it.sniper_how_many || q.text.chars().any(|c| c.is_ascii_digit());
    if p.enable_dyad {
        for members in &ix.session_turns {
            let speaker_names = session_speaker_names(ix, members);
            if speaker_names.is_empty() {
                continue;
            }
            let mut sess_speaker_ents: HashSet<u32> = HashSet::new();
            for sw in &speaker_names {
                if let Some(id) = ix.vocab.get(&format!("\u{1}ent:{sw}")) {
                    sess_speaker_ents.insert(id);
                }
            }
            for pos in 0..members.len() {
                let qi = members[pos];
                let qt = &ix.turns[qi];
                if t_scene[qi] <= 0.0 || !turn_echoes_query(ix, qt, &exact_groups, &speaker_names) {
                    continue;
                }
                let novel_temp = qt.has_temporal && !query_has_temp;
                let novel_num = qt.has_numeric && !query_has_num;
                let novel_ent = qt
                    .entities
                    .iter()
                    .any(|eid| !q_ents_and_vocatives.contains(eid) && !sess_speaker_ents.contains(eid));
                if novel_temp || novel_num || novel_ent {
                    continue;
                }
                if let Some(&ai) = members.get(pos + 1) {
                    let ans = &ix.turns[ai];
                    if named_partner(&qt.speaker, &ans.speaker) && !ans.is_question && !ans.is_backchannel {
                        t_mask[ai] |= t_mask[qi];
                        t_exact_mask[ai] |= t_exact_mask[qi];
                        if t_scene[qi] > t_scene[ai] {
                            t_scene[ai] = t_scene[ai].max(t_scene[qi] * 1.06);
                            t_scene[qi] *= 0.65;
                        }
                    }
                }
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

                // Pass 1: Normalized Seed Turn -> Discriminative Bridge Entities + Rare Exact Symbols
                for &s in &seed_indices {
                    if t_scene[s] < 0.15 * t_max {
                        break;
                    }
                    let seed_norm = (t_scene[s] / t_max) / ((ix.turns[s].content.len() + 4) as f32).sqrt();
                    for &e in &ix.turns[s].entities {
                        if is_technical_handle(&ix.vocab, e) {
                            continue;
                        }
                        if let Some(&df) = ix.t_stats.df.get(&e) {
                            if df >= 2 && df <= max_df {
                                let idf_term = 1.4 / ((df as f32).sqrt() * (df as f32 + 1.1).ln());
                                *activation.entry(e).or_insert(0.0) += 2.0 * seed_norm * idf_term;
                            }
                        }
                    }
                    for &w in &ix.turns[s].content {
                        if plan.group_of.contains_key(&w) {
                            continue;
                        }
                        if let Some(&df) = ix.t_stats.df.get(&w) {
                            if df >= 2 && df <= max_df {
                                let idf_term = 1.2 / ((df as f32).sqrt() * (df as f32 + 1.1).ln());
                                *activation.entry(w).or_insert(0.0) += 0.75 * seed_norm * idf_term;
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
                        let mut ppr_sum = ent_hit;
                        for &w in &ix.turns[ti].content {
                            if let Some(&act) = activation.get(&w) {
                                ppr_sum += 0.75 * act;
                            }
                        }
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

    // Adjacency-pair dyad coupling: answers to questions AND short commitment/value replies
    // to statements ("Done.", "Last Tuesday.", "Me too!") ride alongside their setup turn.
    if p.enable_dyad {
        let base = t_scene.clone();
        for members in &ix.session_turns {
            for pos in 1..members.len() {
                let (a, b) = (members[pos - 1], members[pos]);
                if !ix.turns[b].answers_question && !ix.turns[b].is_short_reply {
                    continue;
                }
                let partner_ratio = if ix.turns[b].answers_question { p.dyad_partner } else { 0.84 };
                let (hi, lo) = if base[a] >= base[b] { (a, b) } else { (b, a) };
                if base[hi] <= 0.0 {
                    continue;
                }
                let dyad = base[hi] + p.dyad * base[lo];
                t_scene[hi] = t_scene[hi].max(dyad);
                t_scene[lo] = t_scene[lo].max(dyad * partner_ratio);
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
    let lexical_smax = s_score.iter().cloned().fold(0.0f32, f32::max);
    let lexical_tmax = t_scene.iter().cloned().fold(0.0f32, f32::max);
    let multi_evidence_intent = it.multi_hop
        || it.sniper_how_many
        || it.temporal
        || it.is_event_sequence;
    // Compound/prepositional task wording (" and ", "first ", "before ") must not
    // drop the standing-policy floor. Only count-snipers and event sequences do.
    let policy_blocked = it.sniper_how_many || it.is_event_sequence;
    let standing_rules = rank_standing_rules(ix, &plan, episodic_user_quote, &t_mask, k);
    // Winner floor stays inside task_under_rules, including multi-hop task rules.
    // v1 moved this seat outside that block and excluded multi-hop. Temporal queries
    // then received a global policy seat and displaced gold. v2 does not do that.
    let (s_policy_floored, t_policy_floored) = if it.task_under_rules && !policy_blocked {
        let mut s_floored = raise_floor(&mut s_score, lexical_smax, 0.68, |si| {
            ix.session_turns[si].iter().any(|&ti| ix.turns[ti].is_standing_policy)
        });
        let mut t_floored = raise_floor(&mut t_scene, lexical_tmax, 0.68, |ti| ix.turns[ti].is_standing_policy);
        if let Some(top) = standing_rules.first() {
            let s_win_floor = policy_spec_winner_floor(lexical_smax);
            let t_win_floor = policy_spec_winner_floor(lexical_tmax);
            if let Some(top_rule_sess) = ix.session_ids.iter().position(|id| id == &top.session_id) {
                if s_score[top_rule_sess] < s_win_floor {
                    s_score[top_rule_sess] = s_win_floor;
                    s_floored[top_rule_sess] = true;
                }
            }
            if let Some(top_rule_turn) = ix.turns.iter().position(|t| t.id == top.id) {
                if t_scene[top_rule_turn] < t_win_floor {
                    t_scene[top_rule_turn] = t_win_floor;
                    t_floored[top_rule_turn] = true;
                }
            }
        }
        (s_floored, t_floored)
    } else {
        (vec![false; ns], vec![false; nt])
    };
    let (s_pref_floored, t_pref_floored) = if pref_on && !multi_evidence_intent {
        (
            raise_floor(&mut s_score, lexical_smax, 0.50, |si| {
                ix.session_turns[si].iter().any(|&ti| ix.turns[ti].is_personal_constraint)
            }),
            raise_floor(&mut t_scene, lexical_tmax, 0.50, |ti| ix.turns[ti].is_personal_constraint),
        )
    } else {
        (vec![false; ns], vec![false; nt])
    };
    let smax = s_score.iter().cloned().fold(0.0f32, f32::max);
    if it.is_event_sequence {
        let _ = raise_floor(&mut s_score, lexical_smax, 0.42, |si| {
            ix.session_turns[si].iter().any(|&ti| ix.turns[ti].is_initiative)
        });
        let _ = raise_floor(&mut t_scene, lexical_tmax, 0.38, |ti| ix.turns[ti].is_initiative);
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
            let anchors: Vec<usize> = s_ord_base
                .iter()
                .copied()
                .filter(|&si| !s_policy_floored[si] && !s_pref_floored[si] && s_score[si] >= 0.62 * smax)
                .take(3)
                .collect();
            if !ix.header_dict.is_empty() && ix.header_max_words > 0 && !it.preference && !it.user_attribution {
                let mut hdr_best = vec![0.0f32; ns];
                // Weapon 1: Query-direct canonical header-title match (`1 <= deg <= 4`).
                let (q_ids, q_proper): (Vec<u32>, Vec<bool>) = q
                    .text
                    .split(|c: char| !c.is_alphanumeric())
                    .filter(|s| !s.is_empty())
                    .map(|s| {
                        let proper = s.chars().next().is_some_and(|c| c.is_uppercase());
                        let id = ix.vocab.get(&s.to_lowercase()).unwrap_or(u32::MAX);
                        (id, proper)
                    })
                    .unzip();
                let mut q_idx = 0usize;
                while q_idx < q_ids.len() {
                    let max_l = ix.header_max_words.min(q_ids.len() - q_idx);
                    // Every span length, then advance one token. A degree-4 outer phrase
                    // must not shadow a degree-1 canonical entity header inside it.
                    for len in 1..=max_l {
                        let slice = &q_ids[q_idx..q_idx + len];
                        if slice.contains(&u32::MAX) {
                            continue;
                        }
                        if len == 1 && !q_proper[q_idx] {
                            continue;
                        }
                        let hh = hash_u32_slice(slice);
                        if let Some(bucket) = ix.header_dict.get(&hh) {
                            if let Some(hk) = bucket.iter().find(|k| k.tokens.as_slice() == slice) {
                                let deg = hk.sessions.len() as u32;
                                if (1..=4).contains(&deg) {
                                    let extra = deg.saturating_sub(1);
                                    let damp = 1.0 / (2.0 + extra as f32).log2();
                                    for &(target_si, mult) in &hk.sessions {
                                        let cand_b = 0.68 * smax * mult * damp;
                                        if cand_b > hdr_best[target_si] {
                                            hdr_best[target_si] = cand_b;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    q_idx += 1;
                }
                for &anc in &anchors {
                    let anc_w = (s_score[anc] / smax).clamp(0.35, 1.0);
                    let members = &ix.session_turns[anc];
                    let anc_tmax = members.iter().map(|&ti| t_scene[ti]).fold(0.0f32, f32::max);
                    let use_hot = anc_tmax > 0.0 && members.len() > 2;
                    for &ti in members {
                        if use_hot && t_scene[ti] < 0.25 * anc_tmax {
                            continue;
                        }
                        let hot_turn_bonus = if anc_tmax > 0.0 && t_scene[ti] >= 0.60 * anc_tmax { 1.18 } else { 1.0 };
                        let toks = ix.turns[ti].span_ids_cache.get_or_init(|| {
                            ix.turns[ti]
                                .body_for_headers
                                .split(|c: char| !c.is_alphanumeric())
                                .filter(|s| !s.is_empty())
                                .map(|s| ix.vocab.get(&s.to_lowercase()).unwrap_or(u32::MAX))
                                .collect()
                        });
                        let mut idx = 0usize;
                        while idx < toks.len() {
                            let max_l = ix.header_max_words.min(toks.len() - idx);
                            let mut matched_len = 0usize;
                            for len in (1..=max_l).rev() {
                                let slice = &toks[idx..idx + len];
                                if slice.contains(&u32::MAX) {
                                    continue;
                                }
                                let hh = hash_u32_slice(slice);
                                if let Some(bucket) = ix.header_dict.get(&hh) {
                                    if let Some(hk) = bucket.iter().find(|k| k.tokens.as_slice() == slice) {
                                        let deg = hk.sessions.len() as u32;
                                        if (1..=6).contains(&deg) {
                                            let extra = deg.saturating_sub(1);
                                            let damp = 1.0 / (2.0 + extra as f32).log2();
                                            for &(target_si, mult) in &hk.sessions {
                                                if target_si != anc && target_si != top0 {
                                                    let cand_b = smax * 0.76 * anc_w * mult * damp * hot_turn_bonus;
                                                    if cand_b > hdr_best[target_si] {
                                                        hdr_best[target_si] = cand_b;
                                                    }
                                                }
                                            }
                                            matched_len = len;
                                            break;
                                        }
                                    }
                                }
                            }
                            idx += matched_len.max(1);
                        }
                    }
                }
                for si in 0..ns {
                    if hdr_best[si] > bridge_boost[si] {
                        bridge_boost[si] = hdr_best[si];
                    }
                }
            }
            let max_s_df = if ns >= 8 {
                ((ix.s_stats.n * 0.10).clamp(4.0, (0.28 * ix.s_stats.n).max(4.0)).min(28.0)) as u32
            } else {
                4
            };
            let rare_sym_max_df = ((ix.s_stats.n * 0.05).clamp(3.0, 18.0)) as u32;
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
                        if is_technical_handle(&ix.vocab, e) {
                            continue;
                        }
                        if let Some(&df) = ix.s_stats.df.get(&e) {
                            if df >= 2 && df <= max_s_df {
                                let idf_s = 1.4 / ((df as f32).sqrt() * (df as f32 + 1.1).ln());
                                *s_act.entry(e).or_insert(0.0) += (anc_w * 2.5 * idf_s) / turn_len_norm;
                            }
                        }
                    }
                    // Exact byte-equal rare symbols (content holds unstemmed exact token IDs only, never stems).
                    for &w in &ix.turns[ti].content {
                        if plan.group_of.contains_key(&w) {
                            continue;
                        }
                        if let Some(&df) = ix.s_stats.df.get(&w) {
                            if df >= 2 && df <= rare_sym_max_df {
                                let idf_s = 1.2 / ((df as f32).sqrt() * (df as f32 + 1.1).ln());
                                *s_act.entry(w).or_insert(0.0) += (anc_w * 0.75 * idf_s) / turn_len_norm;
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
                    let mut sum = ent_sum;
                    for &w in &ix.session_content[si] {
                        if let Some(&a) = s_act.get(&w) {
                            sum += 0.75 * a;
                        }
                    }
                    // Require direct query overlap (s_score > 0), or an explicit proper-entity bridge
                    // on non-preference/non-user queries, so zero-match distractors never displace personal facts.
                    let allow_zero_hit = !it.preference && !it.user_attribution && ent_sum > 0.0;
                    if sum > 0.0 && (s_score[si] > 0.0 || allow_zero_hit) {
                        let slen = (ix.session_bags[si].len[0] + ix.session_bags[si].len[1] + 8.0).sqrt();
                        let norm = sum / slen;
                        let sat = norm / (norm + 0.05);
                        let mut coeff = if it.multi_hop || plan.groups.len() >= 3 { 0.55 } else { 0.12 };
                        if (s_exact_mask[si] & !s_exact_mask[top0]) != 0 {
                            coeff *= 1.30;
                        }
                        let boosted = smax * coeff * sat;
                        if boosted > bridge_boost[si] {
                            bridge_boost[si] = boosted;
                        }
                    }
                }
            }

            // Backward-only RSB Handle/CaptureWindow, Structural Contradiction Supersession,
            // and Temporal Context Episode Contiguity (`later -> earlier` ONLY).
            for &anc in &anchors {
                let anc_w = (s_score[anc] / smax).clamp(0.35, 1.0);
                // 1. Exact RSB technical handles & 9h Synaptic Tagging CaptureWindow backward edges
                for &(deg, prev_si) in &ix.session_rsb_back[anc] {
                    if prev_si != top0 {
                        let damp = ghostlink_hub_damp(deg.max(2), 1);
                        let cand = smax * 0.64 * anc_w * damp;
                        if cand > bridge_boost[prev_si] {
                            bridge_boost[prev_si] = cand;
                        }
                    }
                }
                // 2. Structural contradiction / state-update supersession (`!it.current_state` pulls the earlier state
                //    when the earlier session also has lexical grounding or user attribution, without displacing QA2 header bridges)
                if !it.current_state {
                    for &prev_si in &ix.session_supersedes_back[anc] {
                        if prev_si != top0 && (s_score[prev_si] >= 0.12 * smax || it.user_attribution) {
                            let cand = smax * 0.44 * anc_w;
                            if cand > bridge_boost[prev_si] {
                                bridge_boost[prev_si] = cand;
                            }
                        }
                    }
                }
                // 3. Temporal Context Episode backward contiguity (`session_episode_prev`: `si -> si - 1`)
                if it.is_event_sequence {
                    let mut cur = anc;
                    let mut decay = 0.52f32;
                    for _ in 0..2 {
                        let Some(prev_ep) = ix.session_episode_prev[cur] else { break };
                        if prev_ep != top0 {
                            let cand = smax * decay * anc_w;
                            if cand > bridge_boost[prev_ep] {
                                bridge_boost[prev_ep] = cand;
                            }
                        }
                        cur = prev_ep;
                        decay *= 0.65;
                    }
                }
            }
        }
    }

    let total_q_groups = plan.groups.len().min(64);
    let mask_limit = if total_q_groups == 64 { u64::MAX } else { (1u64 << total_q_groups) - 1 };

    let final_session_indices: Vec<usize> = if p.enable_submodular && ns > 1 {
        let mut selected: Vec<usize> = Vec::with_capacity(k);
        if let Some(&top0) = s_ord_base.first() {
            if s_score[top0] > 0.0 {
                selected.push(top0);
                let mut covered_mask = s_mask[top0];
                let mut covered_ents: HashSet<u32> = ix.session_entities[top0].clone();
                let mut mask_counts: HashMap<u64, usize> = HashMap::new();
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
                        let policy_prior = s_policy_floored[si];
                        let pref_prior = s_pref_floored[si] && !policy_prior;
                        let zero_prior = (policy_prior || pref_prior) && m_eff == 0;
                        let policy_slots = selected.iter().filter(|&&s| s_policy_floored[s]).count();
                        let pref_slots = selected.iter().filter(|&&s| s_pref_floored[s] && !s_policy_floored[s]).count();
                        let zero_slots = selected.iter().filter(|&&s| {
                            (s_policy_floored[s] || s_pref_floored[s]) && (s_mask[s] & mask_limit) == 0
                        }).count();
                        if (policy_prior && policy_slots >= 1)
                            || (pref_prior && pref_slots >= 1)
                            || (zero_prior && zero_slots >= 1)
                        {
                            continue;
                        }
                        let shared_ent = ix.session_entities[si].iter().any(|e| covered_ents.contains(e));
                        let same_date_seen = it.is_event_sequence && selected.iter().any(|&s| ix.session_ts[s] == ix.session_ts[si] && ix.session_ts[si] > 0);
                        let topic_seen = it.is_event_sequence
                            && !initiative_content[si].is_empty()
                            && selected.iter().any(|&s| jaccard(&initiative_content[s], &initiative_content[si]) >= 0.45);
                        let new_event = it.is_event_sequence && !initiative_content[si].is_empty() && !topic_seen;
                        let min_ochiai = selected
                            .iter()
                            .filter_map(|&s| ochiai_divergence(&ix.session_rare_nt[s], &ix.session_rare_nt[si]))
                            .fold(None::<f32>, |acc, d| Some(acc.map_or(d, |a| a.min(d))));
                        let distinct_episode = min_ochiai.is_some_and(|d| d >= 0.58)
                            && (ix.session_ts[si] == 0 || selected.iter().all(|&s| ix.session_ts[s] != ix.session_ts[si]));
                        let redundancy = if same_date_seen {
                            0.25
                        } else if topic_seen {
                            0.35
                        } else if !it.is_event_sequence
                            && !it.sniper_how_many
                            && !distinct_episode
                            && uncovered == 0.0
                            && bridge_boost[si] < 0.12 * smax
                            && shared_ent
                            && same_mask_seen >= 3
                        {
                            0.62
                        } else {
                            1.0
                        };
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
                    let chosen = remaining.remove(best_pos);
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
            if selected.contains(si) {
                continue;
            }
            if s_score[*si] <= 0.0 && bridge_boost[*si] <= 0.0 {
                continue;
            }
            let policy_prior = s_policy_floored[*si];
            let pref_prior = s_pref_floored[*si] && !policy_prior;
            let zero_prior = (policy_prior || pref_prior) && (s_mask[*si] & mask_limit) == 0;
            let policy_slots = selected.iter().filter(|&&s| s_policy_floored[s]).count();
            let pref_slots = selected.iter().filter(|&&s| s_pref_floored[s] && !s_policy_floored[s]).count();
            let zero_slots = selected.iter().filter(|&&s| {
                (s_policy_floored[s] || s_pref_floored[s]) && (s_mask[s] & mask_limit) == 0
            }).count();
            if (policy_prior && policy_slots >= 1)
                || (pref_prior && pref_slots >= 1)
                || (zero_prior && zero_slots >= 1)
            {
                continue;
            }
            selected.push(*si);
        }
        selected
    } else {
        s_ord_base
            .into_iter()
            .filter(|&si| s_score[si] > 0.0 || bridge_boost[si] > 0.0)
            .take(k)
            .collect()
    };

    let strong_top_sessions = final_session_indices
        .iter()
        .take(5)
        .filter(|&&si| smax > 0.0 && !s_policy_floored[si] && !s_pref_floored[si] && (s_score[si] + bridge_boost[si]) >= 0.62 * smax)
        .count();
    let sessions = final_session_indices
        .into_iter()
        .map(|i| RankedItem {
            id: ix.session_ids[i].clone(),
            session_id: ix.session_ids[i].clone(),
            score: s_score[i] + bridge_boost[i],
            matched: matched_words(&plan, s_mask[i]),
        })
        .collect();

    // Weapon 4: Greedy Submodular Set Cover for Top-K Turn Selection
    let max_per_session_top5 = if it.is_event_sequence {
        1
    } else if it.multi_hop || (!pref_on && !it.task_under_rules && strong_top_sessions >= 2) {
        2
    } else {
        3
    };

    let turn_ts: Vec<i64> = ix.turns.iter().map(|t| ix.session_ts[t.session]).collect();
    let mut final_turn_indices: Vec<usize> = if p.enable_submodular {
        let mut selected_turns: Vec<usize> = Vec::with_capacity(k);
        let mut covered_query_terms: u64 = 0;
        let mut session_counts: HashMap<usize, usize> = HashMap::new();
        // Ingest order. Equal gains keep the earlier turn, then the higher recorded stability.
        let mut remaining: Vec<usize> = (0..nt).filter(|&ti| t_scene[ti] > 0.0).collect();

        let total_query_groups = plan.groups.len().min(64);
        let turn_mask_limit = if total_query_groups == 64 { u64::MAX } else { (1u64 << total_query_groups) - 1 };

        while selected_turns.len() < k && !remaining.is_empty() {
            let mut best_turn: Option<usize> = None;
            let mut best_pos = 0usize;
            let mut best_gain = -1.0f32;

            for (pos, &ti) in remaining.iter().enumerate() {
                let sc = t_scene[ti];
                let t_sess = ix.turns[ti].session;
                let count = session_counts.get(&t_sess).copied().unwrap_or(0);
                let topic_repeat = it.is_event_sequence
                    && ix.turns[ti].is_initiative
                    && selected_turns.iter().any(|&prev| {
                        ix.turns[prev].is_initiative && jaccard(&ix.turns[prev].content, &ix.turns[ti].content) >= 0.45
                    });
                let t_m = t_mask[ti];
                let uncovered_mask = t_exact_mask[ti] & !covered_query_terms;
                let uncovered_count = (uncovered_mask & turn_mask_limit).count_ones() as f32;
                let short_reply = (ix.turns[ti].is_short_reply
                    || (ix.turns[ti].answers_question && ix.turns[ti].content.len() <= 6))
                    && selected_turns.iter().any(|&prev| {
                        ix.turns[prev].session == t_sess && (prev as isize - ti as isize).abs() == 1
                    });
                let policy_prior = t_policy_floored[ti];
                let pref_prior = t_pref_floored[ti] && !policy_prior;
                let zero_prior = (policy_prior || pref_prior) && (t_m & turn_mask_limit) == 0;
                let policy_slots = selected_turns.iter().filter(|&&t| t_policy_floored[t]).count();
                let pref_slots = selected_turns.iter().filter(|&&t| t_pref_floored[t] && !t_policy_floored[t]).count();
                let zero_slots = selected_turns.iter().filter(|&&t| {
                    (t_policy_floored[t] || t_pref_floored[t]) && (t_mask[t] & turn_mask_limit) == 0
                }).count();
                if (policy_prior && policy_slots >= 1)
                    || (pref_prior && pref_slots >= 1)
                    || (zero_prior && zero_slots >= 1)
                {
                    continue;
                }
                // One chat may not fill the top 5. A short reply beside a hit is the same evidence, not a new slot.
                let session_capped = selected_turns.len() < 5
                    && count >= max_per_session_top5
                    && !(short_reply && !it.is_event_sequence);
                if session_capped {
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
                let stab = ix.turns[ti].stability;
                let take = match best_turn {
                    None => true,
                    Some(cur) => {
                        gain > best_gain
                            || (gain == best_gain && stab > ix.turns[cur].stability)
                    }
                };
                if take {
                    best_gain = gain;
                    best_turn = Some(ti);
                    best_pos = pos;
                }
            }

            if let Some(chosen) = best_turn {
                selected_turns.push(chosen);
                remaining.remove(best_pos);
                covered_query_terms |= t_exact_mask[chosen];
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
                if selected_turns.contains(&ti) {
                    continue;
                }
                if t_scene[ti] <= 0.0 {
                    continue;
                }
                let policy_prior = t_policy_floored[ti];
                let pref_prior = t_pref_floored[ti] && !policy_prior;
                let zero_prior = (policy_prior || pref_prior) && (t_mask[ti] & turn_mask_limit) == 0;
                let policy_slots = selected_turns.iter().filter(|&&t| t_policy_floored[t]).count();
                let pref_slots = selected_turns.iter().filter(|&&t| t_pref_floored[t] && !t_policy_floored[t]).count();
                let zero_slots = selected_turns.iter().filter(|&&t| {
                    (t_policy_floored[t] || t_pref_floored[t]) && (t_mask[t] & turn_mask_limit) == 0
                }).count();
                if (policy_prior && policy_slots >= 1)
                    || (pref_prior && pref_slots >= 1)
                    || (zero_prior && zero_slots >= 1)
                {
                    continue;
                }
                let count = selected_turns.iter().filter(|&&p| ix.turns[p].session == ix.turns[ti].session).count();
                let short_reply = (ix.turns[ti].is_short_reply
                    || (ix.turns[ti].answers_question && ix.turns[ti].content.len() <= 6))
                    && selected_turns.iter().any(|&prev| {
                        ix.turns[prev].session == ix.turns[ti].session && (prev as isize - ti as isize).abs() == 1
                    });
                if selected_turns.len() < 5 && count >= max_per_session_top5 && !(short_reply && !it.is_event_sequence) {
                    continue;
                }
                selected_turns.push(ti);
            }
        }
        selected_turns
    } else {
        order(&t_scene, &turn_ts)
            .into_iter()
            .filter(|&ti| t_scene[ti] > 0.0)
            .take(k)
            .collect()
    };

    // Exact-token selection is finished. A recorded backward reply or exact edge
    // promotes its earlier target to the rank position earned by `credit`, including
    // when `target` was previously sitting in the tail (`5..k`) at a lower score.
    if !final_turn_indices.is_empty() {
        let pair = 1.0 / 3.0f32.log2();
        let mut slot_score: Vec<f32> = final_turn_indices.iter().map(|&i| t_scene[i]).collect();
        let mut best_incoming: HashMap<usize, f32> = HashMap::new();
        for &ti in &final_turn_indices {
            let src = t_scene[ti];
            if src <= 0.0 {
                continue;
            }
            if let Some(ai) = ix.turns[ti].answers {
                let c = src * pair;
                let cur = best_incoming.entry(ai).or_insert(0.0);
                if c > *cur {
                    *cur = c;
                }
            }
            for &(degree, target) in &ix.turns[ti].exact_edges {
                let damp = 1.0 / (2.0 + degree.saturating_sub(1) as f32).log2();
                let c = src * damp;
                let cur = best_incoming.entry(target).or_insert(0.0);
                if c > *cur {
                    *cur = c;
                }
            }
            if !it.current_state {
                for &target in &ix.turns[ti].supersedes_turns {
                    let c = src * 0.66;
                    let cur = best_incoming.entry(target).or_insert(0.0);
                    if c > *cur {
                        *cur = c;
                    }
                }
            }
        }
        let mut incoming: Vec<(usize, f32)> = best_incoming.into_iter().collect();
        incoming.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        for (target, credit) in incoming {
            if credit <= 0.0 {
                continue;
            }
            if let Some(existing_pos) = final_turn_indices.iter().position(|&t| t == target) {
                if credit <= slot_score[existing_pos] {
                    continue;
                }
                let mut insert_at = existing_pos;
                for (i, &sc) in slot_score[..existing_pos].iter().enumerate() {
                    if credit > sc {
                        insert_at = i;
                        break;
                    }
                }
                if insert_at >= existing_pos {
                    slot_score[existing_pos] = credit;
                    if credit > t_scene[target] {
                        t_scene[target] = credit;
                    }
                    continue;
                }
                let sess = ix.turns[target].session;
                if insert_at < 5 && existing_pos >= 5 {
                    let in_top5 = final_turn_indices
                        .iter()
                        .take(5)
                        .filter(|&&t| ix.turns[t].session == sess)
                        .count();
                    if in_top5 >= max_per_session_top5 {
                        continue;
                    }
                }
                final_turn_indices.remove(existing_pos);
                slot_score.remove(existing_pos);
                final_turn_indices.insert(insert_at, target);
                slot_score.insert(insert_at, credit);
                if credit > t_scene[target] {
                    t_scene[target] = credit;
                }
                continue;
            }
            // An event list already spread one turn per date. A reply link must
            // not pull a second turn from a session that is already in the list.
            if it.is_event_sequence
                && final_turn_indices.iter().any(|&sel| ix.turns[sel].session == ix.turns[target].session)
            {
                continue;
            }
            let mut insert_at = slot_score.len();
            for (i, &sc) in slot_score.iter().enumerate() {
                if credit > sc {
                    insert_at = i;
                    break;
                }
            }
            if insert_at >= k {
                continue;
            }
            let sess = ix.turns[target].session;
            if insert_at < 5 {
                let in_top5 = final_turn_indices
                    .iter()
                    .take(5)
                    .filter(|&&t| ix.turns[t].session == sess)
                    .count();
                if in_top5 >= max_per_session_top5 {
                    continue;
                }
            }
            if credit > t_scene[target] {
                t_scene[target] = credit;
            }
            final_turn_indices.insert(insert_at, target);
            slot_score.insert(insert_at, credit);
            if final_turn_indices.len() > k {
                final_turn_indices.truncate(k);
                slot_score.truncate(k);
            }
        }
    }

    // ReplySource: an assistant reply at rank 1 is derived from the user turn it answers
    // (the turn immediately before it in the same session). When that prompt was retrieved
    // inside the top 5 already selected, the source takes slot 1 and the reply follows; top-5
    // membership never changes. Position and role
    // only. Skipped when the query asks what the assistant itself said.
    if !it.assistant_attribution && final_turn_indices.len() > 1 {
        let top = final_turn_indices[0];
        if ix.turns[top].speaker == "assistant" {
            let members = &ix.session_turns[ix.turns[top].session];
            if let Some(pos) = members.iter().position(|&m| m == top) {
                if pos > 0 {
                    let prev = members[pos - 1];
                    if ix.turns[prev].speaker == "user" {
                        if let Some(at) = final_turn_indices.iter().take(5).position(|&t| t == prev) {
                            final_turn_indices.remove(at);
                            final_turn_indices.insert(0, prev);
                        }
                    }
                }
            }
        }
    }
    // BoundaryPrimacy: runs after ReplySource, on the final list, so it shares one path with live recall.
    let q_cues = query_cues(ix, &q.text);
    boundary_primacy(ix, &q_cues, &mut final_turn_indices);
    // GivenNew runs after BoundaryPrimacy on the same final list (one path for live and bench).
    given_new(ix, &q_cues, &mut final_turn_indices);
    // SessionOpenerLift runs after GivenNew, on the same final list (one path for live and bench).
    session_opener_lift(ix, &q_cues, &mut final_turn_indices);
    // Hearsay runs after SessionOpenerLift, on the same final list (one path for live and bench).
    hearsay(ix, &q_cues, &mut final_turn_indices);
    // RSB: the retrieved turn is the symptom. Seat its structural cause
    // immediately after it. The cause can share no words with the query.
    // Never in front of the symptom. A full window does not grow.
    if !it.is_event_sequence {
        rsb_seat_behind(ix, &mut final_turn_indices, k);
    }
    // Rank-1 gatekeeper v2. Slot 1 stays the RSB v4 head. Slots 2..k are the
    // same Kemeny-Young order as PR 710, computed on that window with the head
    // removed. No new constant: the frozen slot is position 0 of this window.
    rank1_gatekeeper_keep_rsb_head(ix, &q.text, &mut final_turn_indices, &t_scene);
    // Steiner bridges. Terminals are the window the gatekeeper just froze,
    // including the RSB head. The tree may only pull a below-window turn into
    // slots 1..k-1. Slot 0 stays the head. Reordering the window cannot move
    // recall, so a bridge that is already inside it is not a seat.
    let steiner_bridge_fired = steiner_bridge::apply(ix, &mut final_turn_indices, k, &t_scene);
    let turns = final_turn_indices
        .into_iter()
        .map(|i| RankedItem {
            id: ix.turns[i].id.clone(),
            session_id: ix.session_ids[ix.turns[i].session].clone(),
            score: t_scene[i],
            matched: matched_words(&plan, t_mask[i]),
        })
        .collect();
    RankResult { intent: plan.intent, sessions, turns, standing_rules, steiner_bridge_fired }
}

/// Rank every query against the supplied sessions. Pure, deterministic, no I/O.
// =========================================================================
// RECALL-CUE SLOT-1 RULES (position, session, role field, token identity only)
// =========================================================================

/// BoundaryPrimacy window: the session opener must already sit inside the top 5 the kernel
/// selected. Fixed before any run; the top-5 membership never changes.
const PRIMACY_WINDOW: usize = 5;

/// Query cues: distinct kernel tokens of the query that occur somewhere in the haystack, minus
/// any token that occurs in a turn's role/speaker field (structural metadata, no word list).
fn query_cues(ix: &Index, text: &str) -> Vec<u32> {
    let mut ids: Vec<u32> = cue_tokens(text)
        .into_iter()
        .filter_map(|w| ix.cue_vocab.get(&w).copied())
        .filter(|id| !ix.role_cues.contains(id))
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// `cues(x)`: the query cues present on turn `ti` (both sides sorted, so a merge walk).
fn turn_query_cues(ix: &Index, q_cues: &[u32], ti: usize) -> Vec<u32> {
    let t = &ix.turn_cues[ti];
    let (mut i, mut j, mut out) = (0usize, 0usize, Vec::new());
    while i < t.len() && j < q_cues.len() {
        match t[i].cmp(&q_cues[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(t[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// `sup ⊇ sub` for sorted, deduplicated id lists.
fn covers_cues(sup: &[u32], sub: &[u32]) -> bool {
    let mut i = 0usize;
    for x in sub {
        while i < sup.len() && sup[i] < *x {
            i += 1;
        }
        if i >= sup.len() || sup[i] != *x {
            return false;
        }
        i += 1;
    }
    true
}

/// Position of turn `ti` inside its own session.
fn turn_pos(ix: &Index, ti: usize) -> Option<usize> {
    ix.session_turns[ix.turns[ti].session].iter().position(|&m| m == ti)
}

/// BoundaryPrimacy (event-boundary primacy; Zheng 2022, Ezzyat & Davachi 2011). If rank 1 is not
/// its session's opening turn, it matched at least one query cue, and that session's opener is
/// already inside the top `PRIMACY_WINDOW` and carries every cue rank 1 matched, the opener takes
/// slot 1 and the rest keep their order. Never crosses sessions; never changes the top-5 set.
fn boundary_primacy(ix: &Index, q_cues: &[u32], list: &mut Vec<usize>) -> bool {
    if list.len() < 2 {
        return false;
    }
    let top = list[0];
    if turn_pos(ix, top) == Some(0) {
        return false;
    }
    let top_cues = turn_query_cues(ix, q_cues, top);
    if top_cues.is_empty() {
        return false;
    }
    let sess = ix.turns[top].session;
    let Some(&opener) = ix.session_turns[sess].first() else { return false };
    let window = list.len().min(PRIMACY_WINDOW);
    let Some(at) = list[1..window].iter().position(|&t| t == opener).map(|i| i + 1) else { return false };
    if !covers_cues(&turn_query_cues(ix, q_cues, opener), &top_cues) {
        return false;
    }
    list.remove(at);
    list.insert(0, opener);
    true
}

/// GivenNew window: the earliest covering turn must already sit in the kernel's top 10.
/// Fixed before any run.
const GIVEN_NEW_WINDOW: usize = 10;
/// GivenNew variant, fixed before the run. `false` = card 2a (any role). `true` = card 2b: fires
/// only when rank 1's role field is the chat schema value `user` or `assistant` (structural field).
const GIVEN_NEW_CHAT_ROLES_ONLY: bool = true;

/// GivenNew (given/new information, Prince 1981; first attestation in lexicography). If rank 1
/// matched at least one query cue, take the earliest turn of rank 1's own session whose cues cover
/// every cue rank 1 matched. When that turn comes before rank 1 and is already inside the top
/// `GIVEN_NEW_WINDOW`, it takes slot 1 and the rest keep their order. Never crosses sessions.
fn given_new(ix: &Index, q_cues: &[u32], list: &mut Vec<usize>) -> bool {
    if list.len() < 2 {
        return false;
    }
    let top = list[0];
    if GIVEN_NEW_CHAT_ROLES_ONLY {
        let sp = ix.turns[top].speaker.as_str();
        if sp != "user" && sp != "assistant" {
            return false;
        }
    }
    let top_cues = turn_query_cues(ix, q_cues, top);
    if top_cues.is_empty() {
        return false;
    }
    for &z in &ix.session_turns[ix.turns[top].session] {
        if z == top {
            return false;
        }
        if covers_cues(&turn_query_cues(ix, q_cues, z), &top_cues) {
            let window = list.len().min(GIVEN_NEW_WINDOW);
            let Some(at) = list[1..window].iter().position(|&t| t == z).map(|i| i + 1) else { return false };
            list.remove(at);
            list.insert(0, z);
            return true;
        }
    }
    false
}

/// SessionOpenerLift window: the session opener must already sit inside the top 5 the kernel
/// selected. Fixed at 5 before any run; the top-5 membership never changes.
const SESSION_OPENER_LIFT_WINDOW: usize = 5;

/// SessionOpenerLift generalizes BoundaryPrimacy by dropping the same-session clause and
/// keeping the chat role gate. If rank 1 is a user or assistant turn, is not position 0 of
/// its session, and matched at least one query cue, take the first turn at ranks 2 through
/// `SESSION_OPENER_LIFT_WINDOW` that opens any session and whose cues cover every cue rank 1
/// matched. That opener takes slot 1 and the rest keep their order. Never changes the top-5 set.
fn session_opener_lift(ix: &Index, q_cues: &[u32], list: &mut Vec<usize>) -> bool {
    if list.len() < 2 {
        return false;
    }
    let top = list[0];
    let sp = ix.turns[top].speaker.as_str();
    if sp != "user" && sp != "assistant" {
        return false;
    }
    if turn_pos(ix, top) == Some(0) {
        return false;
    }
    let top_cues = turn_query_cues(ix, q_cues, top);
    if top_cues.is_empty() {
        return false;
    }
    let window = list.len().min(SESSION_OPENER_LIFT_WINDOW);
    for i in 1..window {
        let cand = list[i];
        if turn_pos(ix, cand) != Some(0) {
            continue;
        }
        if !covers_cues(&turn_query_cues(ix, q_cues, cand), &top_cues) {
            continue;
        }
        list.remove(i);
        list.insert(0, cand);
        return true;
    }
    false
}

/// Hearsay window: the covering user turn must already sit inside the top 5 the kernel
/// selected. Fixed at 5 before any run; the top-5 membership never changes.
const HEARSAY_WINDOW: usize = 5;

/// Hearsay (provenance / hearsay rule). If rank 1 is an assistant turn and matched at
/// least one query cue, take the first user turn at ranks 2 through `HEARSAY_WINDOW`,
/// from any session, whose cues cover every cue rank 1 matched. That user turn takes
/// slot 1 and the rest keep their order. Never changes the top-5 set.
fn hearsay(ix: &Index, q_cues: &[u32], list: &mut Vec<usize>) -> bool {
    if list.len() < 2 {
        return false;
    }
    let top = list[0];
    if ix.turns[top].speaker.as_str() != "assistant" {
        return false;
    }
    let top_cues = turn_query_cues(ix, q_cues, top);
    if top_cues.is_empty() {
        return false;
    }
    let window = list.len().min(HEARSAY_WINDOW);
    for i in 1..window {
        let cand = list[i];
        if ix.turns[cand].speaker.as_str() != "user" {
            continue;
        }
        if !covers_cues(&turn_query_cues(ix, q_cues, cand), &top_cues) {
            continue;
        }
        list.remove(i);
        list.insert(0, cand);
        return true;
    }
    false
}

/// Nearest preceding self-contained turn in the same session. A follow-up is an
/// assistant turn or a turn flagged anaphoric at ingest. The antecedent is the
/// cause; it need not share words with the follow-up.
fn rsb_antecedent(ix: &Index, ti: usize) -> Option<usize> {
    let t = &ix.turns[ti];
    if t.speaker != "assistant" && !t.anaphoric {
        return None;
    }
    let members = &ix.session_turns[t.session];
    let pos = members.iter().position(|&m| m == ti)?;
    for &prev in members[..pos].iter().rev() {
        let p = &ix.turns[prev];
        if p.field == 0 && !p.anaphoric {
            return Some(prev);
        }
    }
    None
}

/// One structural cause for a retrieved symptom: the antecedent if this turn is
/// a follow-up, otherwise the nearest earlier standing policy in the same episode.
fn rsb_cause(ix: &Index, ti: usize) -> Option<usize> {
    if let Some(ante) = rsb_antecedent(ix, ti) {
        return Some(ante);
    }
    ix.turns[ti].policy_backlinks.last().copied()
}

/// Seat each cause immediately after the symptom that points at it.
///
/// Version 4, fixed before the run. Version 3 inserted the cause and then
/// truncated the list to k, which drops whoever already sat in the last slot.
/// That drop is not a salience change. A cause that is not already in the
/// window is inserted only while the window is shorter than k, so an empty
/// slot is filled and nobody already inside is removed. A cause already in
/// the window is moved to sit immediately after its symptom, and is never
/// moved ahead of that symptom. A full window does not grow. k is the
/// caller's top_k. The backchannel skip on the antecedent walk is removed
/// so the walk matches the spec: nearest preceding field-0 turn that is
/// not anaphoric.
fn rsb_seat_behind(ix: &Index, list: &mut Vec<usize>, k: usize) {
    if list.is_empty() || k == 0 {
        return;
    }
    let snapshot = list.clone();
    let mut planned: Vec<(usize, usize)> = Vec::new();
    let mut reserved = snapshot.clone();
    for (i, &ti) in snapshot.iter().enumerate() {
        let Some(cause) = rsb_cause(ix, ti) else { continue };
        if planned.iter().any(|&(_, c)| c == cause) {
            continue;
        }
        let inside = snapshot.contains(&cause);
        if !inside && reserved.len() >= k {
            continue;
        }
        if !inside {
            reserved.push(cause);
        }
        planned.push((i, cause));
    }
    for (i, cause) in planned.into_iter().rev() {
        let Some(sym_at) = list.iter().position(|&t| t == snapshot[i]) else { continue };
        if let Some(cause_at) = list.iter().position(|&t| t == cause) {
            if cause_at <= sym_at {
                continue;
            }
            list.remove(cause_at);
            let sym_at = list.iter().position(|&t| t == snapshot[i]).unwrap();
            list.insert(sym_at + 1, cause);
        } else if list.len() < k {
            list.insert(sym_at + 1, cause);
        }
    }
}

/// Exact Kemeny-Young inside one Condorcet component. 8! = 40320 is the
/// largest search this branch-and-bound solves exactly. Larger components
/// use Copeland plus adjacent swaps. Fixed from the size of the search,
/// before any card.
const KEMENY_EXACT_CAP: usize = 8;

const WH_TIME: u8 = 1;
const WH_QTY: u8 = 2;
const WH_PERSON: u8 = 4;
const WH_PLACE: u8 = 8;
const WH_CAUSE: u8 = 16;

/// Closed class of English interrogative function words. Not a topical list.
fn wh_slots(query: &str) -> u8 {
    let lower = query.to_ascii_lowercase();
    let tokens: Vec<String> = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect();
    let mut slots = 0u8;
    let has = |w: &str| tokens.iter().any(|t| t == w);
    let phrase = |a: &str, b: &str| {
        tokens.windows(2).any(|p| p[0] == a && p[1] == b)
    };
    if phrase("what", "time") || phrase("which", "year") || has("when") {
        slots |= WH_TIME;
    }
    if phrase("how", "many") || phrase("how", "much") {
        slots |= WH_QTY;
    }
    if has("who") || has("whom") || has("whose") {
        slots |= WH_PERSON;
    }
    if has("where") {
        slots |= WH_PLACE;
    }
    if has("why") {
        slots |= WH_CAUSE;
    }
    slots
}

fn has_iso_or_hhmm(body: &str) -> bool {
    let b = body.as_bytes();
    if b.len() >= 10 {
        for i in 0..=b.len() - 10 {
            if b[i].is_ascii_digit()
                && b[i + 1].is_ascii_digit()
                && b[i + 2].is_ascii_digit()
                && b[i + 3].is_ascii_digit()
                && b[i + 4] == b'-'
                && b[i + 5].is_ascii_digit()
                && b[i + 6].is_ascii_digit()
                && b[i + 7] == b'-'
                && b[i + 8].is_ascii_digit()
                && b[i + 9].is_ascii_digit()
            {
                return true;
            }
        }
    }
    for i in 0..b.len() {
        if b[i] == b':'
            && i > 0
            && i + 2 < b.len()
            && b[i - 1].is_ascii_digit()
            && b[i + 1].is_ascii_digit()
            && b[i + 2].is_ascii_digit()
        {
            return true;
        }
    }
    false
}

fn has_ascii_digit(body: &str) -> bool {
    body.bytes().any(|c| c.is_ascii_digit())
}

fn has_handle_or_path(body: &str) -> bool {
    let b = body.as_bytes();
    for i in 0..b.len() {
        if b[i] == b'@' && i + 1 < b.len() && b[i + 1].is_ascii_alphanumeric() {
            return true;
        }
        if i + 2 < b.len() && b[i] == b':' && b[i + 1] == b'/' && b[i + 2] == b'/' {
            return true;
        }
    }
    body.split_whitespace().any(|tok| tok.contains('/') && tok.len() > 1)
}

/// A capitalized word of at least two letters that is not the first word of a sentence.
fn has_mid_sentence_name(body: &str) -> bool {
    let chars: Vec<char> = body.chars().collect();
    let mut sentence_start = true;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '.' || c == '!' || c == '?' || c == '\n' {
            sentence_start = true;
            i += 1;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_uppercase() {
            let start = i;
            while i < chars.len() && chars[i].is_alphabetic() {
                i += 1;
            }
            if !sentence_start && i - start >= 2 {
                return true;
            }
            sentence_start = false;
            continue;
        }
        sentence_start = false;
        i += 1;
    }
    false
}

fn wh_satisfied(ix: &Index, ti: usize, slots: u8, has_cause_edge: bool) -> bool {
    if slots == 0 {
        return true;
    }
    let turn = &ix.turns[ti];
    let ts = ix.session_ts.get(turn.session).copied().unwrap_or(0);
    if slots & WH_TIME != 0 && ts == 0 && !turn.form_clock {
        return false;
    }
    if slots & WH_QTY != 0 && !turn.form_digit {
        return false;
    }
    if slots & WH_PERSON != 0 && !turn.form_handle && !turn.form_midcap {
        return false;
    }
    if slots & WH_PLACE != 0 && !turn.form_midcap && !turn.form_handle {
        return false;
    }
    if slots & WH_CAUSE != 0 && !has_cause_edge {
        return false;
    }
    true
}

/// In-degree inside the pool from edges the index already stored, plus the
/// RSB antecedent. One pass over the log, not one pass per candidate.
fn pool_structural_degree(ix: &Index, pool: &[usize]) -> Vec<i64> {
    let mut at = vec![None; ix.turns.len()];
    for (i, &ti) in pool.iter().enumerate() {
        if let Some(slot) = at.get_mut(ti) {
            *slot = Some(i);
        }
    }
    let mut degree = vec![0i64; pool.len()];
    for (src, t) in ix.turns.iter().enumerate() {
        let mut hit = |target: usize| {
            if target == src {
                return;
            }
            if let Some(Some(slot)) = at.get(target) {
                degree[*slot] += 1;
            }
        };
        for &target in &t.policy_backlinks {
            hit(target);
        }
        for &target in &t.supersedes_turns {
            hit(target);
        }
        for &(_, target) in &t.exact_edges {
            hit(target);
        }
        if let Some(target) = rsb_antecedent(ix, src) {
            hit(target);
        }
    }
    degree
}

fn turn_is_superseded(ix: &Index, ti: usize) -> bool {
    ix.turns.iter().any(|t| t.supersedes_turns.contains(&ti))
}

fn kemeny_pair_score(order: &[usize], net: &[Vec<i64>]) -> i64 {
    let mut score = 0i64;
    for i in 0..order.len() {
        for j in (i + 1)..order.len() {
            score += net[order[i]][order[j]];
        }
    }
    score
}

fn kemeny_bound(prefix: &[usize], used: &[bool], net: &[Vec<i64>]) -> i64 {
    let m = used.len();
    let mut score = kemeny_pair_score(prefix, net);
    for &p in prefix {
        for c in 0..m {
            if !used[c] {
                score += net[p][c];
            }
        }
    }
    let rest: Vec<usize> = (0..m).filter(|&c| !used[c]).collect();
    for i in 0..rest.len() {
        for j in (i + 1)..rest.len() {
            score += net[rest[i]][rest[j]].abs();
        }
    }
    score
}

/// Exact Kemeny order of a component. Equal scores keep the RSB order,
/// which is the initial incumbent, so a tie never falls through to turn id.
fn kemeny_branch_and_bound(members: &[usize], net: &[Vec<i64>], rsb: &[usize]) -> Vec<usize> {
    let m = members.len();
    let mut local = vec![vec![0i64; m]; m];
    for i in 0..m {
        for j in 0..m {
            local[i][j] = net[members[i]][members[j]];
        }
    }
    let mut start: Vec<usize> = (0..m).collect();
    start.sort_by(|&a, &b| rsb[members[a]].cmp(&rsb[members[b]]));
    let key_of = |order: &[usize]| -> Vec<usize> {
        order.iter().map(|&i| rsb[members[i]]).collect()
    };
    let mut best_score = kemeny_pair_score(&start, &local);
    let mut best = start;
    let mut best_key = key_of(&best);
    let mut used = vec![false; m];
    let mut prefix: Vec<usize> = Vec::with_capacity(m);
    fn rec(
        local: &[Vec<i64>],
        members: &[usize],
        rsb: &[usize],
        used: &mut [bool],
        prefix: &mut Vec<usize>,
        best_score: &mut i64,
        best: &mut Vec<usize>,
        best_key: &mut Vec<usize>,
    ) {
        let m = used.len();
        if prefix.len() == m {
            let score = kemeny_pair_score(prefix, local);
            let key: Vec<usize> = prefix.iter().map(|&i| rsb[members[i]]).collect();
            if score > *best_score || (score == *best_score && key < *best_key) {
                *best_score = score;
                *best_key = key;
                *best = prefix.clone();
            }
            return;
        }
        if kemeny_bound(prefix, used, local) < *best_score {
            return;
        }
        for c in 0..m {
            if used[c] {
                continue;
            }
            used[c] = true;
            prefix.push(c);
            rec(local, members, rsb, used, prefix, best_score, best, best_key);
            prefix.pop();
            used[c] = false;
        }
    }
    rec(&local, members, rsb, &mut used, &mut prefix, &mut best_score, &mut best, &mut best_key);
    best.into_iter().map(|i| members[i]).collect()
}

fn kemeny_copeland(members: &[usize], net: &[Vec<i64>], rsb: &[usize]) -> Vec<usize> {
    let mut order = members.to_vec();
    order.sort_by(|&a, &b| {
        let sa: i64 = members.iter().map(|&c| (net[a][c] > 0) as i64 - (net[a][c] < 0) as i64).sum();
        let sb: i64 = members.iter().map(|&c| (net[b][c] > 0) as i64 - (net[b][c] < 0) as i64).sum();
        sb.cmp(&sa).then(rsb[a].cmp(&rsb[b]))
    });
    let mut improved = true;
    while improved {
        improved = false;
        for i in 0..order.len().saturating_sub(1) {
            let a = order[i];
            let b = order[i + 1];
            if net[a][b] < 0 {
                order.swap(i, i + 1);
                improved = true;
            }
        }
    }
    order
}

fn tarjan_sccs_best_first(net: &[Vec<i64>], rsb: &[usize]) -> Vec<Vec<usize>> {
    let n = net.len();
    let mut index = 0usize;
    let mut indices = vec![None; n];
    let mut low = vec![0usize; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut on = vec![false; n];
    let mut sccs: Vec<Vec<usize>> = Vec::new();
    fn strong(
        v: usize,
        net: &[Vec<i64>],
        index: &mut usize,
        indices: &mut [Option<usize>],
        low: &mut [usize],
        stack: &mut Vec<usize>,
        on: &mut [bool],
        sccs: &mut Vec<Vec<usize>>,
    ) {
        indices[v] = Some(*index);
        low[v] = *index;
        *index += 1;
        stack.push(v);
        on[v] = true;
        for w in 0..net.len() {
            if v != w && net[v][w] > 0 {
                if indices[w].is_none() {
                    strong(w, net, index, indices, low, stack, on, sccs);
                    low[v] = low[v].min(low[w]);
                } else if on[w] {
                    low[v] = low[v].min(indices[w].unwrap_or(0));
                }
            }
        }
        if low[v] == indices[v].unwrap_or(0) {
            let mut comp = Vec::new();
            while let Some(w) = stack.pop() {
                on[w] = false;
                comp.push(w);
                if w == v {
                    break;
                }
            }
            sccs.push(comp);
        }
    }
    for v in 0..n {
        if indices[v].is_none() {
            strong(v, net, &mut index, &mut indices, &mut low, &mut stack, &mut on, &mut sccs);
        }
    }
    let mut id_of = vec![0usize; n];
    for (i, comp) in sccs.iter().enumerate() {
        for &v in comp {
            id_of[v] = i;
        }
    }
    let ns = sccs.len();
    let mut indeg = vec![0usize; ns];
    let mut next: Vec<Vec<usize>> = vec![Vec::new(); ns];
    for a in 0..n {
        for b in 0..n {
            if a != b && net[a][b] > 0 && id_of[a] != id_of[b] {
                let (u, v) = (id_of[a], id_of[b]);
                if !next[u].contains(&v) {
                    next[u].push(v);
                    indeg[v] += 1;
                }
            }
        }
    }
    let anchor = |comp: &[usize]| comp.iter().map(|&v| rsb.get(v).copied().unwrap_or(usize::MAX)).min().unwrap_or(usize::MAX);
    let mut ready: Vec<usize> = (0..ns).filter(|&s| indeg[s] == 0).collect();
    let sort_ready = |ready: &mut Vec<usize>, sccs: &[Vec<usize>]| {
        ready.sort_by(|&a, &b| anchor(&sccs[b]).cmp(&anchor(&sccs[a])).then(b.cmp(&a)));
    };
    sort_ready(&mut ready, &sccs);
    let mut ordered = Vec::with_capacity(ns);
    while let Some(u) = ready.pop() {
        ordered.push(u);
        let mut outs = next[u].clone();
        outs.sort_unstable();
        for v in outs {
            indeg[v] -= 1;
            if indeg[v] == 0 {
                ready.push(v);
            }
        }
        sort_ready(&mut ready, &sccs);
    }
    if ordered.len() != ns {
        ordered = (0..ns).collect();
    }
    // Tarjan's component ids are not best-first. Kahn from sources of the
    // "beats" condensation puts undefeated components first.
    ordered.into_iter().map(|i| sccs[i].clone()).collect()
}

fn kemeny_young(net: &[Vec<i64>], rsb: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(net.len());
    for scc in tarjan_sccs_best_first(net, rsb) {
        if scc.len() <= 1 {
            out.extend(scc);
        } else if scc.len() <= KEMENY_EXACT_CAP {
            out.extend(kemeny_branch_and_bound(&scc, net, rsb));
        } else {
            out.extend(kemeny_copeland(&scc, net, rsb));
        }
    }
    out
}

/// Rank-1 gatekeeper. Runs after RSB. Reorders the RSB window. It does not
/// grow the window. The pool is that window, so its size is the caller's k.
fn rank1_gatekeeper(ix: &Index, query: &str, list: &mut Vec<usize>, t_scene: &[f32]) {
    let n = list.len();
    if n < 2 {
        return;
    }
    let pool = list.clone();
    let slots = wh_slots(query);
    let degree = pool_structural_degree(ix, &pool);
    let mut net = vec![vec![0i64; n]; n];
    for a in 0..n {
        for b in (a + 1)..n {
            let (ta, tb) = (pool[a], pool[b]);
            let mut margin = 0i64;
            let scene_a = t_scene.get(ta).copied().unwrap_or(0.0);
            let scene_b = t_scene.get(tb).copied().unwrap_or(0.0);
            match scene_a.partial_cmp(&scene_b).unwrap_or(std::cmp::Ordering::Equal) {
                std::cmp::Ordering::Greater => margin += 1,
                std::cmp::Ordering::Less => margin -= 1,
                std::cmp::Ordering::Equal => {}
            }
            match degree[a].cmp(&degree[b]) {
                std::cmp::Ordering::Greater => margin += 1,
                std::cmp::Ordering::Less => margin -= 1,
                std::cmp::Ordering::Equal => {}
            }
            let stab_a = ix.turns.get(ta).map(|t| t.stability).unwrap_or(0.0);
            let stab_b = ix.turns.get(tb).map(|t| t.stability).unwrap_or(0.0);
            match stab_a.partial_cmp(&stab_b).unwrap_or(std::cmp::Ordering::Equal) {
                std::cmp::Ordering::Greater => margin += 1,
                std::cmp::Ordering::Less => margin -= 1,
                std::cmp::Ordering::Equal => {}
            }
            if slots != 0 {
                let ok_a = wh_satisfied(ix, ta, slots, degree[a] > 0);
                let ok_b = wh_satisfied(ix, tb, slots, degree[b] > 0);
                if ok_a && !ok_b {
                    margin += 1;
                } else if ok_b && !ok_a {
                    margin -= 1;
                }
            }
            net[a][b] = margin;
            net[b][a] = -margin;
        }
    }
    let rsb: Vec<usize> = (0..n).collect();
    let mut order = kemeny_young(&net, &rsb);
    if order.len() != n {
        order = rsb.clone();
    }
    let mut turned: Vec<usize> = order.iter().map(|&i| pool[i]).collect();
    if turn_is_superseded(ix, turned[0]) {
        if let Some(pos) = turned.iter().position(|&ti| !turn_is_superseded(ix, ti)) {
            let keeper = turned.remove(pos);
            turned.insert(0, keeper);
        }
    }
    *list = turned;
}

/// Slot 1 is the RSB v4 head. The gatekeeper reorders only the rest.
/// ATMS, the four ballots, and the Copeland cap run on that tail exactly as
/// they do on a full window. The head is not a candidate for demotion.
fn rank1_gatekeeper_keep_rsb_head(ix: &Index, query: &str, list: &mut Vec<usize>, t_scene: &[f32]) {
    if list.len() < 2 {
        return;
    }
    let head = list[0];
    let mut tail = list[1..].to_vec();
    rank1_gatekeeper(ix, query, &mut tail, t_scene);
    list.clear();
    list.push(head);
    list.extend(tail);
}

pub fn rank(req: &RankRequest) -> RankResponse {

    let ix = build_index(&req.sessions, &req.params);
    let results: Vec<RankResult> = req.queries.iter().map(|q| rank_one(&ix, q, &req.params, req.top_k)).collect();
    let steiner_bridge_fires = results.iter().filter(|r| r.steiner_bridge_fired).count() as u64;
    RankResponse {
        engine: ENGINE,
        sessions_indexed: ix.session_ids.len(),
        turns_indexed: ix.turns.len(),
        results,
        steiner_bridge_fires,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(role: &str, content: &str) -> Turn {
        Turn { role: role.into(), content: content.into(), ..Turn::default() }
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
                    Turn { id: "D1:1".into(), role: "Caroline".into(), content: "I went hiking with my dog.".into(), ..Turn::default() },
                    Turn { id: "D1:2".into(), role: "Melanie".into(), content: "Are you still painting that sunrise landscape?".into(), ..Turn::default() },
                    Turn { id: "D1:3".into(), role: "Caroline".into(), content: "Yeah, I switched to acrylics for it yesterday.".into(), ..Turn::default() },
                    Turn { id: "D1:4".into(), role: "Melanie".into(), content: "My kids love the beach.".into(), ..Turn::default() },
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
                    Turn { id: "gold".into(), role: "user".into(), content: "My first sprint ends on March 29 for login features.".into(), ..Turn::default() },
                    Turn { id: "reply".into(), role: "assistant".into(), content: format!("Here is a sprint plan ending March 29: {filler_essay}"), ..Turn::default() },
                ]},
                Session { id: "s2".into(), date: String::new(), timestamp: 0, turns: vec![
                    Turn { id: "noise".into(), role: "assistant".into(), content: format!("Another unrelated deployment guide: {filler_essay}"), ..Turn::default() },
                ]},
                Session { id: "s3".into(), date: String::new(), timestamp: 0, turns: vec![
                    Turn { id: "noise2".into(), role: "user".into(), content: "Unrelated question about SQLite migrations.".into(), ..Turn::default() },
                    Turn { id: "noise3".into(), role: "assistant".into(), content: "Sure, here are SQLite migration commands.".into(), ..Turn::default() },
                    Turn { id: "noise4".into(), role: "user".into(), content: "Thanks!".into(), ..Turn::default() },
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

    #[test]
    fn one_chat_cannot_fill_the_top_five_when_another_fact_exists() {
        let walnut = "The walnut cabinet oil was mixed, brushed, and left to cure overnight.";
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "walnut".into(),
                    date: "2024/01/02 (Tue) 09:00".into(),
                    timestamp: 0,
                    turns: vec![
                        turn("user", walnut),
                        turn("user", "More walnut cabinet oil notes about the second coat and the brush."),
                        turn("user", "Still the walnut cabinet oil, now describing the rag and the cure."),
                        turn("user", "Again the walnut cabinet oil, with the tin and the cloth."),
                        turn("user", "Yet more walnut cabinet oil, including the shelf and the lid."),
                    ],
                },
                Session {
                    id: "maple".into(),
                    date: "2024/03/04 (Mon) 09:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "The maple drawer rails were cut, sanded, and fitted today.")],
                },
            ],
            queries: vec![Query {
                text: "What connects the walnut cabinet oil and the maple drawer rails?".into(),
                date: String::new(),
            }],
            top_k: 5,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(ranked.results[0].intent.multi_hop);
        let sessions: Vec<&str> = ranked.results[0].turns.iter().map(|t| t.session_id.as_str()).collect();
        assert!(sessions.contains(&"maple"), "second fact was buried in one chat: {sessions:?}");
        let walnut_n = sessions.iter().filter(|s| **s == "walnut").count();
        assert!(walnut_n <= 2, "one chat took {walnut_n} of the top 5: {sessions:?}");
    }

    #[test]
    fn pseudo_trigger_idioms_are_not_classified_as_standing_policies() {
        assert!(!utterance_is_standing_policy("Don't worry about the rain."));
        assert!(!utterance_is_standing_policy("Don't know where the wrench went."));
        assert!(!utterance_is_standing_policy("Don't think we tried the cedar stain."));
        assert!(!utterance_is_standing_policy("Never mind, I found the key."));
        assert!(!utterance_is_standing_policy("Never been to Lisbon."));
        assert!(!utterance_is_standing_policy("Always wanted to learn cello."));
        assert!(!utterance_is_standing_policy("Make sure I understand your diagram."));
        assert!(utterance_is_standing_policy("Please always quote file paths."));
        assert!(utterance_is_standing_policy("Make sure every migration has a rollback."));
    }

    #[test]
    fn backchannel_reaction_does_not_outrank_the_fact_turn() {
        let req = RankRequest {
            sessions: vec![Session {
                id: "pergola".into(),
                date: String::new(),
                timestamp: 1,
                turns: vec![
                    turn("nora", "I stained the cedar pergola beams with tung oil yesterday."),
                    turn("theo", "Wow, awesome!"),
                ],
            }],
            queries: vec![Query { text: "What oil did Nora use on the cedar pergola?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(ranked.results[0].turns[0].id.ends_with("#0"), "{:?}", ranked.results[0].turns);
    }

    #[test]
    fn ordinal_prepositional_and_bare_may_dates_parse_cleanly() {
        assert!(inline_date_tokens("How may I sand a drawer?").is_empty());
        let may_prep = inline_date_tokens("What did Nora build in May?");
        assert!(may_prep.contains(&"__MONTH__05".to_string()), "{may_prep:?}");
        let ord = inline_date_tokens("We met on the 15th of March before dusk.");
        assert!(ord.contains(&"__MD__03_15".to_string()), "{ord:?}");
        let full_ord = inline_date_tokens("Did the kiln fire on May 1st, 2023?");
        assert!(full_ord.contains(&"__DATE__2023_05_01".to_string()), "{full_ord:?}");
        let season = inline_date_tokens("Where did they camp in the summer of 2023?");
        assert!(season.contains(&"__SY__2023_SU".to_string()), "{season:?}");
    }

    #[test]
    fn single_token_entity_and_header_title_bridge_second_hop() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "hop1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "The Harbor Glassworks exhibition was curated by Mateo in Lisbon.")] },
                Session { id: "d1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Harbor Glassworks opened a small gift counter near the ferry.")] },
                Session { id: "d2".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Visitors praised the Harbor Glassworks lighting display.")] },
                Session { id: "d3".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Tickets for Harbor Glassworks sold out by noon.")] },
                Session { id: "d4".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Catalogs from Harbor Glassworks listed twelve bowls.")] },
                Session { id: "hop2".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Mateo (ceramist)\nMateo trained at the Porto Kiln Guild before curating glass.")] },
                Session { id: "noise".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Sourdough starter hydration depends on flour protein.")] },
            ],
            queries: vec![Query { text: "Where did the curator of the Harbor Glassworks exhibition train?".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        let top5: Vec<String> = rank(&req).results[0].sessions.iter().map(|s| s.id.clone()).collect();
        assert!(top5.contains(&"hop2".to_string()), "single-token proper bridge missed: {top5:?}");
    }

    #[test]
    fn unrelated_hobbies_do_not_flood_topical_preference_queries() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "h1".into(), date: String::new(), timestamp: 1, turns: vec![turn("user", "I love playing blitz chess on rainy evenings.")] },
                Session { id: "h2".into(), date: String::new(), timestamp: 2, turns: vec![turn("user", "I enjoy knitting wool scarves in winter.")] },
                Session { id: "h3".into(), date: String::new(), timestamp: 3, turns: vec![turn("user", "I love collecting vintage postage stamps.")] },
                Session { id: "h4".into(), date: String::new(), timestamp: 4, turns: vec![turn("user", "I prefer fountain pens with fine nibs.")] },
                Session { id: "h5".into(), date: String::new(), timestamp: 5, turns: vec![turn("user", "I would like to visit Patagonia someday.")] },
                Session { id: "gold".into(), date: String::new(), timestamp: 6, turns: vec![turn("user", "My camera bag holds a compact prime lens for street photography.")] },
            ],
            queries: vec![Query { text: "What would be a good lens to add to my camera kit?".into(), date: String::new() }],
            top_k: 3,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(ranked.results[0].intent.preference);
        assert_eq!(ranked.results[0].sessions[0].id, "gold");
    }

    #[test]
    fn structured_typed_slots_bind_and_supersede_on_current_state() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "old".into(), date: "2024-01-10".into(), timestamp: 0, turns: vec![turn("user", "Quota: 250gb\nInitial storage plan.")] },
                Session { id: "new".into(), date: "2024-08-15".into(), timestamp: 0, turns: vec![turn("user", "Quota: 500gb\nUpgraded storage plan.")] },
            ],
            queries: vec![Query { text: "What is my current storage quota?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert_eq!(ranked.results[0].sessions[0].id, "new");
    }

    #[test]
    fn tied_turns_are_identical_across_two_runs() {
        let mut sessions = Vec::new();
        for i in 0..12 {
            sessions.push(Session {
                id: format!("s{i}"),
                date: String::new(),
                timestamp: i,
                turns: vec![turn("user", "alpha beta gamma delta"), turn("assistant", "alpha beta gamma delta")],
            });
        }
        let req = RankRequest {
            sessions,
            queries: vec![Query { text: "alpha beta gamma delta".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        assert_eq!(serde_json::to_string(&rank(&req)).unwrap(), serde_json::to_string(&rank(&req)).unwrap());
    }

    #[test]
    fn exact_edge_replaces_a_weaker_slot_without_shared_words() {
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "now".into(),
                    date: String::new(),
                    timestamp: 2,
                    turns: vec![Turn {
                        id: "trigger".into(),
                        role: "user".into(),
                        content: "The ledger key is cedar.".into(),
                        links: vec![TurnLink { kind: "backfill".into(), to: "earlier".into(), degree: 2 }],
                        stability: 4.0,
                    }],
                },
                Session {
                    id: "then".into(),
                    date: String::new(),
                    timestamp: 1,
                    turns: vec![Turn {
                        id: "earlier".into(),
                        role: "user".into(),
                        content: "quartz violin under the stairs".into(),
                        stability: 9.0,
                        ..Turn::default()
                    }],
                },
                Session { id: "n1".into(), date: String::new(), timestamp: 3, turns: vec![turn("user", "marble compass on the table")] },
                Session { id: "n2".into(), date: String::new(), timestamp: 4, turns: vec![turn("user", "linen harbor in the fog")] },
                Session { id: "n3".into(), date: String::new(), timestamp: 5, turns: vec![turn("user", "pewter lantern by the door")] },
                Session { id: "n4".into(), date: String::new(), timestamp: 6, turns: vec![turn("user", "silk orchard past the gate")] },
            ],
            queries: vec![Query { text: "Where is the ledger key?".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        let ranked = rank(&req);
        let ids: Vec<String> = ranked.results[0].turns.iter().map(|t| t.id.clone()).collect();
        assert!(ids.iter().any(|id| id == "trigger"), "{ids:?}");
        assert!(ids.iter().any(|id| id == "earlier"), "exact edge missed the zero-overlap turn: {ids:?}");
    }

    #[test]
    fn a_backward_edge_lands_inside_the_first_five_not_the_tail() {
        let mut sessions = vec![
            Session {
                id: "now".into(),
                date: String::new(),
                timestamp: 20,
                turns: vec![Turn {
                    id: "trigger".into(),
                    role: "user".into(),
                    content: "The ledger key is cedar.".into(),
                    links: vec![TurnLink { kind: "backfill".into(), to: "earlier".into(), degree: 2 }],
                    ..Turn::default()
                }],
            },
            Session {
                id: "then".into(),
                date: String::new(),
                timestamp: 1,
                turns: vec![Turn {
                    id: "earlier".into(),
                    role: "user".into(),
                    content: "quartz violin under the stairs".into(),
                    ..Turn::default()
                }],
            },
        ];
        for i in 0..8 {
            sessions.push(Session {
                id: format!("f{i}"),
                date: String::new(),
                timestamp: 2 + i,
                turns: vec![turn("user", &format!("ledger {}", "padding ".repeat(40)))],
            });
        }
        let req = RankRequest {
            sessions,
            queries: vec![Query { text: "Where is the ledger key?".into(), date: String::new() }],
            top_k: 10,
            params: Params::default(),
        };
        let ranked = rank(&req);
        let rows: Vec<String> = ranked.results[0].turns.iter().map(|t| format!("{}:{:.3}", t.id, t.score)).collect();
        let pos = rows.iter().position(|row| row.starts_with("earlier:"));
        assert!(pos.is_some_and(|p| p < 5), "backward edge parked in the tail: {rows:?}");
    }

    #[test]
    fn rabin_karp_header_title_dictionary_bridges_numeric_and_multiword_titles() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "anchor".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "The 1928 Winter Games figure skating pair event was won in St. Moritz alongside the premiere of Symphony No. 9.")] },
                Session { id: "d1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Figure skating pair events in St. Moritz drew Alpine crowds.")] },
                Session { id: "d2".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "St. Moritz figure skating pair medals were awarded outdoors.")] },
                Session { id: "d3".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Winter figure skating pair routines included spins.")] },
                Session { id: "d4".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Judges scored the figure skating pair program in St. Moritz.")] },
                Session { id: "target".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Symphony No. 9 (piece)\nComposed in Vienna, this symphony features a choral finale.")] },
            ],
            queries: vec![Query { text: "In which city was the symphony premiered alongside the St. Moritz figure skating pair event composed?".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        let top5: Vec<String> = rank(&req).results[0].sessions.iter().map(|s| s.id.clone()).collect();
        assert!(top5.contains(&"target".to_string()), "Rabin-Karp header bridge missed: {top5:?}");
    }

    #[test]
    fn word_boundary_cues_never_misfire_on_substrings() {
        let army = classify("Which Army Academy did Amy attend in Miami's district?");
        assert!(!army.user_attribution, "Army/Amy/Miami's tripped user attribution: {army:?}");

        let server = classify("How does the server service preserve gifted packets?");
        assert!(!server.preference, "server/gifted tripped preference: {server:?}");

        let renew = classify("What did Matthew renew before curfew?");
        assert!(!renew.current_state, "renew tripped current_state: {renew:?}");

        let claim = "Did the bilateral candidate validate the $0 claim?";
        let bilateral = classify(claim);
        assert!(!bilateral.temporal, "bilateral/validate/$0 tripped temporal: {bilateral:?}");
        let padded = word_padded(claim);
        assert!(padded.contains("$0"), "currency amount was stripped: {padded}");

        assert!(is_content("one"), "Formula One lost the content token one");
        assert!(raw_tokens("Formula One").contains(&"one".to_string()));

        let ix = build_index(
            &[Session {
                id: "claim".into(),
                date: String::new(),
                timestamp: 1,
                turns: vec![
                    turn("user", "They claimed victory"),
                    turn("user", "Their aim was victory"),
                ],
            }],
            &Params::default(),
        );
        assert!(!ix.turns[0].has_causal, "aim substring inside claimed set has_causal");
        assert!(ix.turns[1].has_causal, "whole-word aim cue did not fire");
    }

    #[test]
    fn standing_rules_channel_fires_without_stealing_multihop_episodic_slots() {
        let episodic = [
            ("e1", "2024/01/03 (Wed) 09:00", "I logged an otter survey and counted 2 pups by the reed bank."),
            ("e2", "2024/02/11 (Sun) 09:00", "I logged an otter survey and counted 4 pups by the cove."),
            ("e3", "2024/03/19 (Tue) 09:00", "I logged an otter survey and counted 1 pup by the marsh."),
            ("e4", "2024/04/07 (Sun) 09:00", "I logged an otter survey and counted 6 pups by the pier."),
            ("e5", "2024/05/22 (Wed) 09:00", "I logged an otter survey and counted 3 pups by the inlet."),
        ];
        let mut sessions = Vec::new();
        for (id, date, content) in episodic {
            sessions.push(Session {
                id: id.into(),
                date: date.into(),
                timestamp: 0,
                turns: vec![turn("user", content)],
            });
        }
        sessions.push(Session {
            id: "rule".into(),
            date: "2023/12/01 (Fri) 09:00".into(),
            timestamp: 0,
            turns: vec![turn("user", "Always include the unit symbol when writing measurements.")],
        });
        let req = RankRequest {
            sessions,
            queries: vec![Query { text: "How many otter surveys did I log?".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        let ranked = rank(&req);
        let intent = &ranked.results[0].intent;
        assert!(intent.sniper_how_many && intent.multi_hop, "{intent:?}");
        let ids: Vec<&str> = ranked.results[0].sessions.iter().take(5).map(|s| s.id.as_str()).collect();
        for id in ["e1", "e2", "e3", "e4", "e5"] {
            assert!(ids.contains(&id), "episodic slot missing {id}: {ids:?}");
        }
        assert!(!ids.contains(&"rule"), "standing rule stole an episodic slot: {ids:?}");
        assert!(
            ranked.results[0].standing_rules.iter().any(|r| r.session_id == "rule"),
            "standing channel missed the zero-overlap rule: {:?}",
            ranked.results[0].standing_rules
        );
    }

    #[test]
    fn ingest_rsb_handle_and_contradiction_bridge_backward_only() {
        let path_words = "Glossary lists src auth token rs as separate syllables.";
        let noise = "Marble compass beside a cedar shelf.";
        let mut sessions = vec![
            Session {
                id: "S0".into(),
                date: String::new(),
                timestamp: 1_700_000_000,
                turns: vec![Turn {
                    id: "T0".into(),
                    role: "user".into(),
                    content: "Pinned legacy rotation window to 15m in src/auth/token.rs".into(),
                    ..Turn::default()
                }],
            },
        ];
        for (i, content) in [path_words, path_words, path_words, noise, noise].into_iter().enumerate() {
            sessions.push(Session {
                id: format!("F{}", i + 1),
                date: String::new(),
                timestamp: 1_700_000_000 + ((i as i64) + 1) * 200_000,
                turns: vec![turn("user", content)],
            });
        }
        sessions.push(Session {
            id: "S6".into(),
            date: String::new(),
            timestamp: 1_700_000_000 + 6 * 200_000,
            turns: vec![Turn {
                id: "T6".into(),
                role: "user".into(),
                content: "Production outage caused by expired JWT signature in src/auth/token.rs".into(),
                ..Turn::default()
            }],
        });
        sessions.push(Session {
            id: "S7".into(),
            date: String::new(),
            timestamp: 1_700_000_000 + 7 * 200_000,
            turns: vec![Turn {
                id: "T7".into(),
                role: "user".into(),
                content: "Quiet refactor landed beside src/auth/token.rs overnight.".into(),
                ..Turn::default()
            }],
        });
        assert_eq!(sessions.len(), 8);

        let ix = build_index(&sessions, &Params::default());
        assert!(
            ix.session_rsb_back[6].iter().any(|&(_, prev)| prev == 0),
            "later outage did not link backward to the earlier path: {:?}",
            ix.session_rsb_back
        );
        assert!(
            ix.session_rsb_back.iter().enumerate().all(|(later, edges)| {
                edges.iter().all(|&(_, earlier)| earlier < later)
            }),
            "RSB edge pointed forward: {:?}",
            ix.session_rsb_back
        );
        assert!(
            ix.session_supersedes_back.iter().enumerate().all(|(later, edges)| {
                edges.iter().all(|&earlier| earlier < later)
            }),
            "contradiction edge pointed forward: {:?}",
            ix.session_supersedes_back
        );
        assert!(ix.session_rsb_back[6].iter().all(|&(_, prev)| prev != 7));

        let req = RankRequest {
            sessions,
            queries: vec![Query {
                text: "What caused the production outage with expired JWT signature?".into(),
                date: String::new(),
            }],
            top_k: 8,
            params: Params::default(),
        };
        let ranked = rank(&req);
        let session_ids: Vec<&str> = ranked.results[0].sessions.iter().map(|s| s.id.as_str()).collect();
        assert!(session_ids.contains(&"S6"), "outage session missing: {session_ids:?}");
        assert!(session_ids.contains(&"S0"), "backward RSB missed the earlier path: {session_ids:?}");
        assert!(!session_ids.contains(&"S7"), "RSB pulled a later session forward: {session_ids:?}");
        let turn_ids: Vec<&str> = ranked.results[0].turns.iter().map(|t| t.id.as_str()).collect();
        assert!(turn_ids.contains(&"T0"), "backward RSB missed turn T0: {turn_ids:?}");
        assert!(!turn_ids.contains(&"T7"), "RSB pulled later turn T7 forward: {turn_ids:?}");
    }

    #[test]
    fn uncarried_scene_pooling_prevents_reaction_and_vocative_rank1_steal() {
        let req = RankRequest {
            sessions: vec![Session {
                id: "D1".into(),
                date: String::new(),
                timestamp: 1,
                turns: vec![
                    Turn {
                        id: "D1:1".into(),
                        role: "Caroline".into(),
                        content: "I adopted a rescue greyhound yesterday!".into(),
                        ..Turn::default()
                    },
                    Turn {
                        id: "D1:2".into(),
                        role: "Melanie".into(),
                        content: "Aww amazing Caroline! Why did you choose a greyhound?".into(),
                        ..Turn::default()
                    },
                ],
            }],
            queries: vec![Query { text: "What dog did Caroline adopt?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        let top = ranked.results[0].turns.first().map(|t| t.id.as_str());
        assert_eq!(top, Some("D1:1"), "vocative follow-up stole rank 1: {:?}", ranked.results[0].turns);
    }

    #[test]
    fn query_direct_header_title_match_boosts_canonical_subject_doc() {
        let bury = "coastal conservatory rostrum downbeat catalog notes";
        let mut sessions = vec![
            Session { id: "scott".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Scott Derrickson\nAmerican director based in Denver.")] },
            Session { id: "wolf".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Wolf Alice\nEnglish band formed in London.")] },
            Session { id: "ambroise".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Ambroise Thomas\nFrench composer who taught in Paris.")] },
            Session { id: "vcu".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Virginia Commonwealth University\nPublic campus in Richmond.")] },
            Session { id: "hayden".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Hayden (musician)\nCanadian singer from Thornhill.")] },
            Session { id: "time_doc".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Time\nWeekly magazine based in New York.")] },
            Session { id: "music_doc".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Music\nTheory of scales and intervals.")] },
        ];
        for i in 0..3 {
            sessions.push(Session {
                id: format!("coast{i}"),
                date: String::new(),
                timestamp: 0,
                turns: vec![turn("user", &format!("{bury} visit {i}"))],
            });
        }
        for i in 0..3 {
            sessions.push(Session {
                id: format!("rehearse{i}"),
                date: String::new(),
                timestamp: 0,
                turns: vec![turn("user", &format!("The rehearsal schedule lists the ensemble downbeat rostrum {i}"))],
            });
        }
        for i in 0..6 {
            sessions.push(Session {
                id: format!("dilute{i}"),
                date: String::new(),
                timestamp: 0,
                turns: vec![turn("user", "padding mentions time and music without a title line")],
            });
        }
        let req = RankRequest {
            sessions,
            queries: vec![
                Query {
                    text: "Where did Scott Derrickson record with Wolf Alice and Hayden after Ambroise Thomas left Virginia Commonwealth University?".into(),
                    date: String::new(),
                },
                Query {
                    text: "What time is the music rehearsal?".into(),
                    date: String::new(),
                },
            ],
            top_k: 5,
            params: Params::default(),
        };
        let ranked = rank(&req);
        let titles: Vec<String> = ranked.results[0].sessions.iter().map(|s| s.id.clone()).collect();
        for id in ["scott", "wolf", "ambroise", "vcu", "hayden"] {
            assert!(titles.contains(&id.to_string()), "canonical title missed: {id} in {titles:?}");
        }
        let common: Vec<String> = ranked.results[1].sessions.iter().map(|s| s.id.clone()).collect();
        assert!(!common.contains(&"time_doc".to_string()), "lowercase time bridged: {common:?}");
        assert!(!common.contains(&"music_doc".to_string()), "lowercase music bridged: {common:?}");
    }

    #[test]
    fn conditional_when_and_scope_matched_standing_rule_win_episodic_slot() {
        let q1 = classify("What are some common responses when something goes wrong with an API?");
        assert!(!q1.temporal && !q1.sniper_when && q1.task_under_rules, "{q1:?}");
        let q2 = classify("When building an application that communicates with a REST API, what typical errors should I be prepared to handle?");
        assert!(!q2.temporal && !q2.sniper_when && q2.task_under_rules, "{q2:?}");
        let q3 = classify("When does my first sprint end?");
        assert!(q3.temporal && q3.sniper_when && !q3.task_under_rules, "{q3:?}");

        let req = RankRequest {
            sessions: vec![
                Session { id: "newer-unit".into(), date: String::new(), timestamp: 400, turns: vec![turn("user", "Always include the unit symbol.")] },
                Session { id: "newer-tabs".into(), date: String::new(), timestamp: 300, turns: vec![turn("user", "Never use tabs in this repo.")] },
                Session { id: "scope-soup".into(), date: String::new(), timestamp: 1, turns: vec![turn("user", "Always keep a lid when labeling soup.")] },
                Session { id: "essay".into(), date: String::new(), timestamp: 50, turns: vec![turn("user", "Store leftover soup.")] },
            ],
            queries: vec![Query { text: "How should I store leftover soup?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        let sessions: Vec<&str> = ranked.results[0].sessions.iter().map(|s| s.id.as_str()).collect();
        assert!(sessions.contains(&"scope-soup"), "spec winner missed episodic slot: {sessions:?}");
        assert!(!sessions.contains(&"newer-unit"), "newer unmatched rule stole slot: {sessions:?}");
    }

    #[test]
    fn beam_timeline_routes_to_user_and_ssa_collective_cues_stay_assistant() {
        let order = classify("Can you list the order in which I brought up different aspects of developing my personal budget tracker throughout our conversations?");
        assert!(order.is_event_sequence && order.user_attribution && !order.assistant_attribution && !order.task_under_rules, "{order:?}");
        let ssa = classify("I think we discussed work from home jobs for seniors earlier. Can you remind me what was the 7th job in the list you provided?");
        assert!(ssa.assistant_attribution && !ssa.user_attribution, "{ssa:?}");
    }

    #[test]
    fn weapon4_versions_orthography_and_abbreviations() {
        assert_eq!(raw_tokens("V8 engine"), vec!["v8".to_string(), "engine".to_string()]);
        assert_eq!(raw_tokens("Bootstrap v5.3.0"), raw_tokens("Bootstrap 5.3.0"));
        assert_eq!(raw_tokens("enable 2FA"), raw_tokens("enable two-factor"));
        assert_eq!(raw_tokens("$1,200"), vec!["1200".to_string()]);
        assert_eq!(stem("html5"), stem("html"));
        assert_eq!(stem("css3"), stem("css"));
        assert_eq!(stem("colour"), stem("color"));
        assert_eq!(stem("theatre"), stem("theater"));
        assert_eq!(stem("organise"), stem("organize"));
        assert_eq!(stem("organises"), stem("organizes"));
        assert_eq!(stem("irises"), stem("iris"));
        assert_eq!(stem("edu"), stem("education"));
        assert_eq!(stem("fam"), stem("family"));
        assert_eq!(stem("pic"), stem("picture"));
        assert_eq!(stem("career"), "career");
    }

    #[test]
    fn multi_sentence_question_echo_and_speaker_nicknames_resolve_cleanly() {
        let req = RankRequest {
            sessions: vec![Session {
                id: "D3".into(),
                date: String::new(),
                timestamp: 1,
                turns: vec![
                    Turn {
                        id: "D3:15".into(),
                        role: "Caroline".into(),
                        content: "Wow, what an amazing family pic! How long have you been married?".into(),
                        ..Turn::default()
                    },
                    Turn {
                        id: "D3:16".into(),
                        role: "Melanie".into(),
                        content: "5 years already! Time flies.".into(),
                        ..Turn::default()
                    },
                ],
            }],
            queries: vec![Query { text: "How long have Mel and her husband been married?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert_eq!(ranked.results[0].turns[0].id, "D3:16", "{:?}", ranked.results[0].turns);
    }

    #[test]
    fn engineering_choice_beats_progress_narrative_and_assistant_expansion() {
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "debug".into(),
                    date: String::new(),
                    timestamp: 4,
                    turns: vec![turn("user", "I've been debugging how to set up and keep the API cache responses simple and lightweight. I've been working on the API cache responses.")],
                },
                Session {
                    id: "choice".into(),
                    date: String::new(),
                    timestamp: 5,
                    turns: vec![turn("user", "I'd prefer to set up the API cache responses by trying to keep it lightweight and simple, without compromising latency.")],
                },
            ],
            queries: vec![Query {
                text: "How should I set up the API cache responses? I'd like to keep it simple and lightweight.".into(),
                date: String::new(),
            }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(ranked.results[0].intent.preference);
        assert_eq!(ranked.results[0].sessions[0].id, "choice");
        assert_eq!(ranked.results[0].turns[0].session_id, "choice");
    }
}

#[cfg(test)]
mod recall_cue_tests {
    use super::*;

    fn t(id: &str, role: &str, content: &str) -> Turn {
        Turn { id: id.into(), role: role.into(), content: content.into(), ..Turn::default() }
    }
    fn s(id: &str, turns: Vec<Turn>) -> Session {
        Session { id: id.into(), date: String::new(), timestamp: 0, turns }
    }
    fn ix_of(sessions: &[Session]) -> Index {
        build_index(sessions, &Params::default())
    }
    fn idx(ix: &Index, id: &str) -> usize {
        ix.turns.iter().position(|x| x.id == id).unwrap()
    }
    fn ids(ix: &Index, list: &[usize]) -> Vec<String> {
        list.iter().map(|&i| ix.turns[i].id.clone()).collect()
    }
    fn haystack() -> Vec<Session> {
        vec![
            s("a", vec![
                t("a0", "user", "My new camera is a Sony A7 and I shoot street photos."),
                t("a1", "assistant", "Nice choice for street work."),
                t("a2", "user", "The Sony camera battery dies fast though."),
            ]),
            s("b", vec![
                t("b0", "user", "Let's talk about gardening tomatoes."),
                t("b1", "user", "My Sony camera takes great tomato shots."),
            ]),
        ]
    }

    #[test]
    fn primacy_moves_opener_that_covers_rank1_cues() {
        let ix = ix_of(&haystack());
        let qc = query_cues(&ix, "Which Sony camera do I have?");
        let mut list = vec![idx(&ix, "a2"), idx(&ix, "a1"), idx(&ix, "a0")];
        assert!(boundary_primacy(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["a0", "a2", "a1"]);
    }

    #[test]
    fn primacy_holds_when_opener_lacks_a_rank1_cue() {
        let ix = ix_of(&haystack());
        // rank 1 (a2) matches "battery"; the opener a0 does not carry it.
        let qc = query_cues(&ix, "How is the Sony camera battery?");
        let mut list = vec![idx(&ix, "a2"), idx(&ix, "a0"), idx(&ix, "a1")];
        assert!(!boundary_primacy(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["a2", "a0", "a1"]);
    }

    #[test]
    fn primacy_never_crosses_sessions() {
        let ix = ix_of(&haystack());
        let qc = query_cues(&ix, "Which Sony camera do I have?");
        // rank 1 is in session b; a0 is an opener that covers, but of another session.
        let mut list = vec![idx(&ix, "b1"), idx(&ix, "a0"), idx(&ix, "a2")];
        assert!(!boundary_primacy(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["b1", "a0", "a2"]);
    }

    #[test]
    fn primacy_respects_top5_window() {
        let ix = ix_of(&haystack());
        let qc = query_cues(&ix, "Which Sony camera do I have?");
        let (a0, a1, a2, b0, b1) = (idx(&ix, "a0"), idx(&ix, "a1"), idx(&ix, "a2"), idx(&ix, "b0"), idx(&ix, "b1"));
        // a0 sits at slot 6, outside the window (a1 repeated only to pad the list).
        let mut list = vec![a2, a1, b0, b1, a1, a0];
        let before = list.clone();
        assert!(!boundary_primacy(&ix, &qc, &mut list));
        assert_eq!(list, before);
    }

    #[test]
    fn speaker_name_tokens_are_not_cues() {
        let ix = ix_of(&[s("d", vec![
            t("d0", "Caroline", "Hey, I went hiking at the lake yesterday."),
            t("d1", "Melanie", "Caroline that sounds fun, Caroline loves the lake."),
        ])]);
        let qc = query_cues(&ix, "What did Caroline do at the lake?");
        let caroline = ix.cue_vocab.get(&cue_tokens("Caroline")[0]).copied().unwrap();
        assert!(!qc.contains(&caroline));
        // Without the name token, the opener d0 covers d1's cue set and moves up.
        let mut list = vec![idx(&ix, "d1"), idx(&ix, "d0")];
        assert!(boundary_primacy(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["d0", "d1"]);
    }

    #[test]
    fn primacy_is_deterministic_under_shuffled_sessions() {
        let h = haystack();
        let mut rev = h.clone();
        rev.reverse();
        for hs in [h, rev] {
            let ix = ix_of(&hs);
            let qc = query_cues(&ix, "Which Sony camera do I have?");
            let mut list = vec![idx(&ix, "a2"), idx(&ix, "b1"), idx(&ix, "a0")];
            boundary_primacy(&ix, &qc, &mut list);
            assert_eq!(ids(&ix, &list), vec!["a0", "a2", "b1"]);
        }
    }

    fn chat() -> Vec<Session> {
        vec![
            s("c", vec![
                t("c0", "user", "Morning! Planning a trip."),
                t("c1", "user", "I booked a flight to Lisbon for March."),
                t("c2", "assistant", "Great, Lisbon in March is lovely."),
                t("c3", "user", "Remind me later about the Lisbon flight in March."),
            ]),
            s("e", vec![t("e0", "user", "Lisbon flight prices in March are high.")]),
        ]
    }

    #[test]
    fn given_new_picks_earliest_covering_turn() {
        let ix = ix_of(&chat());
        let qc = query_cues(&ix, "When is my Lisbon flight?");
        let (c1, c2, c3) = (idx(&ix, "c1"), idx(&ix, "c2"), idx(&ix, "c3"));
        let mut list = vec![c3, c2, c1];
        assert!(given_new(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["c1", "c3", "c2"]);
    }

    #[test]
    fn given_new_does_not_reach_past_top10() {
        let ix = ix_of(&chat());
        let qc = query_cues(&ix, "When is my Lisbon flight?");
        let (c0, c1, c2, c3, e0) = (idx(&ix, "c0"), idx(&ix, "c1"), idx(&ix, "c2"), idx(&ix, "c3"), idx(&ix, "e0"));
        // c1 at slot 11 (padding repeats only fill the list).
        let mut list = vec![c3, c2, c0, e0, c2, c0, e0, c2, c0, e0, c1];
        let before = list.clone();
        assert!(!given_new(&ix, &qc, &mut list));
        assert_eq!(list, before);
    }

    #[test]
    fn given_new_never_crosses_sessions() {
        let ix = ix_of(&chat());
        let qc = query_cues(&ix, "When is my Lisbon flight?");
        // rank 1 is e0 (session e, position 0). c1 covers its cues but lives in session c.
        let mut list = vec![idx(&ix, "e0"), idx(&ix, "c1")];
        assert!(!given_new(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["e0", "c1"]);
    }

    #[test]
    fn given_new_is_deterministic_under_shuffled_sessions() {
        let h = chat();
        let mut rev = h.clone();
        rev.reverse();
        for hs in [h, rev] {
            let ix = ix_of(&hs);
            let qc = query_cues(&ix, "When is my Lisbon flight?");
            let mut list = vec![idx(&ix, "c3"), idx(&ix, "e0"), idx(&ix, "c1")];
            given_new(&ix, &qc, &mut list);
            assert_eq!(ids(&ix, &list), vec!["c1", "c3", "e0"]);
        }
    }

    #[test]
    fn given_new_role_gate_matches_variant() {
        let ix = ix_of(&[s("n", vec![
            t("n0", "Caroline", "I adopted a dog named Biscuit."),
            t("n1", "Melanie", "Biscuit the dog is adorable."),
        ])]);
        let qc = query_cues(&ix, "Biscuit dog?");
        let mut list = vec![idx(&ix, "n1"), idx(&ix, "n0")];
        let moved = given_new(&ix, &qc, &mut list);
        assert_eq!(moved, !GIVEN_NEW_CHAT_ROLES_ONLY);
    }

    fn haystack_two_openers() -> Vec<Session> {
        vec![
            s("a", vec![
                t("a0", "user", "My new camera is a Sony A7 and I shoot street photos."),
                t("a1", "assistant", "Nice choice for street work."),
                t("a2", "user", "The Sony camera battery dies fast though."),
            ]),
            s("b", vec![
                t("b0", "user", "Let's talk about gardening tomatoes."),
                t("b1", "user", "My Sony camera takes great tomato shots."),
            ]),
            s("c", vec![
                t("c0", "user", "The Sony camera I keep in the bag is an A7."),
            ]),
        ]
    }

    #[test]
    fn session_opener_lift_moves_cross_session_opener() {
        let ix = ix_of(&haystack());
        let qc = query_cues(&ix, "Which Sony camera do I have?");
        // rank 1 is b1, not an opener. a0 opens a different session, sits in the window, and covers.
        let mut list = vec![idx(&ix, "b1"), idx(&ix, "a0"), idx(&ix, "a2")];
        assert!(session_opener_lift(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["a0", "b1", "a2"]);
    }

    #[test]
    fn session_opener_lift_holds_when_opener_lacks_a_rank1_cue() {
        let ix = ix_of(&haystack());
        // rank 1 (a2) matches "battery"; openers a0 and b0 do not carry it.
        let qc = query_cues(&ix, "How is the Sony camera battery?");
        let mut list = vec![idx(&ix, "a2"), idx(&ix, "a0"), idx(&ix, "b0")];
        assert!(!session_opener_lift(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["a2", "a0", "b0"]);
    }

    #[test]
    fn session_opener_lift_role_gate() {
        let ix = ix_of(&[s("n", vec![
            t("n0", "Caroline", "I adopted a dog named Biscuit."),
            t("n1", "Melanie", "Biscuit the dog is adorable."),
        ]), s("p", vec![
            t("p0", "user", "Biscuit the dog lives here."),
        ])]);
        let qc = query_cues(&ix, "Biscuit dog?");
        let mut list = vec![idx(&ix, "n1"), idx(&ix, "p0")];
        assert!(!session_opener_lift(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["n1", "p0"]);
    }

    #[test]
    fn session_opener_lift_skips_when_rank1_is_an_opener() {
        let ix = ix_of(&haystack());
        let qc = query_cues(&ix, "Which Sony camera do I have?");
        let mut list = vec![idx(&ix, "a0"), idx(&ix, "b1")];
        assert!(!session_opener_lift(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["a0", "b1"]);
    }

    #[test]
    fn session_opener_lift_respects_top5_window() {
        let ix = ix_of(&haystack());
        let qc = query_cues(&ix, "Which Sony camera do I have?");
        let (a0, a1, a2, b0, b1) = (idx(&ix, "a0"), idx(&ix, "a1"), idx(&ix, "a2"), idx(&ix, "b0"), idx(&ix, "b1"));
        // a0 sits at slot 6, outside the window (a1 repeated only to pad the list).
        let mut list = vec![b1, a1, a2, b0, a1, a0];
        let before = list.clone();
        assert!(!session_opener_lift(&ix, &qc, &mut list));
        assert_eq!(list, before);
    }

    #[test]
    fn session_opener_lift_picks_first_covering_opener_in_rank_order() {
        let ix = ix_of(&haystack_two_openers());
        let qc = query_cues(&ix, "Which Sony camera do I have?");
        let (b1, a0, c0) = (idx(&ix, "b1"), idx(&ix, "a0"), idx(&ix, "c0"));
        // a0 and c0 both open a session and cover b1's cues; the earlier rank wins.
        let mut list = vec![b1, a0, c0];
        assert!(session_opener_lift(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["a0", "b1", "c0"]);
        let mut list = vec![b1, c0, a0];
        assert!(session_opener_lift(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["c0", "b1", "a0"]);
    }

    #[test]
    fn session_opener_lift_is_deterministic_under_shuffled_sessions() {
        let h = haystack();
        let mut rev = h.clone();
        rev.reverse();
        for hs in [h, rev] {
            let ix = ix_of(&hs);
            let qc = query_cues(&ix, "Which Sony camera do I have?");
            let mut list = vec![idx(&ix, "b1"), idx(&ix, "a2"), idx(&ix, "a0")];
            session_opener_lift(&ix, &qc, &mut list);
            assert_eq!(ids(&ix, &list), vec!["a0", "b1", "a2"]);
        }
    }

    fn hearsay_haystack() -> Vec<Session> {
        vec![
            s("a", vec![
                t("a0", "user", "I booked a flight to Lisbon for March."),
                t("a1", "assistant", "Lisbon March looks great."),
                t("a2", "assistant", "I can also look at hotels later."),
            ]),
            s("b", vec![
                t("b0", "user", "Lisbon March works on my side too."),
                t("b1", "assistant", "Noted for the Lisbon March trip."),
            ]),
            s("c", vec![
                t("c0", "user", "Lisbon March is already on the calendar."),
            ]),
        ]
    }

    #[test]
    fn hearsay_moves_covering_user_turn_from_any_session() {
        let ix = ix_of(&hearsay_haystack());
        let qc = query_cues(&ix, "When is the Lisbon trip in March?");
        // rank 1 is assistant a1. b0 is a user turn in another session, sits in the window, and covers.
        let mut list = vec![idx(&ix, "a1"), idx(&ix, "b0"), idx(&ix, "a0")];
        assert!(hearsay(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["b0", "a1", "a0"]);
    }

    #[test]
    fn hearsay_holds_when_user_turn_lacks_a_rank1_cue() {
        let ix = ix_of(&hearsay_haystack());
        // a1 matches Lisbon March; a0 also covers those, but b0-less candidate: use a2 as rank 1
        // with a cue ("hotels") that no user turn carries.
        let qc = query_cues(&ix, "Can you look at hotels in Lisbon in March?");
        let mut list = vec![idx(&ix, "a2"), idx(&ix, "b0"), idx(&ix, "a0")];
        assert!(!hearsay(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["a2", "b0", "a0"]);
    }

    #[test]
    fn hearsay_skips_when_rank1_is_not_assistant() {
        let ix = ix_of(&hearsay_haystack());
        let qc = query_cues(&ix, "When is the Lisbon trip in March?");
        let mut list = vec![idx(&ix, "a0"), idx(&ix, "b0")];
        assert!(!hearsay(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["a0", "b0"]);
    }

    #[test]
    fn hearsay_skips_non_user_candidates() {
        let ix = ix_of(&hearsay_haystack());
        let qc = query_cues(&ix, "When is the Lisbon trip in March?");
        // b1 is an assistant turn that covers; it must not move.
        let mut list = vec![idx(&ix, "a1"), idx(&ix, "b1"), idx(&ix, "a2")];
        assert!(!hearsay(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["a1", "b1", "a2"]);
    }

    #[test]
    fn hearsay_respects_top5_window() {
        let ix = ix_of(&hearsay_haystack());
        let qc = query_cues(&ix, "When is the Lisbon trip in March?");
        let (a1, a2, b1, b0) = (idx(&ix, "a1"), idx(&ix, "a2"), idx(&ix, "b1"), idx(&ix, "b0"));
        // b0 sits at slot 6, outside the window (a2/b1 repeated only to pad the list).
        let mut list = vec![a1, a2, b1, a2, b1, b0];
        let before = list.clone();
        assert!(!hearsay(&ix, &qc, &mut list));
        assert_eq!(list, before);
    }

    #[test]
    fn hearsay_picks_first_covering_user_in_rank_order() {
        let ix = ix_of(&hearsay_haystack());
        let qc = query_cues(&ix, "When is the Lisbon trip in March?");
        let (a1, b0, c0) = (idx(&ix, "a1"), idx(&ix, "b0"), idx(&ix, "c0"));
        let mut list = vec![a1, b0, c0];
        assert!(hearsay(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["b0", "a1", "c0"]);
        let mut list = vec![a1, c0, b0];
        assert!(hearsay(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["c0", "a1", "b0"]);
    }

    #[test]
    fn hearsay_is_deterministic_under_shuffled_sessions() {
        let h = hearsay_haystack();
        let mut rev = h.clone();
        rev.reverse();
        for hs in [h, rev] {
            let ix = ix_of(&hs);
            let qc = query_cues(&ix, "When is the Lisbon trip in March?");
            let mut list = vec![idx(&ix, "a1"), idx(&ix, "a2"), idx(&ix, "b0")];
            hearsay(&ix, &qc, &mut list);
            assert_eq!(ids(&ix, &list), vec!["b0", "a1", "a2"]);
        }
    }

    #[test]
    fn salience_join_writes_a_backward_edge_from_a_later_turn_to_an_earlier_policy() {
        let ix = ix_of(&[s("s", vec![
            t("rule", "user", "Always write the answer in markdown tables."),
            t("later", "user", "Please summarize today's notes."),
        ])]);
        let rule = idx(&ix, "rule");
        let later = idx(&ix, "later");
        assert!(ix.turns[rule].is_standing_policy, "the earlier turn should be a standing policy");
        assert!(
            ix.turns[later].policy_backlinks.contains(&rule),
            "SalienceJoin should record a backward later->policy link, got {:?}",
            ix.turns[later].policy_backlinks
        );
        assert!(
            !ix.turns[rule].policy_backlinks.contains(&later),
            "must not write a forward link"
        );
        assert!(
            !ix.turns[later].exact_edges.iter().any(|&(_, target)| target == rule),
            "v2 must not put the policy link on the general exact-edge list"
        );
    }

    #[test]
    fn salience_join_seats_a_fired_policy_when_a_later_turn_is_retrieved() {
        let sessions = vec![
            s("conv_s0", vec![
                t("rule", "user", "Always write the answer in markdown tables."),
                t("ack", "assistant", "I will follow that."),
            ]),
            s("conv_s1", vec![
                t("later", "user", "Could you outline a backup routine?"),
                t("reply", "assistant", "Here is the outline."),
            ]),
        ];
        let req = RankRequest {
            sessions,
            queries: vec![Query { text: "Could you outline a backup routine?".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(ranked.results[0].intent.task_under_rules);
        let ids: Vec<String> = ranked.results[0].turns.iter().map(|x| x.id.clone()).collect();
        let later_at = ids.iter().position(|id| id == "later").expect("symptom retrieved");
        let rule_at = ids.iter().position(|id| id == "rule").expect(&format!("cause missing, got {ids:?}"));
        assert!(rule_at > later_at, "cause sits behind the symptom, got {ids:?}");
    }

    #[test]
    fn salience_join_raises_a_policy_that_shares_no_words_with_the_query() {
        let sessions = vec![
            s("conv_s0", vec![
                t("rule", "user", "Always write the answer in markdown tables."),
            ]),
            s("conv_s1", vec![
                t("later", "user", "When did the soup boil?"),
            ]),
        ];
        let req = RankRequest {
            sessions,
            queries: vec![Query { text: "When did the soup boil?".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(!ranked.results[0].intent.task_under_rules);
        let ids: Vec<String> = ranked.results[0].turns.iter().map(|x| x.id.clone()).collect();
        let later_at = ids.iter().position(|id| id == "later").expect("symptom retrieved");
        let rule_at = ids.iter().position(|id| id == "rule").expect(&format!("zero-overlap cause missing, got {ids:?}"));
        assert!(rule_at > later_at, "cause sits behind the symptom, got {ids:?}");
    }

    #[test]
    fn salience_join_does_not_seat_a_policy_from_another_episode_on_a_temporal_question() {
        let sessions = vec![
            s("batch_a", vec![
                t("gold", "user", "I reached 1200 in the emergency fund on June 5."),
                t("reply", "assistant", "Reaching 1200 in the emergency fund on June 5 is a milestone."),
            ]),
            s("batch_b", vec![
                t("rule", "user", "Always write the answer in markdown tables."),
            ]),
        ];
        let req = RankRequest {
            sessions,
            queries: vec![Query {
                text: "How long after June 5 did the emergency fund get to 2000?".into(),
                date: String::new(),
            }],
            top_k: 5,
            params: Params::default(),
        };
        let ranked = rank(&req);
        let ids: Vec<String> = ranked.results[0].turns.iter().map(|x| x.id.clone()).collect();
        assert!(
            !ids.contains(&"rule".to_string()),
            "a temporal question must not import a policy from another episode, got {ids:?}"
        );
    }

    #[test]
    fn kemeny_puts_the_condorcet_winner_first() {
        let net = vec![
            vec![0, 1, 1],
            vec![-1, 0, 1],
            vec![-1, -1, 0],
        ];
        assert_eq!(kemeny_young(&net, &[0, 1, 2]), vec![0, 1, 2]);
    }

    #[test]
    fn kemeny_branch_and_bound_matches_brute_force() {
        let net = vec![
            vec![0, 2, -1, 1],
            vec![-2, 0, 3, -1],
            vec![1, -3, 0, 2],
            vec![-1, 1, -2, 0],
        ];
        let rsb = [0, 1, 2, 3];
        let solved = kemeny_branch_and_bound(&[0, 1, 2, 3], &net, &rsb);
        let mut best = i64::MIN;
        let mut best_order = Vec::new();
        let mut perm = vec![0usize, 1, 2, 3];
        fn walk(perm: &mut [usize], i: usize, net: &[Vec<i64>], best: &mut i64, best_order: &mut Vec<usize>) {
            if i == perm.len() {
                let score = kemeny_pair_score(perm, net);
                if score > *best {
                    *best = score;
                    *best_order = perm.to_vec();
                }
                return;
            }
            for j in i..perm.len() {
                perm.swap(i, j);
                walk(perm, i + 1, net, best, best_order);
                perm.swap(i, j);
            }
        }
        walk(&mut perm, 0, &net, &mut best, &mut best_order);
        assert_eq!(kemeny_pair_score(&solved, &net), best);
        let solved_key: Vec<usize> = solved.iter().copied().collect();
        let mut lex_best = solved_key.clone();
        let mut perm = vec![0usize, 1, 2, 3];
        fn walk_lex(perm: &mut [usize], i: usize, net: &[Vec<i64>], best: i64, lex_best: &mut Vec<usize>) {
            if i == perm.len() {
                if kemeny_pair_score(perm, net) == best && perm < lex_best {
                    *lex_best = perm.to_vec();
                }
                return;
            }
            for j in i..perm.len() {
                perm.swap(i, j);
                walk_lex(perm, i + 1, net, best, lex_best);
                perm.swap(i, j);
            }
        }
        walk_lex(&mut perm, 0, &net, best, &mut lex_best);
        assert_eq!(solved_key, lex_best);
    }

    #[test]
    fn kemeny_tie_keeps_the_rsb_order() {
        let net = vec![vec![0i64; 4]; 4];
        assert_eq!(kemeny_young(&net, &[0, 1, 2, 3]), vec![0, 1, 2, 3]);
    }

    #[test]
    fn gatekeeper_demotes_a_superseded_rank_one() {
        let sessions = vec![s("s", vec![
            t("old", "user", "The camera is a Sony A7."),
            t("fresh", "user", "The camera is a Sony A7C."),
        ])];
        let mut ix = ix_of(&sessions);
        let old = idx(&ix, "old");
        let fresh = idx(&ix, "fresh");
        ix.turns[fresh].supersedes_turns.push(old);
        let mut scene = vec![0.0f32; ix.turns.len()];
        scene[old] = 3.0;
        scene[fresh] = 1.0;
        let mut list = vec![old, fresh];
        rank1_gatekeeper(&ix, "which camera", &mut list, &scene);
        assert_eq!(list[0], fresh, "a superseded turn cannot hold rank 1");
    }

    #[test]
    fn gatekeeper_when_query_prefers_a_clock_form() {
        let sessions = vec![s("s", vec![
            t("plain", "user", "We met downtown after the show."),
            t("clock", "user", "We met at 18:40 after the show."),
        ])];
        let ix = ix_of(&sessions);
        let plain = idx(&ix, "plain");
        let clock = idx(&ix, "clock");
        assert!(ix.turns[clock].form_clock);
        assert!(!ix.turns[plain].form_clock);
        let scene = vec![1.0f32; ix.turns.len()];
        let mut list = vec![plain, clock];
        rank1_gatekeeper(&ix, "When did we meet?", &mut list, &scene);
        assert_eq!(list[0], clock);
    }

    #[test]
    fn gatekeeper_v2_keeps_the_rsb_head_and_reorders_the_tail() {
        let sessions = vec![s("s", vec![
            t("plain", "user", "We met downtown after the show."),
            t("other", "user", "We met downtown after the show again."),
            t("clock", "user", "We met at 18:40 after the show."),
        ])];
        let ix = ix_of(&sessions);
        let plain = idx(&ix, "plain");
        let other = idx(&ix, "other");
        let clock = idx(&ix, "clock");
        let scene = vec![1.0f32; ix.turns.len()];
        let mut list = vec![plain, other, clock];
        rank1_gatekeeper_keep_rsb_head(&ix, "When did we meet?", &mut list, &scene);
        assert_eq!(list, vec![plain, clock, other]);
    }

    #[test]
    fn gatekeeper_v2_does_not_demote_the_rsb_head() {
        let sessions = vec![s("s", vec![
            t("head", "user", "The camera is a Sony A7."),
            t("old", "user", "The lens is a fifty."),
            t("fresh", "user", "The lens is a thirty five."),
        ])];
        let mut ix = ix_of(&sessions);
        let head = idx(&ix, "head");
        let old = idx(&ix, "old");
        let fresh = idx(&ix, "fresh");
        ix.turns[fresh].supersedes_turns.push(head);
        ix.turns[fresh].supersedes_turns.push(old);
        let mut scene = vec![0.0f32; ix.turns.len()];
        scene[head] = 1.0;
        scene[old] = 3.0;
        scene[fresh] = 1.0;
        let mut list = vec![head, old, fresh];
        rank1_gatekeeper_keep_rsb_head(&ix, "which camera", &mut list, &scene);
        assert_eq!(list[0], head, "slot 1 stays the RSB head");
        assert_eq!(list[1], fresh, "ATMS still demotes a superseded turn inside the tail");
    }

    #[test]
    fn rank_is_identical_across_repeated_calls() {
        let mk = || RankRequest {
            sessions: haystack(),
            queries: vec![Query { text: "Which Sony camera do I have?".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        let a = rank(&mk());
        let b = rank(&mk());
        let ta: Vec<String> = a.results[0].turns.iter().map(|x| x.id.clone()).collect();
        let tb: Vec<String> = b.results[0].turns.iter().map(|x| x.id.clone()).collect();
        assert_eq!(ta, tb);
        assert_eq!(a.results[0].turns[0].id, b.results[0].turns[0].id);
        assert_eq!(a.steiner_bridge_fires, b.steiner_bridge_fires);
    }

    fn steiner_quad() -> Index {
        ix_of(&[
            s("s0", vec![t("head", "user", "The battery died after the show.")]),
            s("s1", vec![t("other", "user", "A spare charger is in the drawer.")]),
            s("s2", vec![t("bridge", "user", "Leave the spare next to the battery.")]),
            s("s3", vec![t("filler", "user", "The garden needs water tomorrow.")]),
        ])
    }

    #[test]
    fn steiner_bridge_fires_and_keeps_the_head() {
        let mut ix = steiner_quad();
        let head = idx(&ix, "head");
        let other = idx(&ix, "other");
        let bridge = idx(&ix, "bridge");
        let filler = idx(&ix, "filler");
        ix.turns[bridge].exact_edges.push((2, head));
        ix.turns[bridge].exact_edges.push((2, other));
        let scene = vec![1.0f32; ix.turns.len()];
        let mut list = vec![head, other, filler];
        let fired = steiner_bridge::apply(&ix, &mut list, 3, &scene);
        assert!(fired, "the bridge below the window has to enter the window");
        assert_eq!(list[0], head, "slot 0 stays the head");
        assert_eq!(list[1], bridge, "the Steiner node sits in slot 1");
        assert_eq!(list.len(), 3);
        assert!(!list.contains(&filler), "the marginal window turn leaves");
    }

    #[test]
    fn steiner_bridge_is_deterministic() {
        let mut ix = steiner_quad();
        let head = idx(&ix, "head");
        let other = idx(&ix, "other");
        let bridge = idx(&ix, "bridge");
        let filler = idx(&ix, "filler");
        ix.turns[bridge].exact_edges.push((2, head));
        ix.turns[bridge].exact_edges.push((2, other));
        let scene = vec![1.0f32; ix.turns.len()];
        let mut a = vec![head, other, filler];
        let mut b = a.clone();
        assert!(steiner_bridge::apply(&ix, &mut a, 3, &scene));
        assert!(steiner_bridge::apply(&ix, &mut b, 3, &scene));
        assert_eq!(a, b);
        assert_eq!(a[0], head);
    }

    #[test]
    fn steiner_bridge_does_not_fire_when_terminals_are_already_joined() {
        let mut ix = ix_of(&[
            s("s0", vec![
                t("head", "user", "The battery died after the show."),
                t("other", "user", "A spare charger is in the drawer."),
            ]),
            s("s1", vec![t("bridge", "user", "Leave the spare next to the battery.")]),
        ]);
        let head = idx(&ix, "head");
        let other = idx(&ix, "other");
        let bridge = idx(&ix, "bridge");
        ix.turns[bridge].exact_edges.push((2, head));
        ix.turns[bridge].exact_edges.push((2, other));
        let scene = vec![1.0f32; ix.turns.len()];
        let mut list = vec![head, other];
        let fired = steiner_bridge::apply(&ix, &mut list, 3, &scene);
        assert!(!fired);
        assert_eq!(list, vec![head, other]);
    }

    #[test]
    fn steiner_bridge_ignores_a_zero_scene_turn() {
        let mut ix = steiner_quad();
        let head = idx(&ix, "head");
        let other = idx(&ix, "other");
        let bridge = idx(&ix, "bridge");
        let filler = idx(&ix, "filler");
        ix.turns[bridge].exact_edges.push((2, head));
        ix.turns[bridge].exact_edges.push((2, other));
        let mut scene = vec![1.0f32; ix.turns.len()];
        scene[bridge] = 0.0;
        let mut list = vec![head, other, filler];
        assert!(!steiner_bridge::apply(&ix, &mut list, 3, &scene));
        assert_eq!(list, vec![head, other, filler]);
    }

    #[test]
    fn steiner_bridge_fires_on_a_shared_capitalized_name() {
        let ix = ix_of(&[
            s("s0", vec![t("head", "user", "My dog Scout has been limping since Monday.")]),
            s("s1", vec![t("other", "user", "We drove past the Riverside Clinic this morning.")]),
            s("s2", vec![t("bridge", "user", "I booked Scout at the Riverside Clinic.")]),
            s("s3", vec![t("filler", "user", "We bought oat milk at the market.")]),
        ]);
        let head = idx(&ix, "head");
        let other = idx(&ix, "other");
        let bridge = idx(&ix, "bridge");
        let filler = idx(&ix, "filler");
        let names = |ti: usize| -> Vec<String> {
            ix.bridge_tokens[ti].iter().map(|&id| ix.vocab.names[id as usize].clone()).collect()
        };
        assert!(names(head).iter().any(|n| n.contains("scout")), "{:?}", names(head));
        assert!(names(other).iter().any(|n| n.contains("riverside")), "{:?}", names(other));
        assert!(names(bridge).iter().any(|n| n.contains("scout")), "{:?}", names(bridge));
        let scene = vec![1.0f32; ix.turns.len()];
        let mut once = vec![head, other, filler];
        let mut twice = once.clone();
        assert!(steiner_bridge::apply(&ix, &mut once, 3, &scene));
        assert!(steiner_bridge::apply(&ix, &mut twice, 3, &scene));
        assert_eq!(once, twice);
        assert_eq!(once[0], head);
        assert_eq!(once[1], bridge);
    }

    #[test]
    fn steiner_bridge_does_not_break_an_equal_tie() {
        let mut ix = steiner_quad();
        let head = idx(&ix, "head");
        let other = idx(&ix, "other");
        let bridge = idx(&ix, "bridge");
        let filler = idx(&ix, "filler");
        for turn in [bridge, filler] {
            ix.turns[turn].exact_edges.push((2, head));
            ix.turns[turn].exact_edges.push((2, other));
        }
        let scene = vec![1.0f32; ix.turns.len()];
        let mut list = vec![head, other];
        assert!(!steiner_bridge::apply(&ix, &mut list, 3, &scene));
        assert_eq!(list, vec![head, other]);
    }

    #[test]
    fn steiner_bridge_ignores_a_same_session_mate_as_a_touch() {
        let mut ix = ix_of(&[
            s("s0", vec![
                t("head", "user", "The battery died after the show."),
                t("mate", "user", "A spare charger is in the drawer."),
            ]),
            s("s1", vec![t("other", "user", "Leave the spare next to the battery.")]),
            s("s2", vec![t("bridge", "user", "The garden needs water tomorrow.")]),
        ]);
        let head = idx(&ix, "head");
        let mate = idx(&ix, "mate");
        let other = idx(&ix, "other");
        let bridge = idx(&ix, "bridge");
        ix.turns[mate].exact_edges.push((2, other));
        ix.turns[bridge].exact_edges.push((2, head));
        ix.turns[bridge].exact_edges.push((2, other));
        let scene = vec![1.0f32; ix.turns.len()];
        let mut list = vec![head, other];
        assert!(steiner_bridge::apply(&ix, &mut list, 3, &scene));
        assert_eq!(list[0], head);
        assert_eq!(list[1], bridge);
    }

    #[test]
    fn steiner_bridge_fires_inside_rank_and_keeps_slot_zero() {
        let mut bridge = t("bridge", "user", "A sprain was noted in the margin.");
        bridge.links = vec![
            TurnLink { kind: "exact".into(), to: "head".into(), degree: 2 },
            TurnLink { kind: "exact".into(), to: "other".into(), degree: 2 },
        ];
        let sessions = vec![
            s("s0", vec![t("head", "user", "The limping sprain started on the trail after a long hike yesterday.")]),
            s("s1", vec![t("other", "user", "The limping sprain was treated at the clinic after the hike.")]),
            s("s2", vec![bridge]),
            s("s3", vec![t("filler", "user", "The garden needs water tomorrow and the oats are dry.")]),
        ];
        let req = RankRequest {
            sessions,
            queries: vec![Query { text: "When did the limping sprain start?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let a = rank(&req);
        let b = rank(&req);
        let ids = |r: &RankResponse| r.results[0].turns.iter().map(|t| t.id.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&a), ids(&b));
        assert_eq!(a.results[0].turns[0].id, b.results[0].turns[0].id);
        assert_eq!(a.steiner_bridge_fires, b.steiner_bridge_fires);
        assert!(a.results[0].steiner_bridge_fired, "window={:?}", ids(&a));
        assert_ne!(a.results[0].turns[0].id, "bridge", "slot 0 stays the head, window={:?}", ids(&a));
        assert_eq!(a.results[0].turns[1].id, "bridge", "window={:?}", ids(&a));
    }
}
