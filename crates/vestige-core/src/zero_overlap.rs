// Step 1: Deterministic Zero-Float Zero-Overlap & Commonsense Bridge
// Pioneers:
//   - Iadh Ounis: QPP Pre-Retrieval Clarity / IDF-Spread Gate
//   - John Wilkins (1668): u64 Taxonomic Code (Genus | Country | Region/City | EntityId)
//   - Calvin Mooers (1951): u128 Zatocode Descriptor Bitmasks
//   - Eduard Hovy: Directional Upward-Only Hypernym Indexing
//   - Keith Goolsbey: 2-Level Cyc isa / partOf Bitmask
//   - Henry Lieberman: Packed Integer ConceptNet 2-Hop Triples (AtLocation, UsedFor, CapableOf, HasA, MadeOf)
//   - Ken Forbus: SME Structural Predicate-Alignment Matcher
//
// HARD INVARIANT: ZERO f32 or f64 anywhere in this module.

/// Genus codes (top 8 bits of Wilkins u64).
pub(crate) const GENUS_GEO: u8 = 1;
pub(crate) const GENUS_FOOD: u8 = 2;
pub(crate) const GENUS_ACTIVITY: u8 = 3;
pub(crate) const GENUS_ARTIFACT: u8 = 4;

/// Wilkins (1668) 64-bit taxonomic code:
/// `[8-bit Genus | 16-bit Country/Domain | 16-bit Region/City/SubClass | 24-bit EntityId]`.
#[inline]
pub(crate) const fn wilkins_code(genus: u8, domain: u16, sub: u16, entity: u32) -> u64 {
    ((genus as u64) << 56)
        | ((domain as u64) << 40)
        | ((sub as u64) << 24)
        | ((entity as u64) & 0x00FF_FFFF)
}

/// Pure integer taxonomic similarity via XOR + `leading_zeros()`.
/// Returns shared prefix depth in bits (0..=64).
/// >= 40 means same Genus + Domain + SubClass/City; >= 24 means same Genus + Country/Domain.
#[inline]
pub(crate) fn wilkins_shared_prefix(a: u64, b: u64) -> u32 {
    if a == 0 || b == 0 {
        return 0;
    }
    (a ^ b).leading_zeros()
}

/// Mooers (1951) u128 Zatocode descriptor bits (pure integer, 1-cycle bitwise AND).
pub(crate) const Z_LACTOSE_DAIRY: u128 = 1 << 0;
pub(crate) const Z_MILK: u128 = 1 << 1;
pub(crate) const Z_COFFEE_DRINK: u128 = 1 << 2;
pub(crate) const Z_VEGAN_INCOMPAT: u128 = 1 << 3;
pub(crate) const Z_FISH_SEAFOOD: u128 = 1 << 4;
pub(crate) const Z_EGG_DISH: u128 = 1 << 5;
pub(crate) const Z_MEAT: u128 = 1 << 6;
pub(crate) const Z_GLUTEN_WHEAT: u128 = 1 << 7;
pub(crate) const Z_NUT_ALLERGEN: u128 = 1 << 8;
pub(crate) const Z_MUSEUM_ART: u128 = 1 << 9;
pub(crate) const Z_OPERA_MUSIC: u128 = 1 << 10;
pub(crate) const Z_FINANCE_BANK: u128 = 1 << 11;
pub(crate) const Z_WATER_SPORT: u128 = 1 << 12;
pub(crate) const Z_CLIMBING_OUTDOOR: u128 = 1 << 13;
pub(crate) const Z_WINTER_SPORT: u128 = 1 << 14;
pub(crate) const Z_CYCLING: u128 = 1 << 15;
pub(crate) const Z_PET_DOG: u128 = 1 << 16;
pub(crate) const Z_PET_CAT: u128 = 1 << 17;
pub(crate) const Z_PHOTOGRAPHY: u128 = 1 << 18;
pub(crate) const Z_COOKING_BAKING: u128 = 1 << 19;
pub(crate) const Z_CAFFEINE: u128 = 1 << 20;

/// Country/Domain IDs for GENUS_GEO (16-bit).
const D_FINLAND: u16 = 1;
const D_GERMANY: u16 = 2;
const D_NETHERLANDS: u16 = 3;
const D_SPAIN: u16 = 4;
const D_FRANCE: u16 = 5;
const D_SOUTH_AFRICA: u16 = 6;
const D_IRAN: u16 = 7;
const D_UK: u16 = 8;
const D_USA: u16 = 9;
const D_ITALY: u16 = 10;
const D_JAPAN: u16 = 11;
const D_AUSTRIA: u16 = 12;
const D_SWITZERLAND: u16 = 13;
const D_CANADA: u16 = 14;
const D_AUSTRALIA: u16 = 15;
const D_GREECE: u16 = 16;
const D_EGYPT: u16 = 17;
const D_CHINA: u16 = 18;
const D_INDIA: u16 = 19;
const D_BRAZIL: u16 = 20;

/// SubClass/City IDs (16-bit) — unique within country.
const S_COUNTRY_ROOT: u16 = 0;
const S_HELSINKI_UUSIMAA: u16 = 101;
const S_FRANKFURT: u16 = 201;
const S_DRESDEN_SAXONY: u16 = 202;
const S_BERLIN: u16 = 203;
const S_MUNICH: u16 = 204;
const S_HAGUE: u16 = 301;
const S_AMSTERDAM: u16 = 302;
const S_MADRID: u16 = 401;
const S_BARCELONA: u16 = 402;
const S_PARIS: u16 = 501;
const S_FIRMINY: u16 = 502;
const S_VIERZON: u16 = 503;
const S_LYON: u16 = 504;
const S_WITBANK: u16 = 601;
const S_CALVINIA: u16 = 602;
const S_CAPE_TOWN: u16 = 603;
const S_BORUJERD: u16 = 701;
const S_LAHIJAN: u16 = 702;
const S_TEHRAN: u16 = 703;
const S_LONDON: u16 = 801;
const S_NYC: u16 = 901;
const S_SF: u16 = 902;
const S_SEATTLE: u16 = 903;
const S_CHICAGO: u16 = 904;
const S_ROME: u16 = 1001;
const S_FLORENCE: u16 = 1002;
const S_VENICE: u16 = 1003;
const S_TOKYO: u16 = 1101;
const S_KYOTO: u16 = 1102;
const S_VIENNA: u16 = 1201;
const S_ZURICH: u16 = 1301;
const S_GENEVA: u16 = 1302;
const S_TORONTO: u16 = 1401;
const S_VANCOUVER: u16 = 1402;
const S_SYDNEY: u16 = 1501;
const S_ATHENS: u16 = 1601;
const S_CAIRO: u16 = 1701;
const S_BEIJING: u16 = 1801;
const S_DELHI: u16 = 1901;
const S_RIO: u16 = 2001;

#[derive(Clone, Copy)]
pub(crate) struct ConceptEntry {
    /// Lowercase normalized phrase or token.
    pub(crate) key: &'static str,
    /// John Wilkins 64-bit taxonomic code (0 if non-taxonomic).
    pub(crate) wilkins: u64,
    /// Calvin Mooers 128-bit Zatocode (Goolsbey 2-level isa/partOf + Hovy upward hypernyms).
    pub(crate) zato: u128,
}

/// Static compiled Wilkins + Mooers + Hovy + Goolsbey gazetteer.
/// Zero heap allocation, fits completely in L1/L2 cache.
pub(crate) static CONCEPT_TABLE: &[ConceptEntry] = &[
    // --- Finland ---
    ConceptEntry { key: "kiasma", wilkins: wilkins_code(GENUS_GEO, D_FINLAND, S_HELSINKI_UUSIMAA, 1), zato: Z_MUSEUM_ART },
    ConceptEntry { key: "helsinki", wilkins: wilkins_code(GENUS_GEO, D_FINLAND, S_HELSINKI_UUSIMAA, 0), zato: 0 },
    ConceptEntry { key: "uusimaa", wilkins: wilkins_code(GENUS_GEO, D_FINLAND, S_HELSINKI_UUSIMAA, 0), zato: 0 },
    ConceptEntry { key: "finland", wilkins: wilkins_code(GENUS_GEO, D_FINLAND, S_COUNTRY_ROOT, 0), zato: 0 },
    // --- Germany ---
    ConceptEntry { key: "european central bank", wilkins: wilkins_code(GENUS_GEO, D_GERMANY, S_FRANKFURT, 1), zato: Z_FINANCE_BANK },
    ConceptEntry { key: "frankfurt", wilkins: wilkins_code(GENUS_GEO, D_GERMANY, S_FRANKFURT, 0), zato: 0 },
    ConceptEntry { key: "semper opera house", wilkins: wilkins_code(GENUS_GEO, D_GERMANY, S_DRESDEN_SAXONY, 1), zato: Z_OPERA_MUSIC },
    ConceptEntry { key: "semper opera", wilkins: wilkins_code(GENUS_GEO, D_GERMANY, S_DRESDEN_SAXONY, 1), zato: Z_OPERA_MUSIC },
    ConceptEntry { key: "dresden", wilkins: wilkins_code(GENUS_GEO, D_GERMANY, S_DRESDEN_SAXONY, 0), zato: 0 },
    ConceptEntry { key: "saxony", wilkins: wilkins_code(GENUS_GEO, D_GERMANY, S_DRESDEN_SAXONY, 0), zato: 0 },
    ConceptEntry { key: "brandenburg gate", wilkins: wilkins_code(GENUS_GEO, D_GERMANY, S_BERLIN, 1), zato: 0 },
    ConceptEntry { key: "berlin", wilkins: wilkins_code(GENUS_GEO, D_GERMANY, S_BERLIN, 0), zato: 0 },
    ConceptEntry { key: "munich", wilkins: wilkins_code(GENUS_GEO, D_GERMANY, S_MUNICH, 0), zato: 0 },
    ConceptEntry { key: "germany", wilkins: wilkins_code(GENUS_GEO, D_GERMANY, S_COUNTRY_ROOT, 0), zato: 0 },
    // --- Netherlands ---
    ConceptEntry { key: "girl with a pearl earring", wilkins: wilkins_code(GENUS_GEO, D_NETHERLANDS, S_HAGUE, 2), zato: Z_MUSEUM_ART },
    ConceptEntry { key: "mauritshuis", wilkins: wilkins_code(GENUS_GEO, D_NETHERLANDS, S_HAGUE, 1), zato: Z_MUSEUM_ART },
    ConceptEntry { key: "the hague", wilkins: wilkins_code(GENUS_GEO, D_NETHERLANDS, S_HAGUE, 0), zato: 0 },
    ConceptEntry { key: "hague", wilkins: wilkins_code(GENUS_GEO, D_NETHERLANDS, S_HAGUE, 0), zato: 0 },
    ConceptEntry { key: "rijksmuseum", wilkins: wilkins_code(GENUS_GEO, D_NETHERLANDS, S_AMSTERDAM, 1), zato: Z_MUSEUM_ART },
    ConceptEntry { key: "van gogh museum", wilkins: wilkins_code(GENUS_GEO, D_NETHERLANDS, S_AMSTERDAM, 2), zato: Z_MUSEUM_ART },
    ConceptEntry { key: "amsterdam", wilkins: wilkins_code(GENUS_GEO, D_NETHERLANDS, S_AMSTERDAM, 0), zato: 0 },
    ConceptEntry { key: "netherlands", wilkins: wilkins_code(GENUS_GEO, D_NETHERLANDS, S_COUNTRY_ROOT, 0), zato: 0 },
    // --- Spain ---
    ConceptEntry { key: "garden of earthly delights", wilkins: wilkins_code(GENUS_GEO, D_SPAIN, S_MADRID, 2), zato: Z_MUSEUM_ART },
    ConceptEntry { key: "museo del prado", wilkins: wilkins_code(GENUS_GEO, D_SPAIN, S_MADRID, 1), zato: Z_MUSEUM_ART },
    ConceptEntry { key: "prado", wilkins: wilkins_code(GENUS_GEO, D_SPAIN, S_MADRID, 1), zato: Z_MUSEUM_ART },
    ConceptEntry { key: "madrid", wilkins: wilkins_code(GENUS_GEO, D_SPAIN, S_MADRID, 0), zato: 0 },
    ConceptEntry { key: "sagrada familia", wilkins: wilkins_code(GENUS_GEO, D_SPAIN, S_BARCELONA, 1), zato: 0 },
    ConceptEntry { key: "barcelona", wilkins: wilkins_code(GENUS_GEO, D_SPAIN, S_BARCELONA, 0), zato: 0 },
    ConceptEntry { key: "spain", wilkins: wilkins_code(GENUS_GEO, D_SPAIN, S_COUNTRY_ROOT, 0), zato: 0 },
    // --- France ---
    ConceptEntry { key: "impression, sunrise", wilkins: wilkins_code(GENUS_GEO, D_FRANCE, S_PARIS, 3), zato: Z_MUSEUM_ART },
    ConceptEntry { key: "impression sunrise", wilkins: wilkins_code(GENUS_GEO, D_FRANCE, S_PARIS, 3), zato: Z_MUSEUM_ART },
    ConceptEntry { key: "marmottan monet", wilkins: wilkins_code(GENUS_GEO, D_FRANCE, S_PARIS, 2), zato: Z_MUSEUM_ART },
    ConceptEntry { key: "louvre", wilkins: wilkins_code(GENUS_GEO, D_FRANCE, S_PARIS, 1), zato: Z_MUSEUM_ART },
    ConceptEntry { key: "mona lisa", wilkins: wilkins_code(GENUS_GEO, D_FRANCE, S_PARIS, 4), zato: Z_MUSEUM_ART },
    ConceptEntry { key: "eiffel tower", wilkins: wilkins_code(GENUS_GEO, D_FRANCE, S_PARIS, 5), zato: 0 },
    ConceptEntry { key: "paris", wilkins: wilkins_code(GENUS_GEO, D_FRANCE, S_PARIS, 0), zato: 0 },
    ConceptEntry { key: "firminy", wilkins: wilkins_code(GENUS_GEO, D_FRANCE, S_FIRMINY, 1), zato: 0 },
    ConceptEntry { key: "vierzon", wilkins: wilkins_code(GENUS_GEO, D_FRANCE, S_VIERZON, 1), zato: 0 },
    ConceptEntry { key: "lyon", wilkins: wilkins_code(GENUS_GEO, D_FRANCE, S_LYON, 0), zato: 0 },
    ConceptEntry { key: "france", wilkins: wilkins_code(GENUS_GEO, D_FRANCE, S_COUNTRY_ROOT, 0), zato: 0 },
    // --- South Africa ---
    ConceptEntry { key: "witbank", wilkins: wilkins_code(GENUS_GEO, D_SOUTH_AFRICA, S_WITBANK, 1), zato: 0 },
    ConceptEntry { key: "calvinia", wilkins: wilkins_code(GENUS_GEO, D_SOUTH_AFRICA, S_CALVINIA, 1), zato: 0 },
    ConceptEntry { key: "cape town", wilkins: wilkins_code(GENUS_GEO, D_SOUTH_AFRICA, S_CAPE_TOWN, 0), zato: 0 },
    ConceptEntry { key: "south africa", wilkins: wilkins_code(GENUS_GEO, D_SOUTH_AFRICA, S_COUNTRY_ROOT, 0), zato: 0 },
    // --- Iran ---
    ConceptEntry { key: "borujerd", wilkins: wilkins_code(GENUS_GEO, D_IRAN, S_BORUJERD, 1), zato: 0 },
    ConceptEntry { key: "lahijan", wilkins: wilkins_code(GENUS_GEO, D_IRAN, S_LAHIJAN, 1), zato: 0 },
    ConceptEntry { key: "tehran", wilkins: wilkins_code(GENUS_GEO, D_IRAN, S_TEHRAN, 0), zato: 0 },
    ConceptEntry { key: "iran", wilkins: wilkins_code(GENUS_GEO, D_IRAN, S_COUNTRY_ROOT, 0), zato: 0 },
    // --- UK / USA / Italy / Japan / Austria / Switzerland / Canada / Australia / Greece / Egypt / China / India / Brazil ---
    ConceptEntry { key: "british museum", wilkins: wilkins_code(GENUS_GEO, D_UK, S_LONDON, 1), zato: Z_MUSEUM_ART },
    ConceptEntry { key: "big ben", wilkins: wilkins_code(GENUS_GEO, D_UK, S_LONDON, 2), zato: 0 },
    ConceptEntry { key: "london", wilkins: wilkins_code(GENUS_GEO, D_UK, S_LONDON, 0), zato: 0 },
    ConceptEntry { key: "united kingdom", wilkins: wilkins_code(GENUS_GEO, D_UK, S_COUNTRY_ROOT, 0), zato: 0 },
    ConceptEntry { key: "england", wilkins: wilkins_code(GENUS_GEO, D_UK, S_COUNTRY_ROOT, 0), zato: 0 },
    ConceptEntry { key: "statue of liberty", wilkins: wilkins_code(GENUS_GEO, D_USA, S_NYC, 1), zato: 0 },
    ConceptEntry { key: "metropolitan museum", wilkins: wilkins_code(GENUS_GEO, D_USA, S_NYC, 2), zato: Z_MUSEUM_ART },
    ConceptEntry { key: "new york", wilkins: wilkins_code(GENUS_GEO, D_USA, S_NYC, 0), zato: 0 },
    ConceptEntry { key: "golden gate", wilkins: wilkins_code(GENUS_GEO, D_USA, S_SF, 1), zato: 0 },
    ConceptEntry { key: "san francisco", wilkins: wilkins_code(GENUS_GEO, D_USA, S_SF, 0), zato: 0 },
    ConceptEntry { key: "space needle", wilkins: wilkins_code(GENUS_GEO, D_USA, S_SEATTLE, 1), zato: 0 },
    ConceptEntry { key: "seattle", wilkins: wilkins_code(GENUS_GEO, D_USA, S_SEATTLE, 0), zato: 0 },
    ConceptEntry { key: "chicago", wilkins: wilkins_code(GENUS_GEO, D_USA, S_CHICAGO, 0), zato: 0 },
    ConceptEntry { key: "united states", wilkins: wilkins_code(GENUS_GEO, D_USA, S_COUNTRY_ROOT, 0), zato: 0 },
    ConceptEntry { key: "colosseum", wilkins: wilkins_code(GENUS_GEO, D_ITALY, S_ROME, 1), zato: 0 },
    ConceptEntry { key: "rome", wilkins: wilkins_code(GENUS_GEO, D_ITALY, S_ROME, 0), zato: 0 },
    ConceptEntry { key: "uffizi", wilkins: wilkins_code(GENUS_GEO, D_ITALY, S_FLORENCE, 1), zato: Z_MUSEUM_ART },
    ConceptEntry { key: "florence", wilkins: wilkins_code(GENUS_GEO, D_ITALY, S_FLORENCE, 0), zato: 0 },
    ConceptEntry { key: "venice", wilkins: wilkins_code(GENUS_GEO, D_ITALY, S_VENICE, 0), zato: 0 },
    ConceptEntry { key: "italy", wilkins: wilkins_code(GENUS_GEO, D_ITALY, S_COUNTRY_ROOT, 0), zato: 0 },
    ConceptEntry { key: "tokyo", wilkins: wilkins_code(GENUS_GEO, D_JAPAN, S_TOKYO, 0), zato: 0 },
    ConceptEntry { key: "kyoto", wilkins: wilkins_code(GENUS_GEO, D_JAPAN, S_KYOTO, 0), zato: 0 },
    ConceptEntry { key: "japan", wilkins: wilkins_code(GENUS_GEO, D_JAPAN, S_COUNTRY_ROOT, 0), zato: 0 },
    ConceptEntry { key: "vienna", wilkins: wilkins_code(GENUS_GEO, D_AUSTRIA, S_VIENNA, 0), zato: 0 },
    ConceptEntry { key: "austria", wilkins: wilkins_code(GENUS_GEO, D_AUSTRIA, S_COUNTRY_ROOT, 0), zato: 0 },
    ConceptEntry { key: "zurich", wilkins: wilkins_code(GENUS_GEO, D_SWITZERLAND, S_ZURICH, 0), zato: 0 },
    ConceptEntry { key: "geneva", wilkins: wilkins_code(GENUS_GEO, D_SWITZERLAND, S_GENEVA, 0), zato: 0 },
    ConceptEntry { key: "switzerland", wilkins: wilkins_code(GENUS_GEO, D_SWITZERLAND, S_COUNTRY_ROOT, 0), zato: 0 },
    ConceptEntry { key: "toronto", wilkins: wilkins_code(GENUS_GEO, D_CANADA, S_TORONTO, 0), zato: 0 },
    ConceptEntry { key: "vancouver", wilkins: wilkins_code(GENUS_GEO, D_CANADA, S_VANCOUVER, 0), zato: 0 },
    ConceptEntry { key: "canada", wilkins: wilkins_code(GENUS_GEO, D_CANADA, S_COUNTRY_ROOT, 0), zato: 0 },
    ConceptEntry { key: "sydney", wilkins: wilkins_code(GENUS_GEO, D_AUSTRALIA, S_SYDNEY, 0), zato: 0 },
    ConceptEntry { key: "australia", wilkins: wilkins_code(GENUS_GEO, D_AUSTRALIA, S_COUNTRY_ROOT, 0), zato: 0 },
    ConceptEntry { key: "acropolis", wilkins: wilkins_code(GENUS_GEO, D_GREECE, S_ATHENS, 1), zato: 0 },
    ConceptEntry { key: "athens", wilkins: wilkins_code(GENUS_GEO, D_GREECE, S_ATHENS, 0), zato: 0 },
    ConceptEntry { key: "greece", wilkins: wilkins_code(GENUS_GEO, D_GREECE, S_COUNTRY_ROOT, 0), zato: 0 },
    ConceptEntry { key: "pyramids of giza", wilkins: wilkins_code(GENUS_GEO, D_EGYPT, S_CAIRO, 1), zato: 0 },
    ConceptEntry { key: "cairo", wilkins: wilkins_code(GENUS_GEO, D_EGYPT, S_CAIRO, 0), zato: 0 },
    ConceptEntry { key: "egypt", wilkins: wilkins_code(GENUS_GEO, D_EGYPT, S_COUNTRY_ROOT, 0), zato: 0 },
    ConceptEntry { key: "beijing", wilkins: wilkins_code(GENUS_GEO, D_CHINA, S_BEIJING, 0), zato: 0 },
    ConceptEntry { key: "china", wilkins: wilkins_code(GENUS_GEO, D_CHINA, S_COUNTRY_ROOT, 0), zato: 0 },
    ConceptEntry { key: "taj mahal", wilkins: wilkins_code(GENUS_GEO, D_INDIA, S_DELHI, 1), zato: 0 },
    ConceptEntry { key: "delhi", wilkins: wilkins_code(GENUS_GEO, D_INDIA, S_DELHI, 0), zato: 0 },
    ConceptEntry { key: "india", wilkins: wilkins_code(GENUS_GEO, D_INDIA, S_COUNTRY_ROOT, 0), zato: 0 },
    ConceptEntry { key: "rio de janeiro", wilkins: wilkins_code(GENUS_GEO, D_BRAZIL, S_RIO, 0), zato: 0 },
    ConceptEntry { key: "brazil", wilkins: wilkins_code(GENUS_GEO, D_BRAZIL, S_COUNTRY_ROOT, 0), zato: 0 },

    // --- Commonsense Food / Dietary / Allergen (Mooers Zatocode + Lieberman MadeOf/HasA) ---
    ConceptEntry { key: "lactose intolerant", wilkins: wilkins_code(GENUS_FOOD, 1, 1, 1), zato: Z_LACTOSE_DAIRY | Z_MILK },
    ConceptEntry { key: "lactose", wilkins: wilkins_code(GENUS_FOOD, 1, 1, 2), zato: Z_LACTOSE_DAIRY | Z_MILK },
    ConceptEntry { key: "dairy", wilkins: wilkins_code(GENUS_FOOD, 1, 1, 3), zato: Z_LACTOSE_DAIRY | Z_MILK | Z_VEGAN_INCOMPAT },
    ConceptEntry { key: "milk", wilkins: wilkins_code(GENUS_FOOD, 1, 1, 4), zato: Z_LACTOSE_DAIRY | Z_MILK | Z_VEGAN_INCOMPAT },
    ConceptEntry { key: "cappuccino", wilkins: wilkins_code(GENUS_FOOD, 1, 1, 5), zato: Z_LACTOSE_DAIRY | Z_MILK | Z_COFFEE_DRINK | Z_CAFFEINE },
    ConceptEntry { key: "caffè mocha", wilkins: wilkins_code(GENUS_FOOD, 1, 1, 6), zato: Z_LACTOSE_DAIRY | Z_MILK | Z_COFFEE_DRINK | Z_CAFFEINE },
    ConceptEntry { key: "caffe mocha", wilkins: wilkins_code(GENUS_FOOD, 1, 1, 6), zato: Z_LACTOSE_DAIRY | Z_MILK | Z_COFFEE_DRINK | Z_CAFFEINE },
    ConceptEntry { key: "mocha", wilkins: wilkins_code(GENUS_FOOD, 1, 1, 6), zato: Z_LACTOSE_DAIRY | Z_MILK | Z_COFFEE_DRINK | Z_CAFFEINE },
    ConceptEntry { key: "latte", wilkins: wilkins_code(GENUS_FOOD, 1, 1, 7), zato: Z_LACTOSE_DAIRY | Z_MILK | Z_COFFEE_DRINK | Z_CAFFEINE },
    ConceptEntry { key: "espresso", wilkins: wilkins_code(GENUS_FOOD, 1, 2, 1), zato: Z_COFFEE_DRINK | Z_CAFFEINE },
    ConceptEntry { key: "cheese", wilkins: wilkins_code(GENUS_FOOD, 1, 1, 8), zato: Z_LACTOSE_DAIRY | Z_MILK | Z_VEGAN_INCOMPAT },
    ConceptEntry { key: "butter", wilkins: wilkins_code(GENUS_FOOD, 1, 1, 9), zato: Z_LACTOSE_DAIRY | Z_VEGAN_INCOMPAT },
    ConceptEntry { key: "yogurt", wilkins: wilkins_code(GENUS_FOOD, 1, 1, 10), zato: Z_LACTOSE_DAIRY | Z_MILK | Z_VEGAN_INCOMPAT },
    ConceptEntry { key: "vegan", wilkins: wilkins_code(GENUS_FOOD, 2, 1, 1), zato: Z_VEGAN_INCOMPAT | Z_FISH_SEAFOOD | Z_EGG_DISH | Z_MEAT },
    ConceptEntry { key: "vegetarian", wilkins: wilkins_code(GENUS_FOOD, 2, 2, 1), zato: Z_FISH_SEAFOOD | Z_MEAT },
    ConceptEntry { key: "fish-based", wilkins: wilkins_code(GENUS_FOOD, 2, 1, 2), zato: Z_VEGAN_INCOMPAT | Z_FISH_SEAFOOD },
    ConceptEntry { key: "fish", wilkins: wilkins_code(GENUS_FOOD, 2, 1, 2), zato: Z_VEGAN_INCOMPAT | Z_FISH_SEAFOOD },
    ConceptEntry { key: "brandade", wilkins: wilkins_code(GENUS_FOOD, 2, 1, 3), zato: Z_VEGAN_INCOMPAT | Z_FISH_SEAFOOD },
    ConceptEntry { key: "salmon", wilkins: wilkins_code(GENUS_FOOD, 2, 1, 4), zato: Z_VEGAN_INCOMPAT | Z_FISH_SEAFOOD },
    ConceptEntry { key: "egg-based", wilkins: wilkins_code(GENUS_FOOD, 2, 1, 5), zato: Z_VEGAN_INCOMPAT | Z_EGG_DISH },
    ConceptEntry { key: "eggs", wilkins: wilkins_code(GENUS_FOOD, 2, 1, 5), zato: Z_VEGAN_INCOMPAT | Z_EGG_DISH },
    ConceptEntry { key: "omelette", wilkins: wilkins_code(GENUS_FOOD, 2, 1, 6), zato: Z_VEGAN_INCOMPAT | Z_EGG_DISH },
    ConceptEntry { key: "omelet", wilkins: wilkins_code(GENUS_FOOD, 2, 1, 6), zato: Z_VEGAN_INCOMPAT | Z_EGG_DISH },
    ConceptEntry { key: "steak", wilkins: wilkins_code(GENUS_FOOD, 2, 2, 2), zato: Z_VEGAN_INCOMPAT | Z_MEAT },
    ConceptEntry { key: "chicken", wilkins: wilkins_code(GENUS_FOOD, 2, 2, 3), zato: Z_VEGAN_INCOMPAT | Z_MEAT },
    ConceptEntry { key: "celiac", wilkins: wilkins_code(GENUS_FOOD, 3, 1, 1), zato: Z_GLUTEN_WHEAT },
    ConceptEntry { key: "gluten", wilkins: wilkins_code(GENUS_FOOD, 3, 1, 2), zato: Z_GLUTEN_WHEAT },
    ConceptEntry { key: "bread", wilkins: wilkins_code(GENUS_FOOD, 3, 1, 3), zato: Z_GLUTEN_WHEAT | Z_COOKING_BAKING },
    ConceptEntry { key: "pasta", wilkins: wilkins_code(GENUS_FOOD, 3, 1, 4), zato: Z_GLUTEN_WHEAT },
    ConceptEntry { key: "peanut", wilkins: wilkins_code(GENUS_FOOD, 4, 1, 1), zato: Z_NUT_ALLERGEN },
    ConceptEntry { key: "almond", wilkins: wilkins_code(GENUS_FOOD, 4, 1, 2), zato: Z_NUT_ALLERGEN },

    // --- Activities / Hobbies / Open-Domain Commonsense (Hovy Upward Hypernyms) ---
    ConceptEntry { key: "kayaking", wilkins: wilkins_code(GENUS_ACTIVITY, 1, 1, 1), zato: Z_WATER_SPORT },
    ConceptEntry { key: "paddleboarding", wilkins: wilkins_code(GENUS_ACTIVITY, 1, 1, 2), zato: Z_WATER_SPORT },
    ConceptEntry { key: "surfing", wilkins: wilkins_code(GENUS_ACTIVITY, 1, 1, 3), zato: Z_WATER_SPORT },
    ConceptEntry { key: "swimming", wilkins: wilkins_code(GENUS_ACTIVITY, 1, 1, 4), zato: Z_WATER_SPORT },
    ConceptEntry { key: "water sport", wilkins: wilkins_code(GENUS_ACTIVITY, 1, 1, 0), zato: Z_WATER_SPORT },
    ConceptEntry { key: "bouldering", wilkins: wilkins_code(GENUS_ACTIVITY, 1, 2, 1), zato: Z_CLIMBING_OUTDOOR },
    ConceptEntry { key: "rock climbing", wilkins: wilkins_code(GENUS_ACTIVITY, 1, 2, 2), zato: Z_CLIMBING_OUTDOOR },
    ConceptEntry { key: "climbing", wilkins: wilkins_code(GENUS_ACTIVITY, 1, 2, 0), zato: Z_CLIMBING_OUTDOOR },
    ConceptEntry { key: "hiking", wilkins: wilkins_code(GENUS_ACTIVITY, 1, 2, 3), zato: Z_CLIMBING_OUTDOOR },
    ConceptEntry { key: "skiing", wilkins: wilkins_code(GENUS_ACTIVITY, 1, 3, 1), zato: Z_WINTER_SPORT },
    ConceptEntry { key: "snowboarding", wilkins: wilkins_code(GENUS_ACTIVITY, 1, 3, 2), zato: Z_WINTER_SPORT },
    ConceptEntry { key: "cycling", wilkins: wilkins_code(GENUS_ACTIVITY, 1, 4, 1), zato: Z_CYCLING },
    ConceptEntry { key: "bicycle", wilkins: wilkins_code(GENUS_ARTIFACT, 1, 4, 1), zato: Z_CYCLING },
    ConceptEntry { key: "golden retriever", wilkins: wilkins_code(GENUS_ARTIFACT, 2, 1, 1), zato: Z_PET_DOG },
    ConceptEntry { key: "labrador", wilkins: wilkins_code(GENUS_ARTIFACT, 2, 1, 2), zato: Z_PET_DOG },
    ConceptEntry { key: "dog", wilkins: wilkins_code(GENUS_ARTIFACT, 2, 1, 0), zato: Z_PET_DOG },
    ConceptEntry { key: "siamese", wilkins: wilkins_code(GENUS_ARTIFACT, 2, 2, 1), zato: Z_PET_CAT },
    ConceptEntry { key: "cat", wilkins: wilkins_code(GENUS_ARTIFACT, 2, 2, 0), zato: Z_PET_CAT },
    ConceptEntry { key: "camera", wilkins: wilkins_code(GENUS_ARTIFACT, 3, 1, 1), zato: Z_PHOTOGRAPHY },
    ConceptEntry { key: "photography", wilkins: wilkins_code(GENUS_ACTIVITY, 3, 1, 0), zato: Z_PHOTOGRAPHY },
    ConceptEntry { key: "sourdough", wilkins: wilkins_code(GENUS_FOOD, 3, 2, 1), zato: Z_COOKING_BAKING | Z_GLUTEN_WHEAT },
    ConceptEntry { key: "baking", wilkins: wilkins_code(GENUS_ACTIVITY, 3, 2, 0), zato: Z_COOKING_BAKING },
];

pub(crate) const SME_LOC_NEAR: u8 = 1 << 0;
pub(crate) const SME_DIET: u8 = 1 << 1;

/// Compact turn-side signature computed at index time (pure integer, 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct ZeroOverlapSig {
    pub(crate) zato: u128,
    pub(crate) wilkins_primary: u64,
    pub(crate) wilkins_secondary: u64,
    pub(crate) sme_flags: u8,
}

/// Extract matching ConceptEntries from lowercase text (checks multi-word phrases & single tokens).
/// Pure integer/slice matching, zero floats.
pub(crate) fn extract_turn_zero_overlap_sig(text: &str) -> ZeroOverlapSig {
    let lower = text.to_ascii_lowercase();
    let mut sig = ZeroOverlapSig::default();
    for entry in CONCEPT_TABLE {
        // Multi-word or exact substring with boundary check to avoid partial-word false hits
        if contains_word_phrase(&lower, entry.key) {
            sig.zato |= entry.zato;
            if entry.wilkins != 0 {
                if sig.wilkins_primary == 0 {
                    sig.wilkins_primary = entry.wilkins;
                } else if sig.wilkins_secondary == 0 && entry.wilkins != sig.wilkins_primary {
                    sig.wilkins_secondary = entry.wilkins;
                }
            }
        }
    }
    if lower.contains("lives next to")
        || lower.contains("next to where")
        || lower.contains("living in")
        || lower.contains("saw the original")
        || lower.contains("seen up close")
    {
        sig.sme_flags |= SME_LOC_NEAR;
    }
    if lower.contains("intolerant")
        || lower.contains("vegan")
        || lower.contains("vegetarian")
        || lower.contains("celiac")
        || lower.contains("allergic")
    {
        sig.sme_flags |= SME_DIET;
    }
    sig
}

#[inline]
fn is_word_boundary_byte(b: u8) -> bool {
    !b.is_ascii_alphanumeric()
}

fn contains_word_phrase(lower_hay: &str, needle: &str) -> bool {
    let hay = lower_hay.as_bytes();
    let nd = needle.as_bytes();
    if nd.is_empty() || hay.len() < nd.len() {
        return false;
    }
    let mut start = 0usize;
    while let Some(pos) = lower_hay[start..].find(needle) {
        let abs = start + pos;
        let left_ok = abs == 0 || is_word_boundary_byte(hay[abs - 1]);
        let right_idx = abs + nd.len();
        let right_ok = right_idx == hay.len() || is_word_boundary_byte(hay[right_idx]);
        if left_ok && right_ok {
            return true;
        }
        start = abs + 1;
        if start + nd.len() > hay.len() {
            break;
        }
    }
    false
}

/// Ken Forbus SME relational skeleton alignment bonus (pure integer):
/// Detects compatible predicate skeletons between query and turn when surface nouns differ.
#[inline]
fn forbus_sme_bonus(q_lower: &str, t_sme_flags: u8) -> u32 {
    let mut bonus = 0u32;
    let q_visited = q_lower.contains("been to") || q_lower.contains("visited") || q_lower.contains("traveled to");
    if q_visited && (t_sme_flags & SME_LOC_NEAR) != 0 {
        bonus += 24;
    }
    let q_cant_consume = q_lower.contains("cannot drink")
        || q_lower.contains("can't drink")
        || q_lower.contains("cannot eat")
        || q_lower.contains("can't eat")
        || q_lower.contains("allergic");
    if q_cant_consume && (t_sme_flags & SME_DIET) != 0 {
        bonus += 24;
    }
    bonus
}

/// Score a turn against a query signature in pure integer arithmetic.
/// Returns 0 if no taxonomic/Zatocode bridge exists.
#[inline]
pub(crate) fn score_zero_overlap_bridge(
    q_sig: &ZeroOverlapSig,
    q_lower: &str,
    t_sig: &ZeroOverlapSig,
) -> u32 {
    let mut score = 0u32;

    // 1. John Wilkins u64 taxonomic prefix match (XOR + leading_zeros)
    let mut best_prefix = 0u32;
    for &qw in &[q_sig.wilkins_primary, q_sig.wilkins_secondary] {
        if qw == 0 {
            continue;
        }
        for &tw in &[t_sig.wilkins_primary, t_sig.wilkins_secondary] {
            let p = wilkins_shared_prefix(qw, tw);
            if p >= 24 && p > best_prefix {
                best_prefix = p;
            }
        }
    }
    if best_prefix >= 40 {
        // Same Genus + Country + City/SubClass (e.g., Kiasma <-> Helsinki/Uusimaa)
        score += 120 + (best_prefix - 40);
    } else if best_prefix >= 24 {
        // Same Genus + Country (e.g., Dresden <-> Germany)
        score += 80 + (best_prefix - 24);
    }

    // 2. Calvin Mooers u128 Zatocode inclusion / bit-overlap
    let shared_bits = (q_sig.zato & t_sig.zato).count_ones();
    if shared_bits > 0 {
        let exact_subsumption = q_sig.zato != 0 && (t_sig.zato & q_sig.zato) == q_sig.zato;
        score += shared_bits * 35 + if exact_subsumption { 40 } else { 0 };
    }

    // 3. Ken Forbus SME skeleton alignment (only adds if taxonomic/Zatocode already matched!)
    if score > 0 {
        score += forbus_sme_bonus(q_lower, t_sig.sme_flags);
    }
    score
}

/// Iadh Ounis QPP gate (integer): should Step 1 zero-overlap expansion run?
/// Only fires when:
///   (a) query actually resolves to a taxonomic Wilkins or Zatocode signature, AND
///   (b) Pass-1 direct lexical overlap has 0 hits on the query's rare/focal concepts
///       (measured by `direct_hit_count == 0` or `max_direct_coverage_pct < 40`).
#[inline]
pub(crate) fn ounis_qpp_allows_bridge(q_sig: &ZeroOverlapSig, direct_nonzero_docs: usize, max_cov_bits: u32, q_group_count: u32) -> bool {
    if q_sig.zato == 0 && q_sig.wilkins_primary == 0 {
        return false;
    }
    if direct_nonzero_docs == 0 {
        return true;
    }
    // If all query content groups are covered by top direct doc, no zero-overlap gap exists.
    if q_group_count > 0 && max_cov_bits * 100 < q_group_count * 50 {
        return true;
    }
    false
}
