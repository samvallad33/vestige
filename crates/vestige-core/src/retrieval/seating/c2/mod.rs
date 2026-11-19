//! C2 seating passes: openers, speakers, ledger steps 31–38, and pools.

pub(in crate::retrieval) mod common;
pub(in crate::retrieval) mod coverage_and_keys;
pub(in crate::retrieval) mod witnesses;
pub(in crate::retrieval) mod answers_and_pref;
pub(in crate::retrieval) mod openers;
pub(in crate::retrieval) mod session_mates;
pub(in crate::retrieval) mod speakers;
pub(in crate::retrieval) mod step31_33_ledger;
pub(in crate::retrieval) mod step34_bertillonage;
pub(in crate::retrieval) mod step35_mill;
pub(in crate::retrieval) mod step36_38_classical;
pub(in crate::retrieval) mod pools;

pub(in crate::retrieval) use self::common::*;
pub(in crate::retrieval) use self::coverage_and_keys::*;
pub(in crate::retrieval) use self::witnesses::*;
pub(in crate::retrieval) use self::answers_and_pref::*;
pub(in crate::retrieval) use self::openers::*;
pub(in crate::retrieval) use self::session_mates::*;
pub(in crate::retrieval) use self::speakers::*;
pub(in crate::retrieval) use self::step31_33_ledger::*;
pub(in crate::retrieval) use self::step34_bertillonage::*;
pub(in crate::retrieval) use self::step35_mill::*;
pub(in crate::retrieval) use self::step36_38_classical::*;
pub(in crate::retrieval) use self::pools::*;
