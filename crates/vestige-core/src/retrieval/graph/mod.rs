//! Stage 4: Causal, temporal, taxonomic, and bridge graph structure.

pub(in crate::retrieval) mod ac3;
pub(in crate::retrieval) mod actr;
pub(in crate::retrieval) mod bridge;
pub(in crate::retrieval) mod causal;
pub(in crate::retrieval) mod fellegi_sunter;
pub(in crate::retrieval) mod freeman_brokerage;
pub(in crate::retrieval) mod ghostlink_dgs;
pub(in crate::retrieval) mod lachmann_stemma;
pub(in crate::retrieval) mod locke_ledger;
pub(in crate::retrieval) mod ppr_rwr;
pub(in crate::retrieval) mod ranganathan_pmest;
pub(in crate::retrieval) mod shepard_citations;
pub(in crate::retrieval) mod supersession;
pub(in crate::retrieval) mod swanson_abc;
pub(in crate::retrieval) mod taxa;
pub(in crate::retrieval) mod temporal_allen;
pub(in crate::retrieval) mod torrens;
pub(in crate::retrieval) mod viterbi;

pub(in crate::retrieval) use self::bridge::*;
pub(in crate::retrieval) use self::supersession::*;
