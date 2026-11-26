//! Stage 5: Rank aggregation, submodular set cover, Kemeny consensus, and PSL/ILP solvers.

pub(in crate::retrieval) mod antimeta_scaldpp_q16;
pub(in crate::retrieval) mod aspect_coverage_q16;
pub(in crate::retrieval) mod emgr;
pub(in crate::retrieval) mod georag_demand_q16;
pub(in crate::retrieval) mod hogbom_clean_vcg_q16;
pub(in crate::retrieval) mod kemeny;
pub(in crate::retrieval) mod kemeny_mrf;
pub(in crate::retrieval) mod lin_bilmes;
pub(in crate::retrieval) mod pit_commensuration_q16;
pub(in crate::retrieval) mod submodular_residual_cover_q16;
pub(in crate::retrieval) mod summary_silhouette_q16;

pub(in crate::retrieval) use self::kemeny::*;
