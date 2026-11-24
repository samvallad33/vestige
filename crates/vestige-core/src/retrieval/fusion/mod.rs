//! Stage 5: Rank aggregation, submodular set cover, Kemeny consensus, and PSL/ILP solvers.

pub(in crate::retrieval) mod aspect_coverage_q16;
pub(in crate::retrieval) mod hypergraph_mdl_q16;
pub(in crate::retrieval) mod kemeny;
pub(in crate::retrieval) mod kemeny_mrf;
pub(in crate::retrieval) mod lin_bilmes;
pub(in crate::retrieval) mod matroid_intersection_q16;
pub(in crate::retrieval) mod pit_commensuration_q16;
pub(in crate::retrieval) mod psl_ilp;
pub(in crate::retrieval) mod scaldpp_q16;
pub(in crate::retrieval) mod summary_silhouette_q16;
pub(in crate::retrieval) mod top10_rrf;

pub(in crate::retrieval) use self::kemeny::*;
