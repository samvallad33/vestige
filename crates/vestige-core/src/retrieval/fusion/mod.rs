//! Stage 5: Rank aggregation, submodular set cover, Kemeny consensus, and PSL/ILP solvers.

pub(in crate::retrieval) mod kemeny;
pub(in crate::retrieval) mod kemeny_mrf;
pub(in crate::retrieval) mod lin_bilmes;
pub(in crate::retrieval) mod psl_ilp;
pub(in crate::retrieval) mod top10_rrf;

pub(in crate::retrieval) use self::kemeny::*;
