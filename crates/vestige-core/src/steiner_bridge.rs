//! Steiner bridges.
//!
//! After the RSB head is frozen, the in-window turns are terminals on the
//! structural graph. A minimum Steiner tree (Dreyfus-Wagner when the
//! component fits the guard, otherwise the Kou-Markowsky-Berman
//! distance-network 2-approximation) connects them. Non-terminal nodes of
//! that tree are bridge turns. They are drawn from candidates below the
//! window and seated in slots 1..k-1. Slot 0 is not a candidate.
//! A length-2 candidate is seated only when it is the only turn adjacent
//! to that disconnected terminal pair (`STEINER_UNIQUE_PAIR`). An index
//! is not a vote.
//!
//! Edges are unweighted. Two turns are adjacent when any one of these holds:
//! a stored `policy_backlinks`, `supersedes_turns`, or `exact_edges` link,
//! the RSB antecedent, an equal session field, an equal nonzero session
//! time, or one exact shared token (capitalized name, technical handle, or
//! digit) that survived the guard.
//!
//! Sources, read before this file was written:
//! - S. E. Dreyfus and R. A. Wagner, "The Steiner problem in graphs,"
//!   Networks 1(3), 1971. Exact DP, O(3^t n + 2^t n^2).
//! - L. Kou, G. Markowsky, and L. Berman, "A fast algorithm for Steiner
//!   trees," Acta Informatica 15, 1981. Distance-network heuristic,
//!   ratio 2(1-1/l).
//! - K. Mehlhorn, "A faster approximation algorithm for the Steiner problem
//!   in graphs," Information Processing Letters 27(3), 1988. Voronoi
//!   speedup of the same heuristic. The node cap makes the exact closure
//!   cheaper than the Voronoi construction, so the fallback is Kou et al.
//!   with the true metric closure, which dominates Mehlhorn's closure.
//! - H. Takahashi and A. Matsuyama, "An approximate solution for the Steiner
//!   problem in graphs," Mathematica Japonica 24, 1980. The other classical
//!   2-approximation. Not used. The assigned fallback is Mehlhorn/KMB.
//! - G. Bhalotia, A. Hulgeri, C. Nakhe, S. Chakrabarti, and S. Sudarshan,
//!   "Keyword searching and browsing in databases using BANKS," ICDE 2002.
//!   Steiner trees whose terminals are the hits and whose internal nodes
//!   are the connecting records.
//! - B. Ding, J. X. Yu, S. Wang, L. Qin, X. Zhang, and X. Lin, "Finding
//!   top-k min-cost connected trees in databases," ICDE 2007 (DPBF). The
//!   same subset DP as Dreyfus-Wagner, on a relationship graph.
//! - G. Kasneci, M. Ramanath, M. Sozio, F. M. Suchanek, and G. Weikum,
//!   "STAR: Steiner-tree approximation in relationship graphs," ICDE 2009.
//! - D. P. Williamson and D. B. Shmoys, The Design of Approximation
//!   Algorithms, Cambridge, 2011, chapter 2. The metric-closure MST is the
//!   backbone of the KMB guarantee.

#[path = "steiner_bridge_guard.rs"]
mod guard;

use guard::{
    STEINER_BELOW_CAP, STEINER_DIGIT_MIN_CHARS, STEINER_EXACT_MAX_NODES, STEINER_EXACT_MAX_TERMINALS,
    STEINER_HUB_DF_MAX, STEINER_KEEP_SOLE_SESSION, STEINER_MAX_BRIDGES, STEINER_NEW_SESSION,
    STEINER_TOKEN_DF_MIN, STEINER_UNIQUE_PAIR,
};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

const INF: u32 = 1_000_000_000;

static FIRES: AtomicU64 = AtomicU64::new(0);

pub(super) fn fire_count() -> u64 {
    FIRES.load(Ordering::Relaxed)
}

pub(super) fn collect_bridge_tokens(
    vocab: &super::Vocab,
    turns: &[super::TurnIx],
    nicks: &super::SpeakerNickIndex,
) -> Vec<Vec<u32>> {
    let speakers = speaker_identities(turns, nicks);
    let mut raw: Vec<Vec<u32>> = Vec::with_capacity(turns.len());
    let mut df: HashMap<u32, u32> = HashMap::new();
    for turn in turns {
        let mut ids: Vec<u32> = Vec::new();
        for &id in &turn.entities {
            if token_kind(vocab, id, &speakers).is_some() {
                ids.push(id);
            }
        }
        for &id in &turn.content {
            if token_kind(vocab, id, &speakers) == Some(Kind::Digit) {
                ids.push(id);
            }
        }
        ids.sort_unstable();
        ids.dedup();
        for &id in &ids {
            *df.entry(id).or_insert(0) += 1;
        }
        raw.push(ids);
    }
    for ids in &mut raw {
        ids.retain(|&id| df_keeps(vocab, id, df.get(&id).copied().unwrap_or(0), &speakers));
    }
    raw
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Name,
    Handle,
    Digit,
}

fn speaker_identities(turns: &[super::TurnIx], nicks: &super::SpeakerNickIndex) -> HashSet<String> {
    let mut speakers = HashSet::new();
    for turn in turns {
        if turn.speaker.is_empty() || turn.speaker == "user" || turn.speaker == "assistant" {
            continue;
        }
        for w in super::raw_tokens(&turn.speaker) {
            if let Some(set) = nicks.witnessed.get(&w) {
                for nick in set {
                    speakers.insert(nick.clone());
                }
            }
            speakers.insert(w);
        }
    }
    speakers
}

fn token_kind(vocab: &super::Vocab, id: u32, speakers: &HashSet<String>) -> Option<Kind> {
    let name = vocab.names.get(id as usize)?;
    if let Some(body) = name.strip_prefix("\u{1}ent:") {
        if body.is_empty() || speaker_identity(body, speakers) {
            return None;
        }
        return Some(Kind::Name);
    }
    if name.starts_with("\u{1}h:") {
        return Some(Kind::Handle);
    }
    if name.len() >= STEINER_DIGIT_MIN_CHARS && name.bytes().all(|b| b.is_ascii_digit()) {
        return Some(Kind::Digit);
    }
    None
}

fn speaker_identity(body: &str, speakers: &HashSet<String>) -> bool {
    let mut any = false;
    for part in body.split('_') {
        if part.is_empty() {
            continue;
        }
        any = true;
        if !speakers.contains(part) {
            return false;
        }
    }
    any
}

fn df_keeps(vocab: &super::Vocab, id: u32, df: u32, speakers: &HashSet<String>) -> bool {
    if df < STEINER_TOKEN_DF_MIN {
        return false;
    }
    match token_kind(vocab, id, speakers) {
        Some(Kind::Name) => true,
        Some(Kind::Handle) | Some(Kind::Digit) => df <= STEINER_HUB_DF_MAX,
        None => false,
    }
}

/// Seat bridge turns. Returns whether the window changed.
pub(super) fn apply(ix: &super::Index, list: &mut Vec<usize>, k: usize, t_scene: &[f32]) -> bool {
    if k < 2 || list.len() < 2 {
        return false;
    }
    let head = list[0];
    let bridges = bridge_turns(ix, list, t_scene);
    if bridges.is_empty() {
        return false;
    }
    let before = list.clone();
    for (turn, seat) in bridges.into_iter().take(STEINER_MAX_BRIDGES) {
        if list.contains(&turn) {
            continue;
        }
        let seat = seat.max(1).min(k - 1).min(list.len());
        if seat == 0 {
            continue;
        }
        // Inserting before index 5 shifts the old index 4 out of the top 5.
        // Keep that turn when it is the only top-5 member of its session.
        if STEINER_KEEP_SOLE_SESSION && seat < 5 && list.len() >= 5 {
            let leaving = list[4];
            let sess = ix.turns[leaving].session;
            let duplicated = list[..4].iter().any(|&t| ix.turns[t].session == sess);
            if !duplicated {
                continue;
            }
        }
        list.insert(seat, turn);
        if list.len() > k {
            list.truncate(k);
        }
    }
    if list.first().copied() != Some(head) {
        *list = before;
        return false;
    }
    if *list == before {
        return false;
    }
    FIRES.fetch_add(1, Ordering::Relaxed);
    true
}

fn bridge_turns(ix: &super::Index, window: &[usize], t_scene: &[f32]) -> Vec<(usize, usize)> {
    let nturns = ix.turns.len();
    if nturns == 0 {
        return Vec::new();
    }
    let terminals: Vec<usize> = window
        .iter()
        .copied()
        .filter(|&t| t < nturns)
        .collect();
    if terminals.len() < 2 {
        return Vec::new();
    }
    let (stored, postings) = structural_index(ix);
    let mut in_window = vec![false; nturns];
    for &t in &terminals {
        in_window[t] = true;
    }
    let term_link = terminal_links(ix, &terminals, &stored);
    // Unit edge costs. A connected terminal graph already has a tree of
    // cost |T|-1, and no non-terminal can make that cheaper.
    if terminal_graph_connected(&term_link) {
        return Vec::new();
    }
    let mut seen = in_window.clone();
    let mut hop1: Vec<usize> = Vec::new();
    for &term in &terminals {
        for &nb in &stored[term] {
            if nb < nturns && !seen[nb] {
                seen[nb] = true;
                hop1.push(nb);
            }
        }
        for &tok in &ix.bridge_tokens[term] {
            if let Some(list) = postings.get(&tok) {
                for &nb in list {
                    if nb < nturns && !seen[nb] {
                        seen[nb] = true;
                        hop1.push(nb);
                    }
                }
            }
        }
    }
    let mut group1: Vec<(bool, usize, Vec<usize>)> = Vec::new();
    let mut group2: Vec<usize> = Vec::new();
    for nb in hop1 {
        if t_scene.get(nb).copied().unwrap_or(0.0) <= 0.0 {
            continue;
        }
        let mut touch: Vec<usize> = Vec::new();
        for (i, &term) in terminals.iter().enumerate() {
            // Session and time link terminals to each other. They do not
            // make a co-session turn a bridge. A touch is a stored edge
            // or an exact shared token.
            if signal_adjacent(ix, &stored, nb, term) {
                touch.push(i);
            }
        }
        let useful = touch.iter().any(|&a| touch.iter().any(|&b| a < b && !term_link[a][b]));
        let new_session = !STEINER_NEW_SESSION
            || terminals.iter().all(|&term| ix.turns[term].session != ix.turns[nb].session);
        if useful && new_session {
            let touches_head = touch.contains(&0);
            group1.push((!touches_head, nb, touch));
        } else if !touch.is_empty() {
            group2.push(nb);
        }
    }
    group1.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    if STEINER_UNIQUE_PAIR {
        let forced = unique_pair_mask(&group1, &term_link);
        group1 = group1
            .into_iter()
            .enumerate()
            .filter(|(i, _)| forced[*i])
            .map(|(_, row)| row)
            .collect();
        // A length-3 waypoint is not a pair bridge. It stays out of the
        // graph so it cannot win an index tie against the unique pair.
        group2.clear();
    }
    group2.sort_unstable();
    let mut extras: Vec<usize> = Vec::new();
    for (_, nb, _) in group1 {
        if extras.len() == STEINER_BELOW_CAP {
            break;
        }
        extras.push(nb);
    }
    for nb in group2 {
        if extras.len() == STEINER_BELOW_CAP {
            break;
        }
        extras.push(nb);
    }
    if extras.is_empty() {
        return Vec::new();
    }
    let mut nodes = terminals.clone();
    nodes.extend(extras);
    let m = nodes.len();
    let tcount = terminals.len();
    let mut adj = vec![Vec::<usize>::new(); m];
    for i in 0..m {
        for j in (i + 1)..m {
            if adjacent(ix, &stored, nodes[i], nodes[j]) {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    let mut is_term = vec![false; m];
    for i in 0..tcount {
        is_term[i] = true;
    }
    let mut tree = vec![Vec::<usize>::new(); m];
    for comp in components(&adj) {
        let terms: Vec<usize> = comp.iter().copied().filter(|&i| is_term[i]).collect();
        if terms.len() < 2 {
            continue;
        }
        let edges = if terms.len() <= STEINER_EXACT_MAX_TERMINALS && comp.len() <= STEINER_EXACT_MAX_NODES {
            dreyfus_wagner_edges(&comp, &terms, &adj)
        } else {
            kmb_edges(&comp, &terms, &adj)
        };
        for (a, b) in edges {
            let (ga, gb) = (comp[a], comp[b]);
            tree[ga].push(gb);
            tree[gb].push(ga);
        }
    }
    for nbrs in &mut tree {
        nbrs.sort_unstable();
        nbrs.dedup();
    }
    let mut steiner: Vec<usize> = Vec::new();
    for i in tcount..m {
        if !tree[i].is_empty() {
            steiner.push(i);
        }
    }
    if steiner.is_empty() {
        return Vec::new();
    }
    let dist_from_head = bfs_dist(&tree, 0);
    steiner.sort_by(|&a, &b| {
        let da = dist_from_head[a];
        let db = dist_from_head[b];
        let disconnected = |d: usize| d == usize::MAX;
        disconnected(da)
            .cmp(&disconnected(db))
            .then(da.cmp(&db))
            .then(nodes[a].cmp(&nodes[b]))
    });
    let mut out = Vec::new();
    for compact in steiner {
        let turn = nodes[compact];
        let earliest = earliest_terminal(&tree, compact, tcount).unwrap_or(0);
        let seat = earliest.saturating_add(1);
        out.push((turn, seat));
        if out.len() == STEINER_MAX_BRIDGES {
            break;
        }
    }
    out
}

fn structural_index(ix: &super::Index) -> (Vec<Vec<usize>>, HashMap<u32, Vec<usize>>) {
    let n = ix.turns.len();
    let mut stored = vec![Vec::<usize>::new(); n];
    for ti in 0..n {
        let mut nxt = Vec::new();
        nxt.extend_from_slice(&ix.turns[ti].policy_backlinks);
        nxt.extend_from_slice(&ix.turns[ti].supersedes_turns);
        for &(_, target) in &ix.turns[ti].exact_edges {
            nxt.push(target);
        }
        if let Some(target) = super::rsb_antecedent(ix, ti) {
            nxt.push(target);
        }
        for target in nxt {
            if target < n && target != ti {
                stored[ti].push(target);
                stored[target].push(ti);
            }
        }
    }
    for row in &mut stored {
        row.sort_unstable();
        row.dedup();
    }
    let mut postings: HashMap<u32, Vec<usize>> = HashMap::new();
    for (ti, toks) in ix.bridge_tokens.iter().enumerate() {
        for &tok in toks {
            postings.entry(tok).or_default().push(ti);
        }
    }
    (stored, postings)
}

fn unique_pair_mask(group1: &[(bool, usize, Vec<usize>)], term_link: &[Vec<bool>]) -> Vec<bool> {
    let n = group1.len();
    let mut forced = vec![false; n];
    for i in 0..n {
        let touch = &group1[i].2;
        for a in 0..touch.len() {
            for b in (a + 1)..touch.len() {
                let (ta, tb) = (touch[a], touch[b]);
                if term_link[ta][tb] {
                    continue;
                }
                let mut support = 0usize;
                for row in group1 {
                    if row.2.contains(&ta) && row.2.contains(&tb) {
                        support += 1;
                    }
                }
                if support == 1 {
                    forced[i] = true;
                }
            }
        }
    }
    forced
}

fn terminal_graph_connected(link: &[Vec<bool>]) -> bool {
    let t = link.len();
    if t == 0 {
        return true;
    }
    let mut seen = vec![false; t];
    let mut stack = vec![0usize];
    seen[0] = true;
    let mut reached = 1usize;
    while let Some(u) = stack.pop() {
        for v in 0..t {
            if link[u][v] && !seen[v] {
                seen[v] = true;
                reached += 1;
                stack.push(v);
            }
        }
    }
    reached == t
}

fn terminal_links(ix: &super::Index, terminals: &[usize], stored: &[Vec<usize>]) -> Vec<Vec<bool>> {
    let t = terminals.len();
    let mut link = vec![vec![false; t]; t];
    for i in 0..t {
        for j in (i + 1)..t {
            if adjacent(ix, stored, terminals[i], terminals[j]) {
                link[i][j] = true;
                link[j][i] = true;
            }
        }
    }
    link
}

fn signal_adjacent(ix: &super::Index, stored: &[Vec<usize>], a: usize, b: usize) -> bool {
    if a == b {
        return false;
    }
    if share(&ix.bridge_tokens[a], &ix.bridge_tokens[b]) {
        return true;
    }
    stored.get(a).is_some_and(|row| row.binary_search(&b).is_ok())
}

fn adjacent(ix: &super::Index, stored: &[Vec<usize>], a: usize, b: usize) -> bool {
    if a == b {
        return false;
    }
    let ta = &ix.turns[a];
    let tb = &ix.turns[b];
    if ta.session == tb.session {
        return true;
    }
    let tsa = ix.session_ts.get(ta.session).copied().unwrap_or(0);
    let tsb = ix.session_ts.get(tb.session).copied().unwrap_or(0);
    if tsa > 0 && tsa == tsb {
        return true;
    }
    if share(&ix.bridge_tokens[a], &ix.bridge_tokens[b]) {
        return true;
    }
    stored.get(a).is_some_and(|row| row.binary_search(&b).is_ok())
}

fn share(a: &[u32], b: &[u32]) -> bool {
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => return true,
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    false
}

fn components(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adj.len();
    let mut seen = vec![false; n];
    let mut out = Vec::new();
    for s in 0..n {
        if seen[s] {
            continue;
        }
        let mut stack = vec![s];
        seen[s] = true;
        let mut members = Vec::new();
        while let Some(u) = stack.pop() {
            members.push(u);
            for &v in &adj[u] {
                if !seen[v] {
                    seen[v] = true;
                    stack.push(v);
                }
            }
        }
        members.sort_unstable();
        out.push(members);
    }
    out
}

fn bfs_dist(tree: &[Vec<usize>], root: usize) -> Vec<usize> {
    let mut dist = vec![usize::MAX; tree.len()];
    if root >= tree.len() || tree[root].is_empty() && root >= tree.len() {
        return dist;
    }
    let mut q = std::collections::VecDeque::new();
    // The root is on the tree when it has a neighbor, or it is a terminal
    // we still measure from. Distance 0 even if it is isolated: callers
    // treat MAX as "not in this tree" only for nodes never reached.
    dist[root] = 0;
    q.push_back(root);
    while let Some(u) = q.pop_front() {
        for &v in &tree[u] {
            if dist[v] == usize::MAX {
                dist[v] = dist[u] + 1;
                q.push_back(v);
            }
        }
    }
    dist
}

fn earliest_terminal(tree: &[Vec<usize>], start: usize, tcount: usize) -> Option<usize> {
    let dist = bfs_dist(tree, start);
    let mut best: Option<usize> = None;
    for i in 0..tcount {
        if dist[i] == usize::MAX {
            continue;
        }
        best = Some(best.map_or(i, |b| b.min(i)));
    }
    best
}

fn dreyfus_wagner_edges(members: &[usize], terms: &[usize], adj: &[Vec<usize>]) -> Vec<(usize, usize)> {
    let (dist, next) = floyd(members, adj);
    let local_of = local_index(members);
    let local_terms: Vec<usize> = terms.iter().map(|&g| local_of[g]).collect();
    dw_edges(&dist, &next, &local_terms)
}

fn kmb_edges(members: &[usize], terms: &[usize], adj: &[Vec<usize>]) -> Vec<(usize, usize)> {
    let (dist, next) = floyd(members, adj);
    let local_of = local_index(members);
    let local_terms: Vec<usize> = terms.iter().map(|&g| local_of[g]).collect();
    kmb_on_metric(&dist, &next, &local_terms)
}

fn local_index(members: &[usize]) -> Vec<usize> {
    let mut n = 0usize;
    for &m in members {
        n = n.max(m + 1);
    }
    let mut at = vec![usize::MAX; n];
    for (i, &m) in members.iter().enumerate() {
        at[m] = i;
    }
    at
}

fn floyd(members: &[usize], adj: &[Vec<usize>]) -> (Vec<Vec<u32>>, Vec<Vec<u16>>) {
    let n = members.len();
    let mut dist = vec![vec![INF; n]; n];
    let mut next = vec![vec![u16::MAX; n]; n];
    for i in 0..n {
        dist[i][i] = 0;
        next[i][i] = i as u16;
    }
    let local = local_index(members);
    for (i, &g) in members.iter().enumerate() {
        for &nb in &adj[g] {
            let j = local[nb];
            if j == usize::MAX || i == j {
                continue;
            }
            if 1 < dist[i][j] {
                dist[i][j] = 1;
                next[i][j] = j as u16;
            }
        }
    }
    for k in 0..n {
        for i in 0..n {
            let dik = dist[i][k];
            if dik >= INF {
                continue;
            }
            for j in 0..n {
                let alt = dik.saturating_add(dist[k][j]);
                if dist[k][j] < INF && alt < dist[i][j] {
                    dist[i][j] = alt;
                    next[i][j] = next[i][k];
                }
            }
        }
    }
    (dist, next)
}

fn dw_edges(dist: &[Vec<u32>], next: &[Vec<u16>], terms: &[usize]) -> Vec<(usize, usize)> {
    let t = terms.len();
    let n = dist.len();
    if t < 2 || t > 16 || (1usize << t) > (1usize << STEINER_EXACT_MAX_TERMINALS) {
        return kmb_on_metric(dist, next, terms);
    }
    let masks = 1usize << t;
    let mut dp = vec![vec![INF; n]; masks];
    // 0 unset, 1 path from a terminal, 2 join of two subtrees, 3 one-edge attachment of a finished tree.
    let mut kind = vec![vec![0u8; n]; masks];
    let mut via = vec![vec![0u16; n]; masks];
    let mut submask = vec![vec![0u32; n]; masks];
    for (bit, &term) in terms.iter().enumerate() {
        let mask = 1usize << bit;
        for v in 0..n {
            dp[mask][v] = dist[term][v];
            kind[mask][v] = 1;
            via[mask][v] = term as u16;
        }
        dp[mask][term] = 0;
        kind[mask][term] = 1;
        via[mask][term] = term as u16;
    }
    for mask in 1..masks {
        if mask.count_ones() < 2 {
            continue;
        }
        let mut sub = (mask - 1) & mask;
        while sub > 0 {
            let other = mask ^ sub;
            if sub < other {
                for v in 0..n {
                    let cost = sat(dp[sub][v], dp[other][v]);
                    if cost < dp[mask][v] {
                        dp[mask][v] = cost;
                        kind[mask][v] = 2;
                        submask[mask][v] = sub as u32;
                    }
                }
            }
            if sub == 0 {
                break;
            }
            sub = (sub - 1) & mask;
        }
        let base = dp[mask].clone();
        let base_kind = kind[mask].clone();
        let base_sub = submask[mask].clone();
        let base_via = via[mask].clone();
        for v in 0..n {
            for u in 0..n {
                if dist[u][v] >= INF || base[u] >= INF {
                    continue;
                }
                let cost = sat(base[u], dist[u][v]);
                if cost < dp[mask][v] {
                    dp[mask][v] = cost;
                    kind[mask][v] = 3;
                    via[mask][v] = u as u16;
                }
            }
        }
        // Restoring the join record at nodes the attachment did not improve
        // keeps reconstruction pointed at the pre-attachment tree.
        for v in 0..n {
            if kind[mask][v] != 3 {
                kind[mask][v] = base_kind[v];
                submask[mask][v] = base_sub[v];
                via[mask][v] = base_via[v];
            }
        }
    }
    let full = masks - 1;
    let mut root: Option<usize> = None;
    let mut best = INF;
    for v in 0..n {
        if dp[full][v] < best {
            best = dp[full][v];
            root = Some(v);
        }
    }
    let Some(root) = root else {
        return Vec::new();
    };
    if best >= INF {
        return Vec::new();
    }
    let mut edges = Vec::new();
    dw_collect(full, root, &kind, &via, &submask, next, &mut edges, 0);
    normalize_edges(&mut edges);
    edges
}

fn dw_collect(
    mask: usize,
    v: usize,
    kind: &[Vec<u8>],
    via: &[Vec<u16>],
    submask: &[Vec<u32>],
    next: &[Vec<u16>],
    edges: &mut Vec<(usize, usize)>,
    depth: usize,
) {
    if depth > kind.len().saturating_add(kind.first().map(|r| r.len()).unwrap_or(0)) || mask == 0 || v >= kind[mask].len() {
        return;
    }
    match kind[mask][v] {
        1 => {
            let src = via[mask][v] as usize;
            path_edges(src, v, next, edges);
        }
        2 => {
            let sub = submask[mask][v] as usize;
            let other = mask ^ sub;
            if sub != 0 && other != 0 && sub < mask {
                dw_collect(sub, v, kind, via, submask, next, edges, depth + 1);
                dw_collect(other, v, kind, via, submask, next, edges, depth + 1);
            }
        }
        3 => {
            let u = via[mask][v] as usize;
            if u != v {
                dw_collect(mask, u, kind, via, submask, next, edges, depth + 1);
                path_edges(u, v, next, edges);
            }
        }
        _ => {}
    }
}

fn kmb_on_metric(dist: &[Vec<u32>], next: &[Vec<u16>], terms: &[usize]) -> Vec<(usize, usize)> {
    let t = terms.len();
    if t < 2 {
        return Vec::new();
    }
    let mut pairs = Vec::new();
    for i in 0..t {
        for j in (i + 1)..t {
            let d = dist[terms[i]][terms[j]];
            if d < INF {
                pairs.push((d, i, j));
            }
        }
    }
    pairs.sort_unstable();
    let mut parent: Vec<usize> = (0..t).collect();
    let mut chosen = Vec::new();
    for &(_, i, j) in &pairs {
        if unite(&mut parent, i, j) {
            chosen.push((i, j));
        }
    }
    let n = dist.len();
    let mut edges = Vec::new();
    for (i, j) in chosen {
        path_edges(terms[i], terms[j], next, &mut edges);
    }
    normalize_edges(&mut edges);
    // Break cycles. Unit cost, so the tie break is the endpoint order.
    let mut uf: Vec<usize> = (0..n).collect();
    let mut tree = Vec::new();
    for &(a, b) in &edges {
        if unite(&mut uf, a, b) {
            tree.push((a, b));
        }
    }
    prune_nonterminal_leaves(&mut tree, terms, n);
    tree
}

fn prune_nonterminal_leaves(edges: &mut Vec<(usize, usize)>, terms: &[usize], n: usize) {
    let mut is_term = vec![false; n];
    for &t in terms {
        if t < n {
            is_term[t] = true;
        }
    }
    loop {
        let mut deg = vec![0usize; n];
        for &(a, b) in edges.iter() {
            if a < n && b < n {
                deg[a] += 1;
                deg[b] += 1;
            }
        }
        let leaves: Vec<usize> = (0..n).filter(|&i| deg[i] <= 1 && !is_term[i] && deg[i] > 0).collect();
        // degree 0 non-terminals are already absent. degree 1 non-terminals are dangling.
        let leaves: Vec<usize> = leaves.into_iter().filter(|&i| deg[i] == 1).collect();
        if leaves.is_empty() {
            break;
        }
        let drop: HashSet<usize> = leaves.into_iter().collect();
        edges.retain(|&(a, b)| !drop.contains(&a) && !drop.contains(&b));
    }
}

fn path_edges(mut u: usize, v: usize, next: &[Vec<u16>], edges: &mut Vec<(usize, usize)>) {
    if u == v || u >= next.len() || v >= next.len() {
        return;
    }
    for _ in 0..next.len() {
        let nxt = next[u][v] as usize;
        if nxt == u16::MAX as usize || nxt >= next.len() || nxt == u {
            break;
        }
        edges.push((u.min(nxt), u.max(nxt)));
        u = nxt;
        if u == v {
            break;
        }
    }
}

fn normalize_edges(edges: &mut Vec<(usize, usize)>) {
    for e in edges.iter_mut() {
        if e.0 > e.1 {
            *e = (e.1, e.0);
        }
    }
    edges.retain(|&(a, b)| a != b);
    edges.sort_unstable();
    edges.dedup();
}

fn sat(a: u32, b: u32) -> u32 {
    if a >= INF || b >= INF {
        INF
    } else {
        a.saturating_add(b)
    }
}

fn unite(parent: &mut [usize], a: usize, b: usize) -> bool {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra == rb {
        return false;
    }
    if ra < rb {
        parent[rb] = ra;
    } else {
        parent[ra] = rb;
    }
    true
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

#[cfg(test)]
mod solver_tests {
    use super::*;

    fn line_graph(n: usize) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); n];
        for i in 0..n - 1 {
            adj[i].push(i + 1);
            adj[i + 1].push(i);
        }
        adj
    }

    #[test]
    fn dreyfus_wagner_and_kmb_agree_on_a_path() {
        let adj = line_graph(5);
        let members: Vec<usize> = (0..5).collect();
        let terms = vec![0usize, 4];
        let (dist, next) = floyd(&members, &adj);
        let dw = dw_edges(&dist, &next, &terms);
        let kmb = kmb_on_metric(&dist, &next, &terms);
        let nodes = |edges: &[(usize, usize)]| {
            let mut s = HashSet::new();
            for &(a, b) in edges {
                s.insert(a);
                s.insert(b);
            }
            s
        };
        assert_eq!(nodes(&dw), nodes(&kmb));
        assert!(nodes(&dw).is_superset(&HashSet::from([1, 2, 3])));
        assert_eq!(dw.len(), 4);
    }

    #[test]
    fn a_direct_edge_adds_no_steiner_node() {
        let mut adj = vec![Vec::new(); 3];
        adj[0].push(2);
        adj[2].push(0);
        adj[0].push(1);
        adj[1].push(0);
        adj[1].push(2);
        adj[2].push(1);
        let members: Vec<usize> = (0..3).collect();
        let terms = vec![0usize, 2];
        let (dist, next) = floyd(&members, &adj);
        let dw = dw_edges(&dist, &next, &terms);
        assert_eq!(dw, vec![(0, 2)]);
    }
}
