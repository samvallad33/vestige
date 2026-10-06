//! Full test surface for the gate runtime. Every test is deterministic:
//! fixed seeds, no clocks, no env, no runtime RNG (the fuzz fixture is
//! generated from a const-seeded SplitMix64 stream, so it is byte-stable
//! across runs).

use std::collections::HashMap;

use borsh::BorshSerialize;

use crate::inputs::{FORGET_FLOOR_MILLI, compute_inputs};
use crate::log::{EventLog, MemLog, SeqAck};
use crate::policy::{Policy, Rule, WILDCARD_PREFIX, gate_verdict};
use crate::record::{
    ActionKindCode, AlertRecord, DutyKind, EffectRecord, GapDetail, GateEvent, GateRecord,
    LessonAlarmRecord, Propose, RecordKind, Verdict, action_kind,
};
use crate::runtime::GateRuntime;
use crate::{Rejected, admit, sweep};

// ---------- helpers ----------

fn h32(seed: u64) -> [u8; 32] {
    *blake3::hash(&seed.to_le_bytes()).as_bytes()
}

fn write_propose(context: Vec<u64>) -> Propose {
    Propose {
        action_hash: h32(context.first().copied().unwrap_or(0) ^ 0xA5A5),
        action_kind: action_kind::WRITE,
        params_hash: h32(context.first().copied().unwrap_or(0) ^ 0x5A5A),
        context,
    }
}

fn allow_writes(max_blast: u32) -> Policy {
    Policy::new(vec![Rule {
        match_kind: ActionKindCode::WRITE,
        match_params_hash_prefix: WILDCARD_PREFIX,
        max_blast_radius: max_blast,
        forbid_forgotten_lessons: false,
        require_human: false,
        verdict: Verdict::Allow,
    }])
}

fn deny_all() -> Policy {
    Policy::new(vec![Rule {
        match_kind: ActionKindCode::WRITE,
        match_params_hash_prefix: WILDCARD_PREFIX,
        max_blast_radius: u32::MAX,
        forbid_forgotten_lessons: false,
        require_human: false,
        verdict: Verdict::Deny,
    }])
}

fn append_raw<T: BorshSerialize>(log: &mut MemLog, kind: RecordKind, record: &T) -> SeqAck {
    let payload = borsh::to_vec(record).expect("borsh infallible");
    log.append(kind, payload)
}

fn effect_for(propose: &Propose, propose_seq: u64, gate_seq: u64) -> EffectRecord {
    EffectRecord {
        propose_seq,
        gate_seq,
        action_hash: propose.action_hash,
        payload_digest: h32(propose_seq ^ 0xEFFE),
    }
}

// ---------- happy path ----------

#[test]
fn happy_path_propose_gate_allow_effect_admitted() {
    let mut rt = GateRuntime::new(MemLog::new(), allow_writes(50));
    let propose = write_propose(vec![1, 2, 3]);
    let p_ack = rt.commit_propose(propose.clone());
    let g_ack = rt.commit_gate(p_ack.seq).expect("gate commits");

    let stored_gate = rt.log().all()[g_ack.seq as usize]
        .gate()
        .expect("gate record");
    assert_eq!(stored_gate.verdict, Verdict::Allow);
    assert_eq!(stored_gate.propose_seq, p_ack.seq);
    assert_eq!(stored_gate.policy_hash, rt.pinned().policy_hash());

    let effect = effect_for(&propose, p_ack.seq, g_ack.seq);
    let ticket = rt
        .admit(&effect)
        .expect("admission ticket projects the slot");
    let e_ack = rt.commit_effect(effect).expect("admitted");
    assert_eq!(ticket.seq, e_ack.seq);
    assert_eq!(
        rt.log().all().last().expect("tail").kind,
        RecordKind::Effect
    );
    assert_eq!(rt.log().tip(), 3);
}

// ---------- rejections ----------

#[test]
fn no_gate_rejects() {
    let mut rt = GateRuntime::new(MemLog::new(), allow_writes(50));
    let propose = write_propose(vec![9]);
    let p_ack = rt.commit_propose(propose.clone());
    let err = rt
        .commit_effect(effect_for(&propose, p_ack.seq, p_ack.seq + 1))
        .expect_err("no gate before the write");
    assert_eq!(err, Rejected::NoGate);
    assert_eq!(rt.log().tip(), 1, "nothing was appended");
}

#[test]
fn gate_deny_rejects() {
    let mut rt = GateRuntime::new(MemLog::new(), deny_all());
    let propose = write_propose(vec![]);
    let p_ack = rt.commit_propose(propose.clone());
    let g_ack = rt.commit_gate(p_ack.seq).expect("gate commits");
    assert_eq!(
        rt.log().all()[g_ack.seq as usize].gate().unwrap().verdict,
        Verdict::Deny
    );

    let err = rt
        .commit_effect(effect_for(&propose, p_ack.seq, g_ack.seq))
        .expect_err("denied gate rejects the write");
    assert_eq!(err, Rejected::GateDenied);
}

#[test]
fn stale_policy_rejects() {
    let mut rt = GateRuntime::new(MemLog::new(), allow_writes(50));
    let propose = write_propose(vec![5]);
    let p_ack = rt.commit_propose(propose.clone());
    let g_ack = rt.commit_gate(p_ack.seq).expect("gate under policy A");
    let stale_hash = rt.pinned().policy_hash();

    rt.pin_policy(deny_all()); // rotate the pin
    let err = rt
        .commit_effect(effect_for(&propose, p_ack.seq, g_ack.seq))
        .expect_err("stale gate rejects the write");
    assert_eq!(err, Rejected::GateStale(stale_hash));
}

#[test]
fn inputs_drift_rejects() {
    // Hand-built log: the stored GATE carries tampered inputs.
    let policy = allow_writes(50);
    let mut log = MemLog::new();
    let propose = write_propose(vec![11]);
    let p_ack = append_raw(&mut log, RecordKind::Propose, &propose);

    let mut tampered = compute_inputs(&log, p_ack.seq).expect("real inputs");
    tampered.live_facts_digest[0] ^= 0xFF;
    let gate = GateRecord {
        propose_seq: p_ack.seq,
        verdict: Verdict::Allow,
        policy_hash: policy.policy_hash(),
        inputs: tampered,
    };
    let g_ack = append_raw(&mut log, RecordKind::Gate, &gate);

    let mut rt = GateRuntime::new(log, policy);
    let err = rt
        .commit_effect(effect_for(&propose, p_ack.seq, g_ack.seq))
        .expect_err("drifted inputs reject the write");
    assert_eq!(err, Rejected::InputsDrift);
}

// ---------- forgotten lessons ----------

#[test]
fn forgotten_lesson_alarm_blocks_when_policy_forbids() {
    let mut policy = allow_writes(50);
    policy.rules[0].forbid_forgotten_lessons = true;
    let mut rt = GateRuntime::new(MemLog::new(), policy);

    // Alarm planted BEFORE the proposal, retention below the floor.
    rt.commit_lesson_alarm(LessonAlarmRecord {
        propose_seq: 0,
        lesson_id: 7,
        retention_milli: FORGET_FLOOR_MILLI - 1,
    });
    let propose = write_propose(vec![1]);
    let p_ack = rt.commit_propose(propose.clone());

    let inputs = compute_inputs(rt.log(), p_ack.seq).expect("inputs");
    assert_eq!(inputs.forgotten_lessons, vec![(7, FORGET_FLOOR_MILLI - 1)]);

    let g_ack = rt.commit_gate(p_ack.seq).expect("gate commits");
    assert_eq!(
        rt.log().all()[g_ack.seq as usize].gate().unwrap().verdict,
        Verdict::Deny
    );
    assert_eq!(
        rt.commit_effect(effect_for(&propose, p_ack.seq, g_ack.seq)),
        Err(Rejected::GateDenied),
        "no Allow gate exists, so the effect cannot land"
    );
}

#[test]
fn forbidden_by_alarm_reason_is_reachable() {
    let mut policy = allow_writes(50);
    policy.rules[0].forbid_forgotten_lessons = true;

    // Path A — alarm in the PREFIX (before the propose), plus a forged Allow
    // gate over the real inputs: the live re-evaluation hits the
    // forgotten-lesson veto (inputs carry the lesson).
    let mut log = MemLog::new();
    append_raw(
        &mut log,
        RecordKind::LessonAlarm,
        &LessonAlarmRecord {
            propose_seq: 1,
            lesson_id: 42,
            retention_milli: -1,
        },
    );
    let propose = write_propose(vec![2]);
    let p0 = append_raw(&mut log, RecordKind::Propose, &propose); // seq 1, matches the alarm
    let real = compute_inputs(&log, p0.seq).expect("inputs");
    assert_eq!(real.forgotten_lessons, vec![(42, -1)]);
    let forged_allow = GateRecord {
        propose_seq: p0.seq,
        verdict: Verdict::Allow,
        policy_hash: policy.policy_hash(),
        inputs: real,
    };
    let g0 = append_raw(&mut log, RecordKind::Gate, &forged_allow);
    let mut rt = GateRuntime::new(log, policy.clone());
    assert_eq!(
        rt.commit_effect(effect_for(&propose, p0.seq, g0.seq)),
        Err(Rejected::ForbiddenByAlarm)
    );

    // Path B — legitimate Allow gate, THEN a matching alarm inside the
    // window (gate -> effect): the write is still blocked.
    let mut rt = GateRuntime::new(MemLog::new(), policy);
    let p1 = rt.commit_propose(propose.clone());
    let g1 = rt.commit_gate(p1.seq).expect("gate");
    assert_eq!(
        rt.log().all()[g1.seq as usize].gate().unwrap().verdict,
        Verdict::Allow
    );
    rt.commit_lesson_alarm(LessonAlarmRecord {
        propose_seq: p1.seq,
        lesson_id: 43,
        retention_milli: FORGET_FLOOR_MILLI - 1,
    });
    assert_eq!(
        rt.commit_effect(effect_for(&propose, p1.seq, g1.seq)),
        Err(Rejected::ForbiddenByAlarm)
    );
}

// ---------- blast radius ----------

#[test]
fn blast_radius_over_50_denies() {
    let mut rt = GateRuntime::new(MemLog::new(), allow_writes(50));

    // Closure of 50 direct references: allowed (50 <= 50).
    let propose_ok = write_propose((1..=50).collect());
    let ok_ack = rt.commit_propose(propose_ok.clone());
    rt.commit_gate(ok_ack.seq).expect("gate commits");
    assert_eq!(
        rt.log()
            .all()
            .last()
            .and_then(|e| e.gate())
            .map(|g| g.verdict),
        Some(Verdict::Allow)
    );

    // Closure of 60: the rule no longer matches, default Deny.
    let propose_big = write_propose((1..=60).collect());
    let big_ack = rt.commit_propose(propose_big.clone());
    rt.commit_gate(big_ack.seq).expect("gate commits");
    assert_eq!(
        rt.log()
            .all()
            .last()
            .and_then(|e| e.gate())
            .map(|g| g.verdict),
        Some(Verdict::Deny)
    );

    // Hub-and-spoke: 60 prior proposals each referencing {hub, i}; a new
    // proposal touching only the hub pulls the whole closure in (61, tiers>=1).
    let hub: u64 = 900;
    for i in 0..60u64 {
        rt.commit_propose(write_propose(vec![hub, 1000 + i]));
    }
    let hub_propose = write_propose(vec![hub]);
    let h_ack = rt.commit_propose(hub_propose.clone());
    let inputs = compute_inputs(rt.log(), h_ack.seq).expect("inputs");
    assert!(
        inputs.blast_radius.closure_size >= 61,
        "closure pulls spokes: {}",
        inputs.blast_radius.closure_size
    );
    assert!(inputs.blast_radius.tiers >= 1);
    rt.commit_gate(h_ack.seq).expect("gate commits");
    assert_eq!(
        rt.log()
            .all()
            .last()
            .and_then(|e| e.gate())
            .map(|g| g.verdict),
        Some(Verdict::Deny)
    );
}

// ---------- canaries ----------

#[test]
fn canary_read_alerts_and_holds_descendants() {
    let mut rt = GateRuntime::new(MemLog::new(), allow_writes(50));
    rt.commit_canary(42);

    // A read that touches the canary id trips an ALERT.
    let reader = write_propose(vec![42, 7]);
    let r_ack = rt.commit_propose(reader);
    let tail = rt.log().all().last().expect("tail");
    assert_eq!(tail.kind, RecordKind::Alert);
    let alert = tail.alert().expect("alert record");
    assert_eq!(alert.canary_id, 42);
    assert_eq!(alert.reader_seq, r_ack.seq);

    // Descendants are auto-held: would-be Allow clamps to Hold.
    let clean = write_propose(vec![8]);
    let c_ack = rt.commit_propose(clean.clone());
    let inputs = compute_inputs(rt.log(), c_ack.seq).expect("inputs");
    assert_eq!(inputs.canary_hits, 1);
    let cg_ack = rt.commit_gate(c_ack.seq).expect("gate commits");
    assert_eq!(
        rt.log().all()[cg_ack.seq as usize].gate().unwrap().verdict,
        Verdict::Hold
    );
    assert_eq!(
        rt.commit_effect(effect_for(&clean, c_ack.seq, cg_ack.seq)),
        Err(Rejected::GateDenied),
        "Hold is not an approving verdict"
    );
}

// ---------- re-derivation over a 1000-event deterministic fixture ----------

struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[test]
fn rederive_matches_stored_verdicts_over_1000_event_fixture() {
    // Const-seeded fixture generator: byte-stable, no runtime RNG.
    let mut rng = SplitMix64(0x5354_5241_5441_4745); // "STRATAGE"
    let policy = allow_writes(64);
    let mut rt = GateRuntime::new(MemLog::new(), policy);

    let mut proposes: HashMap<u64, Propose> = HashMap::new();
    let mut order: Vec<u64> = Vec::new();

    while rt.log().tip() < 1000 {
        match rng.next() % 100 {
            0..=39 => {
                let n = (rng.next() % 4) as usize;
                let context: Vec<u64> = (0..n).map(|_| rng.next() % 32).collect();
                let kind = [
                    ActionKindCode::WRITE,
                    ActionKindCode::RETIRE,
                    ActionKindCode::GRANT,
                    ActionKindCode::EFFECT,
                ][(rng.next() % 4) as usize];
                let propose = Propose {
                    action_hash: h32(rng.next()),
                    action_kind: kind,
                    params_hash: h32(rng.next()),
                    context,
                };
                let ack = rt.commit_propose(propose.clone());
                proposes.insert(ack.seq, propose);
                order.push(ack.seq);
            }
            40..=69 => {
                if order.is_empty() {
                    continue;
                }
                let seq = order[(rng.next() as usize) % order.len()];
                let _ = rt.commit_gate(seq);
            }
            70..=89 => {
                if order.is_empty() {
                    continue;
                }
                let seq = order[(rng.next() as usize) % order.len()];
                let Some(propose) = proposes.get(&seq) else {
                    continue;
                };
                // Cite the latest gate when there is one; sometimes cite a
                // stale seq to exercise the rejection paths.
                let gate_seq = rt
                    .latest_gate(seq)
                    .map(|(gs, v)| if v == Verdict::Allow { gs } else { gs ^ 1 })
                    .unwrap_or(seq);
                let _ = rt.commit_effect(effect_for(propose, seq, gate_seq));
            }
            90..=96 => {
                let retention = (rng.next() % 5_200_000_000) as i64 - 200_000_000;
                let _ = rt.commit_lesson_alarm(LessonAlarmRecord {
                    propose_seq: order.first().copied().unwrap_or(0),
                    lesson_id: rng.next() % 16,
                    retention_milli: retention,
                });
            }
            _ => {
                let _ = rt.commit_canary(rng.next() % 64);
            }
        }
    }

    assert!(rt.log().tip() >= 1000);

    // Stored verdicts, in log order.
    let stored: Vec<(u64, Verdict)> = rt
        .log()
        .all()
        .iter()
        .filter(|e| e.kind == RecordKind::Gate)
        .filter_map(|e| e.gate().map(|g| (e.seq, g.verdict)))
        .collect();
    assert!(stored.len() > 50, "fixture produced {} gates", stored.len());

    let rederived = rt.rederive_verdicts().expect("rederive succeeds");
    assert_eq!(
        rederived, stored,
        "re-derived verdicts must match bit-for-bit"
    );

    // Everything the runtime appended as an effect passed admission, so the
    // sweep must report no orphan effects (reads/duty gaps may exist).
    let orphans = sweep(rt.log())
        .into_iter()
        .filter(|g| g.duty == DutyKind::OrphanEffect)
        .count();
    assert_eq!(orphans, 0);
}

// ---------- GAP sweep ----------

/// Log that leaves a hole between every pair of seqs (duty counter test).
struct HoleLog {
    events: Vec<GateEvent>,
    next: u64,
}

impl EventLog for HoleLog {
    fn events_before(&self, bound: u64) -> Vec<GateEvent> {
        self.events
            .iter()
            .take_while(|e| e.seq < bound)
            .cloned()
            .collect()
    }
    fn append(&mut self, kind: RecordKind, payload: Vec<u8>) -> SeqAck {
        let seq = self.next;
        self.next += 2;
        let frame_hash = h32(seq ^ (kind.to_u8() as u64));
        self.events.push(GateEvent { seq, kind, payload });
        SeqAck { seq, frame_hash }
    }
}

#[test]
fn sweep_catches_orphan_effect_reads_and_duty_gaps() {
    // Orphan effect: appended with no gate at all.
    let mut log = MemLog::new();
    let propose = write_propose(vec![999]); // 999 is also a dangling read
    let p_ack = append_raw(&mut log, RecordKind::Propose, &propose);
    append_raw(
        &mut log,
        RecordKind::Effect,
        &effect_for(&propose, p_ack.seq, p_ack.seq + 1),
    );

    let gaps = sweep(&log);
    assert!(
        gaps.iter().any(|g| matches!(
            &g.detail,
            GapDetail::OrphanEffect {
                effect_seq: 1,
                propose_seq: 0,
                reason: 2
            } // NoGate
        )),
        "orphan effect flagged: {gaps:?}"
    );
    assert!(
        gaps.iter().any(|g| matches!(
            &g.detail,
            GapDetail::ReadNoReceipt {
                reader_seq: 0,
                dangling_id: 999
            }
        )),
        "dangling read flagged: {gaps:?}"
    );

    // Duty sequence hole.
    let mut holey = HoleLog {
        events: Vec::new(),
        next: 0,
    };
    holey.append(RecordKind::Canary, vec![]);
    holey.append(RecordKind::Canary, vec![]);
    let gaps = sweep(&holey);
    assert!(
        gaps.iter().any(|g| matches!(
            &g.detail,
            GapDetail::DutySeqGap {
                source: 0,
                expected: 1,
                found: 2
            }
        )),
        "duty gap flagged: {gaps:?}"
    );
}

// ---------- policy VM unit checks ----------

#[test]
fn policy_vm_first_match_wins_default_deny() {
    let inputs = crate::GateInputs {
        live_facts_digest: [0; 32],
        retired_facts_digest: [0; 32],
        blast_radius: crate::BlastRadius {
            closure_size: 10,
            tiers: 1,
        },
        forgotten_lessons: vec![],
        canary_hits: 0,
    };
    // No rules: default Deny.
    assert_eq!(crate::evaluate(&Policy::default(), &inputs), Verdict::Deny);

    // First matching rule wins.
    let policy = Policy::new(vec![
        Rule {
            match_kind: ActionKindCode::WRITE,
            match_params_hash_prefix: WILDCARD_PREFIX,
            max_blast_radius: 5, // does not match (closure 10 > 5)
            forbid_forgotten_lessons: false,
            require_human: false,
            verdict: Verdict::Deny,
        },
        Rule {
            match_kind: ActionKindCode::WRITE,
            match_params_hash_prefix: WILDCARD_PREFIX,
            max_blast_radius: 50,
            forbid_forgotten_lessons: false,
            require_human: false,
            verdict: Verdict::Allow,
        },
    ]);
    assert_eq!(crate::evaluate(&policy, &inputs), Verdict::Allow);

    // Canary clamp: Allow over a tripped prefix becomes Hold.
    let mut tripped = inputs.clone();
    tripped.canary_hits = 3;
    let propose = write_propose(vec![]);
    assert_eq!(gate_verdict(&policy, &propose, &tripped), Verdict::Hold);
    assert_eq!(gate_verdict(&policy, &propose, &inputs), Verdict::Allow);
}

// ---------- wire layout spot checks (layouts documented in record.rs) ----------

#[test]
fn wire_layouts_are_exact() {
    fn len<T: BorshSerialize>(v: &T) -> usize {
        borsh::to_vec(v).expect("borsh infallible").len()
    }
    assert_eq!(len(&write_propose(vec![])), 69); // 32+1+32+4
    assert_eq!(len(&write_propose(vec![1, 2])), 69 + 16);
    assert_eq!(
        len(&effect_for(&write_propose(vec![]), 1, 2)),
        80 // 8+8+32+32
    );
    assert_eq!(
        len(&LessonAlarmRecord {
            propose_seq: 1,
            lesson_id: 2,
            retention_milli: 3
        }),
        24
    );
    assert_eq!(len(&crate::record::CanaryRecord { canary_id: 1 }), 8);
    assert_eq!(
        len(&AlertRecord {
            canary_id: 1,
            reader_seq: 2
        }),
        16
    );
    assert_eq!(len(&allow_writes(50).rules[0]), 16); // 1+8+4+1+1+1
    assert_eq!(len(&allow_writes(50)), 4 + 16); // u32 len + 1 rule
    // GateInputs with no forgotten lessons: 32+32+(4+2)+4+4 = 78.
    let inputs = crate::GateInputs {
        live_facts_digest: [1; 32],
        retired_facts_digest: [2; 32],
        blast_radius: crate::BlastRadius {
            closure_size: 7,
            tiers: 3,
        },
        forgotten_lessons: vec![],
        canary_hits: 9,
    };
    assert_eq!(len(&inputs), 78);
    // GateRecord with those inputs: 8+1+32+78 = 119.
    assert_eq!(
        len(&GateRecord {
            propose_seq: 1,
            verdict: Verdict::Hold,
            policy_hash: [3; 32],
            inputs
        }),
        119
    );

    // Verdict wire codes by declaration order.
    assert_eq!(borsh::to_vec(&Verdict::Allow).unwrap(), vec![0]);
    assert_eq!(borsh::to_vec(&Verdict::Deny).unwrap(), vec![1]);
    assert_eq!(borsh::to_vec(&Verdict::Hold).unwrap(), vec![2]);
    assert_eq!(RecordKind::Propose.to_u8(), 1);
    assert_eq!(RecordKind::Alert.to_u8(), 7);
}

// `admit` never appends: two calls produce identical tickets.
#[test]
fn admit_is_pure() {
    let mut rt = GateRuntime::new(MemLog::new(), allow_writes(50));
    let propose = write_propose(vec![3]);
    let p_ack = rt.commit_propose(propose.clone());
    let g_ack = rt.commit_gate(p_ack.seq).expect("gate");
    let effect = effect_for(&propose, p_ack.seq, g_ack.seq);
    let t1 = admit(rt.log(), &effect, rt.pinned()).expect("ticket");
    let t2 = admit(rt.log(), &effect, rt.pinned()).expect("ticket");
    assert_eq!(t1, t2);
    assert_eq!(rt.log().tip(), 2, "admit appended nothing");
}
