# Technical Report: Prospective Intention Memory in LLM Agents
**Document ID**: `VESTIGE-TR-2026-08`  
**Date**: October 31, 2026  
**Authors**: Sovereign Architect & Entity002 (Vestige Research)  
**Subject**: Prospective Memory Evaluation on OpenAI `gpt-5.4-nano` via PM-Bench (arXiv:2607.12385)  
**Status**: Deterministic Empirical Evidence / Production Evaluation  

---

## Executive Summary

Prospective memory (PM)—the cognitive capability to encode delayed intentions and execute them upon encountering specific temporal or environmental cues—is a critical failure mode in current autonomous LLM agents. Naive models and in-context scratchpads suffer severe cognitive degradation when balancing routine daily tasks against dynamic environmental triggers, distractors, and cancellation notices.

In this showdown, we evaluate **OpenAI `gpt-5.4-nano`** (\$0.10 / 1M tokens) on the official **PM-Bench "Virtual Week" paradigm** (Liu et al., arXiv:2607.12385). We pit standard in-context memory prompting (**Arm A: In-Context Baseline**) directly against **Vestige Intention Memory** (**Arm B: Vestige Intention Reflex**).

### Key Empirical Results
- **Set Precision**: **77.8%** (Vestige) vs. **54.5%** (In-Context Baseline) — a **+23.3 percentage point improvement**.
- **Exact-Set Match Rate**: **61.5%** (Vestige) vs. **30.8%** (In-Context Baseline) — **2.0x higher exact match fidelity**.
- **False Alarms**: **2** on Vestige vs. **5** on In-Context Baseline — **60.0% reduction in false alarm rate**.
- **Financial Cost**: **$0.00107** for the complete multi-step virtual simulation on `gpt-5.4-nano`.

---

## 1. Experimental Methodology

### 1.1 The PM-Bench Virtual Week Paradigm
PM-Bench simulates an agent's daily trajectory through continuous virtual time, narrative events, and distractor actions. Tasks consist of:
1. **Regular Event Intentions**: Daily recurring tasks triggered by physical cues (e.g. taking medication at breakfast or dinner).
2. **Time-Based Intentions**: Tasks triggered by clock timestamps (e.g. taking an inhaler at 11:00 or 21:00).
3. **Dynamic Overrides & Cancellations**: Runtime changes communicated in environmental narrative (e.g. an appointment portal delay overriding prior plans).
4. **Adversarial Distractor Lures**: Action candidates presented in the step menu designed to trigger premature or incorrect intention execution.

### 1.2 System Comparison
- **Arm A (In-Context Baseline)**: The model is given the day's full loose plan at the start of the day. At each step, the model observes the current narrative, time, and cue channels, and must select actions from an anonymized handle menu (`task_1`..`task_25`).
- **Arm B (Vestige Intention Memory)**: The agent registers delayed intentions into Vestige's unified intention service (`action: "set"` with deterministic `type: "event"` or `type: "time"` triggers). At each step, Vestige evaluates current cues and timestamps, filtering out completed tasks, honoring dependencies, and passing verified due handles to the model.

---

## 2. Quantitative Results & Comparison

| Metric | Arm A: In-Context Baseline | Arm B: Vestige Intention Memory | Delta / Advantage |
|---|:---:|:---:|:---:|
| **Hits (True Positives)** | 6 | **7** | +1 Hit |
| **Misses (False Negatives)** | 6 | **5** | -16.7% Misses |
| **False Alarms (FP)** | 5 | **2** | **-60.0% False Alarms** |
| **Set Precision** | 54.5% | **77.8%** | **+23.3%** |
| **Set Recall** | 50.0% | **58.3%** | **+8.3%** |
| **Set F1 Score** | 52.2% | **66.7%** | **+14.5%** |
| **Exact-Set Match Rate** | 30.8% | **61.5%** | **2.0x Higher** |
| **Cost** | \$0.00054 | \$0.00053 | Total: \$0.00107 |

---

## 3. Cognitive Failure Analysis

### 3.1 Failure Mode 1: Distractor Lure Susceptibility in Arm A
Without an external intention trigger registry, `gpt-5.4-nano` repeatedly falls for lure distractors in the action menu:
- At step `d1_s2` (09:10) and `d1_s3` (10:20), no intentions were due. Arm A fired false alarms (`task_5` and `task_2`), hallucinating that early execution was appropriate.
- Vestige suppressed these lures because its trigger engine matched 0 registered conditions against the step cues.

### 3.2 Failure Mode 2: Overkill and Stale Execution
At step `d1_s6` (12:45), Arm A selected `task_5`, which had already been executed, creating a commission violation. Vestige automatically transitions completed intentions to `status: "completed"`, mathematically preventing double-execution.

---

## 4. Conclusion & Reproduction

Vestige's prospective intention reflex converts erratic in-context memory into deterministic trigger execution. Even OpenAI's smallest model (`gpt-5.4-nano`) doubles its exact-set match rate while cutting false alarms by 60%.

To reproduce:
```bash
sh benchmarks/pm_bench/run-nano-showdown.sh --days 1
```
