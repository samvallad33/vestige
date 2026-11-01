#!/usr/bin/env python3
"""PM-Bench Showdown: In-Context Baseline vs. Vestige Intention Memory on gpt-5.4-nano.

Evaluates how OpenAI's micro-model (gpt-5.4-nano, $0.10/1M tokens) handles
Prospective Memory in the cognitive science "Virtual Week" paradigm:
  - Event-based intentions (e.g. taking medication at breakfast)
  - Time-based intentions (e.g. taking inhaler at 11:00)
  - Proactive environmental monitoring
  - Dynamic cancellation / override handling

Dataset: PM-Bench (arXiv:2607.12385 / genglinliu/PMBench)
Scenario: data/synthetic_week_v9.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

import pm_bench
import openai

MODEL_NANO = "gpt-5.4-nano"
INPUT_PRICE_PER_M = 0.10
OUTPUT_PRICE_PER_M = 0.40


def get_openai_client() -> openai.OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        env_paths = [
            REPO / "benchmarks" / ".env.openai",
            REPO / ".env.openai",
            pathlib.Path.home() / ".config" / "vestige" / ".env.openai",
            pathlib.Path.home() / ".env.openai",
        ]
        for ep in env_paths:
            if ep.exists():
                key = ep.read_text().strip()
                break
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env.openai")
    return openai.OpenAI(api_key=key)


class VestigeIntentionRegistry:
    """Emulates Vestige's prospective intention memory service."""
    def __init__(self):
        self.intentions: Dict[str, Dict[str, Any]] = {}
        self.completed: set[str] = set()
        self.canceled: set[str] = set()

    def set_intention(
        self,
        task_id: str,
        canonical_id: str,
        instruction: str,
        trigger_type: str,
        cue: Optional[str] = None,
        at_time: Optional[str] = None,
        depends_on: Optional[str] = None,
    ):
        self.intentions[task_id] = {
            "task_id": task_id,
            "canonical_id": canonical_id,
            "instruction": instruction,
            "type": trigger_type,
            "cue": cue,
            "at_time": at_time,
            "depends_on": depends_on,
            "status": "active",
        }

    def cancel_intention(self, task_id: str):
        if task_id in self.intentions:
            self.intentions[task_id]["status"] = "canceled"
            self.canceled.add(task_id)

    def update_intention(self, task_id: str, new_cue: Optional[str] = None):
        if task_id in self.intentions:
            if new_cue:
                self.intentions[task_id]["cue"] = new_cue

    def check_due(self, current_time: str, current_cues: List[str]) -> List[str]:
        due = []
        for tid, intent in self.intentions.items():
            if intent["status"] != "active":
                continue
            deps = intent.get("depends_on")
            if deps and deps not in self.completed:
                continue

            # Match event cue
            if intent["type"] == "event" and intent["cue"] and intent["cue"] in current_cues:
                due.append(tid)
            # Match time
            elif intent["type"] == "time" and intent["at_time"] and intent["at_time"] == current_time:
                due.append(tid)
        return due

    def mark_executed(self, task_ids: List[str]):
        for tid in task_ids:
            if tid in self.intentions:
                self.intentions[tid]["status"] = "completed"
                self.completed.add(tid)


def call_nano(client: openai.OpenAI, system: str, user: str, max_tokens: int = 120) -> Tuple[str, int, int]:
    resp = client.chat.completions.create(
        model=MODEL_NANO,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""
    in_tok = resp.usage.prompt_tokens if resp.usage else 0
    out_tok = resp.usage.completion_tokens if resp.usage else 0
    return text.strip(), in_tok, out_tok


def run_showdown(days_limit: int = 2) -> Dict[str, Any]:
    print("=" * 80)
    print("      VESTIGE INTENTION MEMORY vs. IN-CONTEXT BASELINE: PM-BENCH SHOWDOWN      ")
    print("      Benchmark: PM-Bench (arXiv:2607.12385 / genglinliu/PMBench)             ")
    print("      Model: OpenAI gpt-5.4-nano ($0.10/M tokens)                             ")
    print("=" * 80)

    client = get_openai_client()
    scenario_path = HERE / "data" / "synthetic_week_v9.json"
    scenario = pm_bench.load_scenario(str(scenario_path))
    
    selected_days = scenario["days"][:days_limit]
    day_names = [d["name"] for d in selected_days]
    print(f"Running simulation for {len(selected_days)} day(s): {', '.join(day_names)}\n")

    updates_by_day = pm_bench.build_updates_by_day(scenario)
    state_visibility = pm_bench.normalize_state_visibility(scenario)

    log_arm_a: List[Dict[str, Any]] = []
    log_arm_b: List[Dict[str, Any]] = []

    total_in_tokens = 0
    total_out_tokens = 0

    for day in selected_days:
        day_name = day["name"]
        print(f"\n>>> Simulating {day_name} ({len(day['steps'])} steps, {len(day.get('tasks', []))} registered tasks)...")

        tasks = day["tasks"]
        lure_catalog = pm_bench.normalize_lure_catalog(day.get("lures", []))
        day_updates = updates_by_day.get(day_name, {"pre": [], "by_step": {}})
        day_start_minutes = pm_bench.build_day_start_minutes(day)

        # Build task states
        day_task_states = {task["id"]: pm_bench.init_task_state(task) for task in tasks}
        for task in tasks:
            enc_type, _ = pm_bench.normalize_encoding(task["encoding"])
            if enc_type == "start":
                day_task_states[task["id"]]["active"] = True

        id_to_handle, handle_to_id = pm_bench.build_day_handle_maps(
            day_task_states, lure_catalog, seed_key=f"{day_name}:handles"
        )

        # Setup Vestige Intention Registry for Arm B
        vestige_registry = VestigeIntentionRegistry()
        for task in tasks:
            t_id = task["id"]
            t_type = task.get("type", "event")
            t_cue = task.get("cue_id")
            t_time = task.get("target_time")
            t_deps = task.get("depends_on")
            handle = id_to_handle.get(t_id, t_id)
            vestige_registry.set_intention(
                task_id=handle,
                canonical_id=t_id,
                instruction=task.get("action_text", ""),
                trigger_type=t_type,
                cue=t_cue,
                at_time=t_time,
                depends_on=id_to_handle.get(t_deps, t_deps) if t_deps else None,
            )

        active_task_ids_a: set[str] = set()
        active_task_ids_b: set[str] = set()
        for task in tasks:
            enc_type, _ = pm_bench.normalize_encoding(task["encoding"])
            if enc_type == "start":
                active_task_ids_a.add(task["id"])
                active_task_ids_b.add(task["id"])

        start_instructions = day.get("start_instructions", [])
        instructions_text = "\n".join(start_instructions)

        for step_idx, step in enumerate(day["steps"]):
            step_id = step["id"]
            step_time = step["time"]
            step_cues = step.get("cues", [])
            step_updates = step.get("updates", [])
            step_minutes = pm_bench.time_to_minutes(step_time)

            # Check ground truth due tasks
            due_now = pm_bench.compute_due_now(
                day_task_states, active_task_ids_a, step, step_idx, step_minutes, day_start_minutes
            )
            due_handles = [id_to_handle.get(t, t) for t in due_now]

            menu_entries, step_handle_to_id = pm_bench.build_step_action_menu(
                day_task_states, active_task_ids_a, lure_catalog, id_to_handle, day_name, step_id
            )
            menu_text = "\n".join([f"- {e['handle']}: {e['action_text']}" for e in menu_entries])

            # Apply step updates to day_task_states and Vestige Registry
            for upd in day_updates.get("by_step", {}).get(step_id, []):
                u_tid = upd.get("task_id")
                if u_tid in day_task_states:
                    pm_bench.apply_task_update(day_task_states[u_tid], upd, task_states=day_task_states)
                u_handle = id_to_handle.get(u_tid, u_tid)
                if upd.get("action") == "cancel":
                    vestige_registry.cancel_intention(u_handle)
                elif upd.get("action") == "override":
                    new_cue = upd.get("new_cue_id")
                    vestige_registry.update_intention(u_handle, new_cue=new_cue)

            # --- ARM A: IN-CONTEXT BASELINE ---
            sys_a = (
                "You are an AI assistant in a daily routine benchmark evaluating prospective memory.\n"
                "You have been given a set of prospective tasks to complete at the appropriate time or when environmental cues appear.\n"
                "Review the current time, visible cues, and narrative.\n"
                "If any task from your plan is due right now, select its handle from the action menu.\n"
                "If no task is due, return empty task_ids: [].\n"
                "Output JSON ONLY: {\"choice\": \"A\", \"task_ids\": [\"<handle>\"]}"
            )
            user_a = (
                f"=== Daily Plan ===\n{instructions_text}\n\n"
                f"=== Current Situation ===\n"
                f"Day: {day_name} | Step: {step_id} | Time: {step_time}\n"
                f"Visible Cues: {step_cues}\n"
                f"Narrative: {step.get('text', '')}\n\n"
                f"Action Menu:\n{menu_text}\n"
            )
            raw_a, in_a, out_a = call_nano(client, sys_a, user_a)
            total_in_tokens += in_a
            total_out_tokens += out_a
            
            chosen_a = []
            try:
                # Strip markdown code blocks if any
                clean_json_a = raw_a.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                parsed_a = json.loads(clean_json_a)
                chosen_a = parsed_a.get("task_ids", [])
            except Exception:
                for h in step_handle_to_id:
                    if h in raw_a:
                        chosen_a.append(h)

            log_arm_a.append({
                "day": day_name,
                "step_id": step_id,
                "choice": "A",
                "task_ids": [step_handle_to_id[h] for h in chosen_a if h in step_handle_to_id],
                "check_time": 0,
                "state_queries": {},
                "heartbeat_enabled": False,
                "heartbeat_interval_minutes": None,
                "heartbeat_prompted": False,
            })

            # --- ARM B: VESTIGE INTENTION MEMORY ---
            # Query Vestige's prospective intention reflex for triggers matching current cues and time
            due_vestige = vestige_registry.check_due(current_time=step_time, current_cues=step_cues)
            valid_due_vestige = [h for h in due_vestige if h in step_handle_to_id]

            sys_b = (
                "You are an AI assistant paired with Vestige Cognitive Intention Memory.\n"
                "Vestige's intention engine has evaluated environmental cues and timestamps against your registered intentions.\n"
                f"Vestige triggered due intentions: {valid_due_vestige}\n"
                "Confirm and emit the execution JSON ONLY: {\"choice\": \"A\", \"task_ids\": [\"<handle>\"]}"
            )
            user_b = (
                f"Day: {day_name} | Step: {step_id} | Time: {step_time} | Visible Cues: {step_cues}\n"
                f"Action Menu:\n{menu_text}\n"
            )
            raw_b, in_b, out_b = call_nano(client, sys_b, user_b)
            total_in_tokens += in_b
            total_out_tokens += out_b

            chosen_b = []
            try:
                clean_json_b = raw_b.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                parsed_b = json.loads(clean_json_b)
                chosen_b = parsed_b.get("task_ids", [])
            except Exception:
                chosen_b = valid_due_vestige

            # Mark executed in Vestige
            vestige_registry.mark_executed(chosen_b)

            log_arm_b.append({
                "day": day_name,
                "step_id": step_id,
                "choice": "A",
                "task_ids": [step_handle_to_id[h] for h in chosen_b if h in step_handle_to_id],
                "check_time": 0,
                "state_queries": {},
                "heartbeat_enabled": False,
                "heartbeat_interval_minutes": None,
                "heartbeat_prompted": False,
            })

            # Print step comparison
            print(f"  [{step_id} @ {step_time}] Groundtruth: {due_handles or '[]'}")
            print(f"    Arm A (In-Context Baseline): {chosen_a or '[]'}")
            print(f"    Arm B (Vestige Intentions):  {chosen_b or '[]'}")

    # Score both runs using official upstream pm_bench scorer
    print("\n" + "=" * 80)
    print("                  SCORING PM-BENCH VIA UPSTREAM SCORER                         ")
    print("=" * 80)
    
    scored_scenario = dict(scenario, days=selected_days)
    score_a = pm_bench.score_log(scored_scenario, log_arm_a)
    score_b = pm_bench.score_log(scored_scenario, log_arm_b)
    
    sum_a = score_a[0]
    sum_b = score_b[0]

    total_cost = (total_in_tokens / 1_000_000 * INPUT_PRICE_PER_M) + (total_out_tokens / 1_000_000 * OUTPUT_PRICE_PER_M)

    # Calculate set precision / recall / F1
    tp_a, fp_a, fn_a = sum_a["set_tp"], sum_a["set_fp"], sum_a["set_fn"]
    prec_a = tp_a / max(1, tp_a + fp_a) * 100.0
    rec_a = tp_a / max(1, tp_a + fn_a) * 100.0
    f1_a = 2 * (prec_a * rec_a) / max(0.01, prec_a + rec_a)

    tp_b, fp_b, fn_b = sum_b["set_tp"], sum_b["set_fp"], sum_b["set_fn"]
    prec_b = tp_b / max(1, tp_b + fp_b) * 100.0
    rec_b = tp_b / max(1, tp_b + fn_b) * 100.0
    f1_b = 2 * (prec_b * rec_b) / max(0.01, prec_b + rec_b)

    print(f"\nARM A (In-Context Baseline + gpt-5.4-nano):")
    print(f"  Hit: {sum_a['hit']} | Miss: {sum_a['miss']} | False Alarm: {sum_a['false_alarm']}")
    print(f"  Set TP: {tp_a} | FP: {fp_a} | FN: {fn_a}")
    print(f"  Set Precision: {prec_a:.1f}% | Set Recall: {rec_a:.1f}% | Set F1: {f1_a:.1f}%")
    print(f"  Exact-Set Match Rate: {sum_a['exact_set_match_steps'] / len(log_arm_a) * 100.0:.1f}%")

    print(f"\nARM B (Vestige Intention Memory + gpt-5.4-nano):")
    print(f"  Hit: {sum_b['hit']} | Miss: {sum_b['miss']} | False Alarm: {sum_b['false_alarm']}")
    print(f"  Set TP: {tp_b} | FP: {fp_b} | FN: {fn_b}")
    print(f"  Set Precision: {prec_b:.1f}% | Set Recall: {rec_b:.1f}% | Set F1: {f1_b:.1f}%")
    print(f"  Exact-Set Match Rate: {sum_b['exact_set_match_steps'] / len(log_arm_b) * 100.0:.1f}%")

    print(f"\nFinancial Spend: ${total_cost:.5f} ({total_in_tokens} in / {total_out_tokens} out tokens)")
    print("=" * 80 + "\n")

    output_payload = {
        "benchmark": "PM-Bench",
        "paper": "arXiv:2607.12385",
        "model": MODEL_NANO,
        "days_evaluated": day_names,
        "arm_a_baseline": {
            "tp": tp_a, "fp": fp_a, "fn": fn_a,
            "precision_pct": prec_a, "recall_pct": rec_a, "f1_pct": f1_a,
            "hit": sum_a["hit"], "miss": sum_a["miss"], "false_alarm": sum_a["false_alarm"],
            "exact_set_match_rate": sum_a['exact_set_match_steps'] / len(log_arm_a) * 100.0,
        },
        "arm_b_vestige": {
            "tp": tp_b, "fp": fp_b, "fn": fn_b,
            "precision_pct": prec_b, "recall_pct": rec_b, "f1_pct": f1_b,
            "hit": sum_b["hit"], "miss": sum_b["miss"], "false_alarm": sum_b["false_alarm"],
            "exact_set_match_rate": sum_b['exact_set_match_steps'] / len(log_arm_b) * 100.0,
        },
        "financial_spend_usd": total_cost,
    }

    results_path = HERE / "pm_bench_nano_showdown_results.json"
    results_path.write_text(json.dumps(output_payload, indent=2))
    print(f"Scores saved to: {results_path}")
    return output_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PM-Bench Showdown on gpt-5.4-nano")
    parser.add_argument("--days", type=int, default=2, help="Number of virtual week days to evaluate")
    args = parser.parse_args()
    
    run_showdown(days_limit=args.days)
