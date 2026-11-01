#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from datetime import datetime

def time_to_minutes(value):
    parts = value.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def list_state_channels(scenario):
    """Return sorted state channel names, defaulting to ['clock'] when absent."""
    channels = set()
    state_visibility = scenario.get("state_visibility") or {}
    state_channels = scenario.get("state_channels") or {}
    channels.update(state_visibility.keys())
    channels.update(state_channels.keys())
    if "time_visible_by_default" in scenario:
        channels.add("clock")
    if not channels:
        channels.add("clock")
    return sorted(channels)


def normalize_state_visibility(scenario):
    """Return channel -> bool map with default False and legacy clock mapping."""
    channels = list_state_channels(scenario)
    visibility = {channel: False for channel in channels}
    provided = scenario.get("state_visibility")
    if isinstance(provided, dict):
        for channel, visible in provided.items():
            visibility[channel] = bool(visible)
    if "time_visible_by_default" in scenario and "clock" not in (provided or {}):
        visibility["clock"] = bool(scenario.get("time_visible_by_default"))
    return visibility


def normalize_state_channels(scenario):
    """Return channel config map with per-channel defaults applied."""
    channels = list_state_channels(scenario)
    provided = scenario.get("state_channels") or {}
    configs = {}
    for channel in channels:
        config = dict(provided.get(channel, {}))
        if "mode" not in config:
            config["mode"] = "delta"
        configs[channel] = config
    return configs


def normalize_state_event_item(event):
    """Normalize a state event into a response item."""
    if isinstance(event, str):
        return {"id": event, "text": event, "value": None, "meta": {}}
    return {
        "id": event.get("id"),
        "text": event.get("text"),
        "value": event.get("value"),
        "meta": event.get("meta", {}),
    }


def collect_state_events_for_range(steps, start_idx, end_idx, channel):
    """Collect state events for a channel across a step range (inclusive)."""
    items = []
    if start_idx < 0:
        start_idx = 0
    if end_idx < start_idx:
        return items
    for idx in range(start_idx, end_idx + 1):
        step = steps[idx]
        events = step.get("state_events", {}).get(channel, [])
        for event in events:
            items.append(normalize_state_event_item(event))
    return items


def build_clock_item(day_name, step_time, day_start_minutes):
    """Return the unified clock item for the current step."""
    step_minutes = time_to_minutes(step_time)
    stopwatch = step_minutes - day_start_minutes
    return {
        "id": "clock",
        "text": f"Time {step_time} | Stopwatch: {stopwatch} min",
        "value": step_time,
        "meta": {"day": day_name, "stopwatch_min": stopwatch},
    }


def build_state_query_response(channel, items, day_name, step_id):
    """Return the unified query_state response payload."""
    return {
        "channel": channel,
        "items": items,
        "meta": {"day": day_name, "step_id": step_id},
    }


def format_state_query_display(channel, items):
    """Format a human-readable state query response line."""
    if not items:
        return f"State [{channel}]: (no updates)"
    texts = []
    for item in items:
        text = item.get("text") or item.get("id") or "update"
        texts.append(text)
    return f"State [{channel}]: " + " | ".join(texts)


def resolve_state_query_items(
    channel,
    steps,
    step_idx,
    day_name,
    day_start_minutes,
    state_channels,
    last_query_step_by_channel,
    last_snapshot_item_by_channel,
):
    """Return state items for a channel query and update the last query index."""
    if channel == "clock":
        last_query_step_by_channel[channel] = step_idx
        return [build_clock_item(day_name, steps[step_idx]["time"], day_start_minutes)]

    mode = state_channels.get(channel, {}).get("mode", "delta")
    if mode == "snapshot":
        # Snapshot channels return only the current step's state.
        items = collect_state_events_for_range(steps, step_idx, step_idx, channel)
        if items:
            last_snapshot_item_by_channel[channel] = items[-1]
        else:
            cached = last_snapshot_item_by_channel.get(channel)
            if cached:
                items = [cached]
            else:
                default_item = state_channels.get(channel, {}).get("default_item")
                if default_item:
                    normalized = normalize_state_event_item(default_item)
                    last_snapshot_item_by_channel[channel] = normalized
                    items = [normalized]
    else:
        # Delta channels return events since the last query (or day start).
        start_idx = last_query_step_by_channel.get(channel, -1) + 1
        items = collect_state_events_for_range(steps, start_idx, step_idx, channel)
    last_query_step_by_channel[channel] = step_idx
    return items
def load_scenario(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sanitize_model_label(model_label):
    return model_label.replace("/", "-").replace(":", "-").replace(" ", "-")


def resolve_output_path(path, out_dir=None, default_filename=None, required_suffix=None):
    if path is None:
        if default_filename is None:
            return None
        path = default_filename
    if required_suffix and not path.endswith(required_suffix):
        path = f"{path}{required_suffix}"
    if out_dir and not os.path.isabs(path):
        path = os.path.join(out_dir, path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def normalize_log_path(log_path, model_label, out_dir=None):
    if log_path is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        safe_model = sanitize_model_label(model_label)
        return resolve_output_path(
            None,
            out_dir=out_dir or "runs",
            default_filename=f"out-{timestamp}-{safe_model}.jsonl",
            required_suffix=".jsonl",
        )
    return resolve_output_path(log_path, out_dir=out_dir, required_suffix=".jsonl")


RUN_METADATA_RECORD_TYPE = "run_metadata"


def now_utc_iso():
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def format_duration_seconds(value):
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if seconds < 0:
        return "n/a"
    minutes, rem = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours > 0:
        return f"{hours}h {minutes}m {rem:.1f}s"
    if minutes > 0:
        return f"{minutes}m {rem:.1f}s"
    return f"{seconds:.3f}s"


def make_run_metadata(
    mode,
    started_at_utc,
    finished_at_utc,
    duration_seconds,
    entry_count,
    model=None,
    backend=None,
):
    metadata = {
        "record_type": RUN_METADATA_RECORD_TYPE,
        "mode": mode,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "duration_seconds": round(float(duration_seconds), 3),
        "entry_count": int(entry_count),
    }
    if model:
        metadata["model"] = model
    if backend:
        metadata["backend"] = backend
    return metadata


def write_log(path, entries, run_metadata=None):
    with open(path, "w", encoding="utf-8") as handle:
        if run_metadata is not None:
            handle.write(json.dumps(run_metadata) + "\n")
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def estimate_input_tokens(messages):
    serialized = json.dumps(messages, ensure_ascii=True)
    chars = len(serialized)
    return (chars + 3) // 4


def normalize_encoding(value):
    if value == "start":
        return ("start", None)
    if value.startswith("step:"):
        return ("step", value.split(":", 1)[1])
    raise ValueError(f"Unknown encoding format: {value}")


def build_updates_by_day(scenario):
    updates = {}
    for day in scenario.get("days", []):
        updates[day["name"]] = {"pre": [], "by_step": {}}

    for day in scenario.get("days", []):
        day_name = day["name"]
        for step in day.get("steps", []):
            for update in step.get("updates", []):
                target_day = update.get("target_day", day_name)
                if target_day not in updates:
                    continue
                if target_day == day_name:
                    updates[target_day]["by_step"].setdefault(step["id"], []).append(update)
                else:
                    updates[target_day]["pre"].append(update)
    return updates


def init_task_state(task):
    current = {
        "type": task.get("type"),
        "cue_id": task.get("cue_id"),
        "cue_channel": task.get("cue_channel", "narrative"),
        "target_time": task.get("target_time"),
        "window_before": task.get("window_before"),
        "window_after": task.get("window_after"),
        "label": task.get("label"),
        "action_text": task.get("action_text"),
    }
    return {
        "completed": False,
        "completed_at": None,
        "cue_seen": False,
        "cue_step_idx": None,
        "active": False,
        "result": None,
        "task": task,
        "current": current,
        "canceled": False,
        "canceled_by_dependency": False,
        "updated": False,
        "has_update": False,
        "completion_had_required_query": False,
    }


def mark_task_canceled(state, by_type=None, by_regular=None, metrics=None, count_metrics=True):
    if state["canceled"]:
        return
    state["canceled"] = True
    state["result"] = "canceled"
    task_type = state["current"]["type"]
    if by_type and task_type in by_type:
        by_type[task_type]["total"] -= 1
    if by_regular:
        key = "regular" if state["task"]["regular"] else "irregular"
        by_regular[key]["total"] -= 1
    if metrics is not None:
        if count_metrics:
            metrics["canceled_total"] += 1
        if state["task"].get("cross_day"):
            metrics["cross_day_total"] -= 1


def cancel_dependents(task_states, task_id, by_type=None, by_regular=None, metrics=None):
    queue = [task_id]
    seen = set()
    while queue:
        current_id = queue.pop()
        if current_id in seen:
            continue
        seen.add(current_id)
        for dependent_state in task_states.values():
            depends_on = dependent_state["task"].get("depends_on")
            if depends_on != current_id:
                continue
            if dependent_state["canceled"]:
                continue
            dependent_state["canceled_by_dependency"] = True
            mark_task_canceled(
                dependent_state, by_type=by_type, by_regular=by_regular, metrics=metrics, count_metrics=False
            )
            queue.append(dependent_state["task"]["id"])


def apply_task_update(state, update, task_states=None, by_type=None, by_regular=None, metrics=None):
    if state["completed"]:
        return
    action = update.get("action")
    if action not in ("cancel", "reschedule", "override"):
        return
    if action == "cancel":
        mark_task_canceled(state, by_type=by_type, by_regular=by_regular, metrics=metrics, count_metrics=True)
        if task_states is not None:
            cancel_dependents(task_states, state["task"]["id"], by_type=by_type, by_regular=by_regular, metrics=metrics)
        return

    state["updated"] = True
    new_type = update.get("new_type")
    if new_type and new_type != state["current"]["type"]:
        old_type = state["current"]["type"]
        if by_type and old_type in by_type:
            by_type[old_type]["total"] -= 1
        if by_type and new_type in by_type:
            by_type[new_type]["total"] += 1
        state["current"]["type"] = new_type
    if update.get("new_cue_id"):
        state["current"]["cue_id"] = update["new_cue_id"]
        state["cue_seen"] = False
        state["cue_step_idx"] = None
    if update.get("new_target_time"):
        state["current"]["target_time"] = update["new_target_time"]
        state["cue_seen"] = False
        state["cue_step_idx"] = None
    if update.get("new_window_before") is not None:
        state["current"]["window_before"] = update["new_window_before"]
    if update.get("new_window_after") is not None:
        state["current"]["window_after"] = update["new_window_after"]
    if update.get("new_label"):
        state["current"]["label"] = update["new_label"]
    if update.get("new_action_text"):
        state["current"]["action_text"] = update["new_action_text"]

def build_day_index(day):
    step_index = {}
    for idx, step in enumerate(day["steps"]):
        step_index[step["id"]] = idx
    return step_index


def build_day_start_minutes(day):
    if not day["steps"]:
        return 0
    return time_to_minutes(day["steps"][0]["time"])


def is_due_event(task, step):
    """Return True if the task's cue appears for the current step/channel."""
    cue_id = task.get("cue_id")
    cue_channel = task.get("cue_channel", "narrative")
    if cue_channel == "narrative":
        return cue_id in step.get("cues", [])
    events = step.get("state_events", {}).get(cue_channel, [])
    for event in events:
        if isinstance(event, str) and event == cue_id:
            return True
        if isinstance(event, dict) and event.get("id") == cue_id:
            return True
    return False


def is_due_time(task, step):
    return step["time"] == task["target_time"]


def timecheck_status(task, step_minutes, day_start_minutes):
    target_minutes = time_to_minutes(task["target_time"]) - day_start_minutes
    current_stopwatch = step_minutes - day_start_minutes
    window_start = target_minutes - task.get("window_before", 0)
    window_end = target_minutes + task.get("window_after", 0)
    if window_start <= current_stopwatch <= window_end:
        return "on_time"
    if current_stopwatch > window_end:
        return "late"
    return "early"


EVENT_LATE_WINDOW_STEPS = 1
TIME_LATE_WINDOW_MINUTES = 60
DAILY_TASK_HEADER_LINES = [
    "Regular tasks for every day:",
    "- Take antibiotic at breakfast and dinner.",
    "- Take asthma medication at 11:00 and 21:00.",
]
PROCESS_CHANNELS = {"laundry_status", "shipment_status"}
PROCESS_PHASES = {"idle", "in_progress", "complete"}
STEP_LURE_DISPLAY_COUNT = 3
HEARTBEAT_INTERVAL_OPTIONS = (30, 60)
HEARTBEAT_MESSAGE_MODES = ("channel_query", "task_reminder")
MAX_ACTION_TASK_IDS = 8
HEARTBEAT_REMINDER_TIME_HORIZON_MINUTES = 240
HEARTBEAT_REMINDER_OVERDUE_HORIZON_MINUTES = 60


def sentence_case(text):
    """Return sentence-cased text with leading capitalization."""
    if not text:
        return text
    return text[0].upper() + text[1:]


def derive_action_text_from_label(label):
    """Derive a natural action phrase from a task label."""
    text = (label or "").strip().rstrip(".")
    if not text:
        return "Do the task."

    text = re.sub(r"\s+when\s+.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(at|around)\s+\d{1,2}:\d{2}$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+after\s+.*$", "", text, flags=re.IGNORECASE)

    followup_match = re.match(
        r"^Follow up on the (.+?) task$",
        text,
        flags=re.IGNORECASE,
    )
    if followup_match:
        inner = followup_match.group(1).strip()
        lowered = inner.lower()
        pickup_match = re.match(r"^pick up the (.+)$", lowered)
        if pickup_match:
            text = f"Follow up on {pickup_match.group(1)} pickup"
        else:
            text = f"Follow up on {lowered}"

    return sentence_case(f"{text}.")


def task_action_text(task):
    """Resolve human-facing action text for a task."""
    return task.get("action_text") or derive_action_text_from_label(task.get("label"))


def runtime_task_action_text(state):
    """Resolve action text from mutable runtime task state."""
    current_text = state.get("current", {}).get("action_text")
    if current_text:
        return current_text
    current_label = state.get("current", {}).get("label")
    if current_label:
        return derive_action_text_from_label(current_label)
    return task_action_text(state.get("task", {}))


def normalize_lure_item(lure):
    """Normalize lure items from string/dict to a canonical dict."""
    if isinstance(lure, dict):
        lure_id = lure.get("id")
        action_text = lure.get("action_text")
        if not lure_id:
            return None
        if not action_text:
            action_text = sentence_case(lure_id.replace("_", " ") + ".")
        return {"id": lure_id, "action_text": action_text}
    if isinstance(lure, str):
        return {
            "id": lure,
            "action_text": sentence_case(lure.replace("_", " ") + "."),
        }
    return None


def normalize_lure_catalog(lures):
    """Return a deduplicated lure catalog with IDs and action text."""
    catalog = []
    seen = set()
    for lure in lures or []:
        item = normalize_lure_item(lure)
        if not item:
            continue
        lure_id = item["id"]
        if lure_id in seen:
            continue
        seen.add(lure_id)
        catalog.append(item)
    return catalog


def build_day_handle_maps(task_states, lure_catalog, seed_key):
    """Build stable-per-day anonymous handles for tasks and lures."""
    ids = list(task_states.keys()) + [item["id"] for item in lure_catalog]
    unique_ids = sorted(set(ids))
    rng = random.Random(seed_key)
    rng.shuffle(unique_ids)
    id_to_handle = {}
    handle_to_id = {}
    for idx, item_id in enumerate(unique_ids, start=1):
        handle = f"task_{idx}"
        id_to_handle[item_id] = handle
        handle_to_id[handle] = item_id
    return id_to_handle, handle_to_id


def format_action_menu(menu_entries, header="Action menu (optional)"):
    """Format action handles + text for display."""
    if not menu_entries:
        return f"\n{header}: (none)"
    lines = [f"\n{header}:"]
    for entry in menu_entries:
        lines.append(f"- {entry['handle']}: {entry['action_text']}")
    return "\n".join(lines)


def compute_due_now(
    task_states,
    active_task_ids,
    step,
    step_idx,
    step_minutes,
    day_start_minutes,
):
    """Compute due task IDs for the current step and update cue tracking."""
    due_now = set()
    for task_id in active_task_ids:
        state = task_states.get(task_id)
        if not state:
            continue
        if not state["active"] or state["completed"] or state["canceled"]:
            continue
        task = state["task"]
        current = state["current"]
        depends_on = task.get("depends_on")
        dependency_met = True
        if depends_on:
            dependency_state = task_states.get(depends_on)
            dependency_met = bool(dependency_state and dependency_state["completed"])
        if current["type"] == "event" and is_due_event(current, step):
            state["cue_seen"] = True
            if state["cue_step_idx"] is None:
                state["cue_step_idx"] = step_idx
            if dependency_met:
                due_now.add(task_id)
        elif current["type"] == "time" and is_due_time(current, step):
            state["cue_seen"] = True
            if state["cue_step_idx"] is None:
                state["cue_step_idx"] = step_idx
            if dependency_met:
                due_now.add(task_id)
        elif current["type"] == "time_check":
            status = timecheck_status(current, step_minutes, day_start_minutes)
            if status == "on_time" and dependency_met:
                due_now.add(task_id)
    return due_now


def apply_runtime_completions(
    task_states,
    chosen_task_ids,
    due_now,
    step,
    step_idx,
    step_minutes,
    day_start_minutes,
):
    """Mark tasks completed in runtime state so menus can hide completed items."""
    for task_id in chosen_task_ids:
        state = task_states.get(task_id)
        if not state:
            continue
        if state["completed"] or state["canceled"] or not state["active"]:
            continue
        task = state["task"]
        depends_on = task.get("depends_on")
        if depends_on:
            dependency_state = task_states.get(depends_on)
            if not dependency_state or not dependency_state["completed"]:
                continue
        current = state["current"]
        if task_id in due_now:
            state["completed"] = True
            state["completed_at"] = step["id"]
            state["result"] = "hit"
            continue
        if current["type"] == "time_check":
            status = timecheck_status(current, step_minutes, day_start_minutes)
            target_minutes = time_to_minutes(current["target_time"])
            delta = step_minutes - target_minutes
            if status == "late" and 0 < delta <= TIME_LATE_WINDOW_MINUTES:
                state["completed"] = True
                state["completed_at"] = step["id"]
                state["result"] = "late"
        elif current["type"] == "time":
            target_minutes = time_to_minutes(current["target_time"])
            delta = step_minutes - target_minutes
            if 0 < delta <= TIME_LATE_WINDOW_MINUTES:
                state["completed"] = True
                state["completed_at"] = step["id"]
                state["result"] = "late"
        else:
            cue_step_idx = state["cue_step_idx"]
            if cue_step_idx is not None:
                delta_steps = step_idx - cue_step_idx
                if 0 < delta_steps <= EVENT_LATE_WINDOW_STEPS:
                    state["completed"] = True
                    state["completed_at"] = step["id"]
                    state["result"] = "late"


def build_step_action_menu(
    task_states,
    active_task_ids,
    lure_catalog,
    id_to_handle,
    day_name,
    step_id,
    lure_count=STEP_LURE_DISPLAY_COUNT,
):
    """Build a per-step menu: active tasks + sampled lures, ordered randomly.

    Canceled tasks remain visible so selecting them can be scored as a cancellation
    memory failure; completed tasks are hidden.
    """
    task_entries = []
    for task_id in sorted(active_task_ids):
        state = task_states.get(task_id)
        if not state:
            continue
        if not state["active"] or state["completed"]:
            continue
        task_entries.append(
            {
                "id": task_id,
                "handle": id_to_handle[task_id],
                "action_text": runtime_task_action_text(state),
            }
        )

    sample_size = min(lure_count, len(lure_catalog))
    lure_entries = []
    if sample_size > 0:
        lure_rng = random.Random(f"{day_name}:{step_id}:lures")
        for lure in lure_rng.sample(lure_catalog, k=sample_size):
            lure_entries.append(
                {
                    "id": lure["id"],
                    "handle": id_to_handle[lure["id"]],
                    "action_text": lure["action_text"],
                }
            )

    menu_entries = task_entries + lure_entries
    order_rng = random.Random(f"{day_name}:{step_id}:menu")
    order_rng.shuffle(menu_entries)
    handle_to_id = {entry["handle"]: entry["id"] for entry in menu_entries}
    return menu_entries, handle_to_id


def compute_groundtruth_for_day(day, updates_by_day):
    """Compute per-step due-task labels and perfect-play solvability for one day."""
    day_name = day["name"]
    day_start_minutes = build_day_start_minutes(day)
    tasks = day.get("tasks", [])
    lure_catalog = normalize_lure_catalog(day.get("lures", []))
    day_updates = updates_by_day.get(day_name, {"pre": [], "by_step": {}})
    task_states = {task["id"]: init_task_state(task) for task in tasks}
    active_task_ids = set()
    for task in tasks:
        enc_type, _ = normalize_encoding(task["encoding"])
        if enc_type == "start":
            active_task_ids.add(task["id"])
            task_states[task["id"]]["active"] = True
    for update in day_updates.get("pre", []):
        task_id = update.get("task_id")
        if task_id in task_states:
            apply_task_update(task_states[task_id], update, task_states=task_states)

    id_to_handle, _ = build_day_handle_maps(
        task_states,
        lure_catalog,
        seed_key=f"{day_name}:handles",
    )
    step_groundtruth = {}
    missing_menu_by_step = {}

    for step_idx, step in enumerate(day.get("steps", [])):
        for update in day_updates.get("by_step", {}).get(step["id"], []):
            task_id = update.get("task_id")
            if task_id in task_states:
                apply_task_update(task_states[task_id], update, task_states=task_states)
        for task in tasks:
            enc_type, enc_step = normalize_encoding(task["encoding"])
            if enc_type == "step" and enc_step == step["id"]:
                active_task_ids.add(task["id"])
                task_states[task["id"]]["active"] = True

        step_minutes = time_to_minutes(step["time"])
        due_now = compute_due_now(
            task_states,
            active_task_ids,
            step,
            step_idx,
            step_minutes,
            day_start_minutes,
        )
        menu_entries, _ = build_step_action_menu(
            task_states,
            active_task_ids,
            lure_catalog,
            id_to_handle,
            day_name,
            step["id"],
        )
        menu_ids = {entry["id"] for entry in menu_entries}
        missing = sorted(task_id for task_id in due_now if task_id not in menu_ids)
        if missing:
            missing_menu_by_step[step["id"]] = missing

        actions = [
            {"id": task_id, "task_handle": id_to_handle[task_id]}
            for task_id in sorted(due_now, key=lambda item: id_to_handle[item])
        ]
        if actions:
            step_groundtruth[step["id"]] = {"status": "due", "actions": actions}
        else:
            step_groundtruth[step["id"]] = {"status": "none", "actions": []}

        apply_runtime_completions(
            task_states,
            sorted(due_now),
            due_now,
            step,
            step_idx,
            step_minutes,
            day_start_minutes,
        )

    unresolved_task_ids = []
    late_task_ids = []
    for task_id, state in task_states.items():
        if state["canceled"]:
            continue
        if not state["completed"]:
            unresolved_task_ids.append(task_id)
        elif state["result"] != "hit":
            late_task_ids.append(task_id)

    return {
        "day_name": day_name,
        "id_to_handle": id_to_handle,
        "step_groundtruth": step_groundtruth,
        "missing_menu_by_step": missing_menu_by_step,
        "unresolved_task_ids": sorted(unresolved_task_ids),
        "late_task_ids": sorted(late_task_ids),
        "solvable": not missing_menu_by_step and not unresolved_task_ids and not late_task_ids,
    }


def compute_scenario_groundtruth(scenario):
    """Compute per-step groundtruth labels and a perfect-play solvability report."""
    updates_by_day = build_updates_by_day(scenario)
    day_reports = {}
    issues = []
    for day in scenario.get("days", []):
        report = compute_groundtruth_for_day(day, updates_by_day)
        day_reports[day["name"]] = report
        if report["missing_menu_by_step"]:
            issues.append(
                f"{day['name']}: due tasks missing from action menu at steps {sorted(report['missing_menu_by_step'])}"
            )
        if report["unresolved_task_ids"]:
            issues.append(
                f"{day['name']}: unresolved non-canceled tasks {report['unresolved_task_ids']}"
            )
        if report["late_task_ids"]:
            issues.append(
                f"{day['name']}: tasks only finish late under perfect play {report['late_task_ids']}"
            )
    return {
        "solvable": not issues,
        "issues": issues,
        "days": day_reports,
    }


def attach_groundtruth_labels(scenario):
    """Attach per-step groundtruth labels in-place and return the solvability report."""
    report = compute_scenario_groundtruth(scenario)
    for day in scenario.get("days", []):
        day_report = report["days"].get(day["name"], {})
        step_groundtruth = day_report.get("step_groundtruth", {})
        for step in day.get("steps", []):
            step["groundtruth"] = step_groundtruth.get(
                step["id"],
                {"status": "none", "actions": []},
            )
    return report


def parse_action(raw, valid_action_tokens, allowed_channels):
    """Parse interactive input into choice, action handles, and state query channel."""
    choice = None
    action_tokens = []
    state_query_channel = None
    saw_none = False
    tokens = [tok.strip() for tok in raw.replace(";", " ").split() if tok.strip()]
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        lower = token.lower()
        if lower == "time":
            state_query_channel = "clock"
            idx += 1
            continue
        if lower.startswith("state:"):
            state_query_channel = lower.split(":", 1)[1]
            idx += 1
            continue
        if lower == "state":
            if idx + 1 < len(tokens):
                state_query_channel = tokens[idx + 1].lower()
                idx += 2
                continue
            idx += 1
            continue
        if token in ("A", "B", "C"):
            choice = token
            idx += 1
            continue
        if lower == "none":
            saw_none = True
            idx += 1
            continue
        if token in valid_action_tokens:
            if token not in action_tokens:
                action_tokens.append(token)
            idx += 1
            continue
        idx += 1
    if saw_none and not action_tokens:
        action_tokens = []
    if state_query_channel and state_query_channel not in allowed_channels:
        state_query_channel = "INVALID"
    return choice, action_tokens, state_query_channel


def print_interactive_instructions(time_visible_by_default, show_task_legend, state_channels):
    """Print interactive instructions, including state query guidance."""
    print("=== Interactive Instructions ===")
    print("Each step shows a vignette and three options (A/B/C).")
    print("Choose one of A/B/C to advance.")
    if show_task_legend:
        print("A daily action-handle legend is shown at the start of each day.")
    else:
        print("A step-level action menu is shown before each choice.")
    print("Optionally perform tasks by appending action handles from the menu.")
    print("Input format: A|B|C [task_handle ...]")
    print("Example: B task_3")
    print("If you don't want to perform a task, just enter A, B, or C.")
    print("Some cues are hidden in state channels and will not appear in the vignette.")
    print("Type 'time' to check the current time (clock channel).")
    if state_channels:
        channels_line = ", ".join(state_channels)
        print(f"Type 'state <channel>' to query a state channel. Available: {channels_line}.")
    print("After a time check, enter your A/B/C choice on a new line.")
    print("Some cues will cancel or reschedule tasks; always follow the latest update.")
    if time_visible_by_default:
        print("Time is shown automatically each step; 'time' will reprint it.")
    else:
        print("Time is hidden unless you type 'time'.")
    print("")


def print_llm_instructions(
    time_visible_by_default,
    show_task_legend,
    state_channels,
    allow_heartbeat=False,
    auto_heartbeat_minutes=None,
    heartbeat_message_mode="channel_query",
):
    """Print LLM instructions, including state query guidance."""
    print("=== LLM Instructions ===")
    print("Each step shows a vignette and three options (A/B/C).")
    print("Choose one of A/B/C to advance.")
    if show_task_legend:
        print("A daily action-handle legend is shown at the start of each day.")
    else:
        print("A step-level action menu is shown before each choice.")
    print("Optionally perform tasks by returning action handles from the menu.")
    action_tokens = ["choose", "check_time", "query_state"]
    if allow_heartbeat:
        action_tokens.append("set_heartbeat")
    print("Output must be JSON with action=" + ", action=".join(action_tokens) + ".")
    print("Example: {\"action\":\"choose\",\"choice\":\"B\",\"task_ids\":[\"task_3\"]}")
    print(
        "Example (state query): "
        "{\"action\":\"query_state\",\"choice\":\"NONE\",\"task_ids\":[],\"channel\":\"calendar\"}"
    )
    print(
        "Example (time check): "
        "{\"action\":\"check_time\",\"choice\":\"NONE\",\"task_ids\":[],\"channel\":\"clock\"}"
    )
    if allow_heartbeat:
        print(
            "Example (heartbeat): "
            "{\"action\":\"set_heartbeat\",\"choice\":\"NONE\",\"task_ids\":[],"
            "\"channel\":\"NONE\",\"heartbeat_enabled\":true,\"heartbeat_minutes\":60}"
        )
        if heartbeat_message_mode == "task_reminder":
            print(
                "Heartbeat reminders are periodic (30 or 60 virtual minutes) and "
                "will surface task reminders for upcoming proactive tasks."
            )
        else:
            print(
                "Heartbeat reminders are periodic (30 or 60 virtual minutes), "
                "not tied to exact cue times."
            )
    if auto_heartbeat_minutes is not None:
        if heartbeat_message_mode == "task_reminder":
            print(
                f"Auto-heartbeat is enabled for each day at {auto_heartbeat_minutes} "
                "virtual-minute intervals with periodic upcoming-task reminders."
            )
        else:
            print(
                f"Auto-heartbeat is enabled for each day at {auto_heartbeat_minutes} "
                "virtual-minute intervals."
            )
    print("Some cues are only visible via state channels; they will not appear in the vignette.")
    print("State query responses appear as: State [channel]: <message>.")
    if state_channels:
        channels_line = ", ".join(state_channels)
        print(f"Available channels for query_state: {channels_line}.")
    if time_visible_by_default:
        print("Time is shown automatically each step; check_time will reprint it.")
    else:
        print("Time is hidden unless you use check_time.")
    print("")


def run_interactive_day(
    day,
    log_entries,
    time_visible_by_default,
    state_channels,
    allowed_channels,
    pre_updates,
    updates_by_step,
    show_task_legend,
):
    """Run one interactive day with optional state channel queries."""
    day_start_minutes = build_day_start_minutes(day)
    step_index = build_day_index(day)
    tasks = day["tasks"]
    lure_catalog = normalize_lure_catalog(day.get("lures", []))
    task_states = {}
    active_task_ids = set()
    for task in tasks:
        task_states[task["id"]] = {
            "completed": False,
            "cue_seen": False,
            "active": False,
        }
    for task in tasks:
        enc_type, enc_step = normalize_encoding(task["encoding"])
        if enc_type == "start":
            active_task_ids.add(task["id"])
            task_states[task["id"]]["active"] = True

    print(f"\n=== {day['name']} ===")
    for line in day.get("start_instructions", []):
        print(line)
    day_task_states = {task["id"]: init_task_state(task) for task in tasks}
    for task in tasks:
        enc_type, _ = normalize_encoding(task["encoding"])
        if enc_type == "start":
            day_task_states[task["id"]]["active"] = True
    for update in pre_updates:
        task_id = update.get("task_id")
        if task_id in day_task_states:
            apply_task_update(
                day_task_states[task_id], update, task_states=day_task_states
            )
    id_to_handle, _ = build_day_handle_maps(
        day_task_states,
        lure_catalog,
        seed_key=f"{day['name']}:handles",
    )
    if show_task_legend:
        legend_entries = []
        for task_id, state in sorted(day_task_states.items()):
            if not state["active"]:
                continue
            legend_entries.append(
                {
                    "id": task_id,
                    "handle": id_to_handle[task_id],
                    "action_text": runtime_task_action_text(state),
                }
            )
        for lure in lure_catalog:
            legend_entries.append(
                {
                    "id": lure["id"],
                    "handle": id_to_handle[lure["id"]],
                    "action_text": lure["action_text"],
                }
            )
        print(format_action_menu(legend_entries, header="Daily action-handle legend"))

    last_query_step_by_channel = {}
    last_snapshot_item_by_channel = {}
    for step_idx, step in enumerate(day["steps"]):
        for update in updates_by_step.get(step["id"], []):
            task_id = update.get("task_id")
            if task_id in day_task_states:
                apply_task_update(
                    day_task_states[task_id],
                    update,
                    task_states=day_task_states,
                )
        for task in tasks:
            enc_type, enc_step = normalize_encoding(task["encoding"])
            if enc_type == "step" and enc_step == step["id"]:
                active_task_ids.add(task["id"])
                task_states[task["id"]]["active"] = True
                day_task_states[task["id"]]["active"] = True

        step_minutes = time_to_minutes(step["time"])
        due_now = compute_due_now(
            day_task_states,
            active_task_ids,
            step,
            step_idx,
            step_minutes,
            day_start_minutes,
        )
        menu_entries, step_handle_to_id = build_step_action_menu(
            day_task_states,
            active_task_ids,
            lure_catalog,
            id_to_handle,
            day["name"],
            step["id"],
        )
        valid_tokens = set(step_handle_to_id.keys()).union(step_handle_to_id.values())

        print(f"\n{step['text']}")
        for option in step["options"]:
            print(option)
        print(format_action_menu(menu_entries, header="Step action menu"))
        print("Format: A|B|C [task_handle ...]. Example: 'B task_2'.")
        print("Type 'time' to check time or 'state <channel>' to query.")

        if time_visible_by_default:
            print(
                f"Time: {step['time']} | Stopwatch: {step_minutes - day_start_minutes} min"
            )

        choice = None
        task_ids = []
        state_query_counts = {}
        while choice is None:
            raw = input("Enter input: ").strip()
            parsed_choice, parsed_tokens, state_query_channel = parse_action(
                raw, valid_tokens, allowed_channels
            )
            if state_query_channel:
                if state_query_channel == "INVALID":
                    print("Unknown channel. Try again with a valid channel name.")
                    continue
                items = resolve_state_query_items(
                    state_query_channel,
                    day["steps"],
                    step_index[step["id"]],
                    day["name"],
                    day_start_minutes,
                    state_channels,
                    last_query_step_by_channel,
                    last_snapshot_item_by_channel,
                )
                state_query_counts[state_query_channel] = (
                    state_query_counts.get(state_query_channel, 0) + 1
                )
                response = build_state_query_response(
                    state_query_channel, items, day["name"], step["id"]
                )
                print(format_state_query_display(response["channel"], response["items"]))
                continue
            choice = parsed_choice
            task_ids = []
            for token in parsed_tokens:
                if token in step_handle_to_id:
                    task_ids.append(step_handle_to_id[token])
                elif token in step_handle_to_id.values():
                    task_ids.append(token)
            deduped = []
            seen_task_ids = set()
            for task_id in task_ids:
                if task_id in seen_task_ids:
                    continue
                seen_task_ids.add(task_id)
                deduped.append(task_id)
            task_ids = deduped
            if choice is None:
                print("Please provide a choice (A/B/C).")

        apply_runtime_completions(
            day_task_states,
            task_ids,
            due_now,
            step,
            step_idx,
            step_minutes,
            day_start_minutes,
        )
        log_entries.append(
            {
                "day": day["name"],
                "step_id": step["id"],
                "choice": choice,
                "task_ids": task_ids,
                "check_time": state_query_counts.get("clock", 0),
                "state_queries": state_query_counts,
            }
        )


def run_interactive(scenario, log_path, show_task_legend, out_dir=None):
    """Run an interactive PM-Bench session."""
    run_started_at_utc = now_utc_iso()
    run_started_perf = time.perf_counter()
    log_entries = []
    state_visibility = normalize_state_visibility(scenario)
    state_channels = normalize_state_channels(scenario)
    allowed_channels = list_state_channels(scenario)
    time_visible_by_default = state_visibility.get("clock", False)
    print_interactive_instructions(
        time_visible_by_default, show_task_legend, allowed_channels
    )
    for line in DAILY_TASK_HEADER_LINES:
        print(line)
    updates_by_day = build_updates_by_day(scenario)
    for day in scenario["days"]:
        day_updates = updates_by_day.get(day["name"], {"pre": [], "by_step": {}})
        run_interactive_day(
            day,
            log_entries,
            time_visible_by_default,
            state_channels,
            allowed_channels,
            day_updates.get("pre", []),
            day_updates.get("by_step", {}),
            show_task_legend,
        )

    log_path = normalize_log_path(log_path, "interactive", out_dir=out_dir)
    run_metadata = make_run_metadata(
        mode="interactive",
        started_at_utc=run_started_at_utc,
        finished_at_utc=now_utc_iso(),
        duration_seconds=time.perf_counter() - run_started_perf,
        entry_count=len(log_entries),
    )
    write_log(log_path, log_entries, run_metadata=run_metadata)
    return log_entries


def load_openai_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    raise SystemExit("OPENAI_API_KEY is not set in the shell environment.")


def load_openrouter_api_key():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        return api_key
    raise SystemExit("OPENROUTER_API_KEY is not set in the shell environment.")


def build_llm_client(backend, api_key=None, base_url=None):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install openai") from exc

    if backend == "sglang":
        resolved_key = api_key or "None"
        resolved_base_url = base_url or "http://127.0.0.1:30002/v1"
        return OpenAI(base_url=resolved_base_url, api_key=resolved_key)
    if backend == "openrouter":
        resolved_key = api_key or load_openrouter_api_key()
        resolved_base_url = base_url or "https://openrouter.ai/api/v1"
        return OpenAI(base_url=resolved_base_url, api_key=resolved_key)
    resolved_key = api_key or load_openai_api_key()
    return OpenAI(api_key=resolved_key, base_url=base_url)


def build_llm_system_prompt(
    show_task_legend,
    state_channels,
    allow_heartbeat=False,
    auto_heartbeat_minutes=None,
    heartbeat_message_mode="channel_query",
):
    """Return the system prompt describing the LLM interaction protocol."""
    menu_line = (
        "A daily action-handle legend is shown at the start of each day."
        if show_task_legend
        else "A step-level action menu is shown before each choice."
    )
    channels_line = ", ".join(state_channels) if state_channels else "clock"
    heartbeat_block = ""
    if allow_heartbeat:
        heartbeat_options = ", ".join(str(v) for v in HEARTBEAT_INTERVAL_OPTIONS)
        reminder_line = (
            "On each heartbeat, you may receive a reminder about upcoming tasks to keep in mind."
            if heartbeat_message_mode == "task_reminder"
            else "On each heartbeat, you may receive a nudge listing channels worth checking."
        )
        heartbeat_block = f"""
You may optionally configure a periodic heartbeat for proactive monitoring.
Use action=set_heartbeat with heartbeat_enabled=true and heartbeat_minutes in
[{heartbeat_options}]. This heartbeat is strictly periodic and is NOT aligned
to cue times. It only nudges proactive checking and does not reveal hidden cues.
You can disable it later with heartbeat_enabled=false.
{reminder_line}
"""
    auto_heartbeat_block = ""
    if auto_heartbeat_minutes is not None:
        auto_heartbeat_block = (
            "For this run, heartbeat is auto-enabled at the start of each day "
            f"every {auto_heartbeat_minutes} virtual minutes."
        )
        if heartbeat_message_mode == "task_reminder":
            auto_heartbeat_block += (
                " Heartbeat notifications will provide upcoming-task reminders."
            )
    return f"""You are taking a prospective memory evaluation that assess how 
you remember to do things later. One virtual day equals one
circuit of the board (7 a.m. -> 10 p.m.). Time advances step by step with
event cards. Your goal is to remember to perform tasks at the correct time
or event without reminders.

You must voluntarily remember to perform tasks at the right moment. If you
are unsure, you may still act; that is allowed. No reminders will be given
for missed tasks.

Some cues will cancel, override, or reschedule tasks. Always follow the most
recent instruction for any task.

Each step presents a short vignette with three ongoing-task options (A/B/C).
Pick one of A/B/C to advance. {menu_line}
You may perform one or more actions from the action menu in the same step.
Each action appears as task_N: <action text>. Return task_N handles in task_ids.
If you do not perform an action, set task_ids to [].
Never repeat the same task_N handle in task_ids.
Do not dump the whole menu into task_ids; include only the handles you are actually choosing now.
Return at most {MAX_ACTION_TASK_IDS} task handles in one choose action.

Some tasks are triggered by state cues that are NOT shown in the vignette.
You can only observe these cues by proactively querying state channels.
Available state channels: {channels_line}.

Use action=query_state to retrieve channel state (choice must be NONE).
The legacy action=check_time is supported and is equivalent to query_state with channel=clock.
No reminders will be given about when to query state; you must decide when to check.
State responses are returned as a single line: "State [channel]: <message>".
{heartbeat_block}
{auto_heartbeat_block}

After any query, you still must make a choose action to advance.

Example (time check):
User: You sit at your desk and feel it's close to 16:30.
Model: {{"action":"check_time","choice":"NONE","task_ids":[], "channel":"clock"}}
User: State [clock]: Time 15:58 | Stopwatch: 538 min
Model: {{"action":"choose","choice":"B","task_ids":["task_7"]}}

Always respond in JSON. Use these exact formats:
- {{"action":"query_state","choice":"NONE","task_ids":[],"channel":"calendar"}}
- {{"action":"check_time","choice":"NONE","task_ids":[],"channel":"clock"}}
- {{"action":"choose","choice":"A","task_ids":[],"channel":"NONE"}}
- {{"action":"choose","choice":"B","task_ids":["task_1"],"channel":"NONE"}}
{('- {{"action":"set_heartbeat","choice":"NONE","task_ids":[],"channel":"NONE","heartbeat_enabled":true,"heartbeat_minutes":60}}' if allow_heartbeat else '')}"""


def build_action_schema(allowed_task_ids, allowed_channels, allow_heartbeat=False):
    """Return a strict JSON schema for LLM actions."""
    action_enum = ["choose", "check_time", "query_state"]
    if allow_heartbeat:
        action_enum.append("set_heartbeat")
    properties = {
        "action": {
            "type": "string",
            "enum": action_enum,
        },
        "choice": {
            "type": "string",
            "enum": ["A", "B", "C", "NONE"],
        },
        "task_ids": {
            "type": "array",
            "items": {"type": "string", "enum": allowed_task_ids},
            "maxItems": min(MAX_ACTION_TASK_IDS, max(0, len(allowed_task_ids))),
        },
        "channel": {
            "type": "string",
            "enum": sorted(set(allowed_channels + ["NONE"])),
        },
    }
    if allow_heartbeat:
        properties["heartbeat_enabled"] = {"type": "boolean"}
        properties["heartbeat_minutes"] = {
            "type": "integer",
            "enum": list(HEARTBEAT_INTERVAL_OPTIONS),
        }
    required_fields = ["action", "choice", "task_ids", "channel"]
    if allow_heartbeat:
        # Some providers validate that required includes every declared property.
        required_fields.extend(["heartbeat_enabled", "heartbeat_minutes"])
    return {
        "type": "object",
        "properties": properties,
        "required": required_fields,
        "additionalProperties": False,
    }


def request_model_action(
    client,
    model,
    messages,
    allowed_task_ids,
    allowed_channels,
    backend,
    allow_heartbeat=False,
    max_tokens=256,
):
    """Request a model action with a strict JSON schema."""
    schema = build_action_schema(
        allowed_task_ids, allowed_channels, allow_heartbeat=allow_heartbeat
    )
    finish_reason = None
    if backend == "openai":
        response = client.responses.create(
            model=model,
            input=messages,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "pm_action",
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        raw_text = response.output_text
        try:
            return parse_json_payload(raw_text)
        except Exception as exc:
            detail = (
                "Failed to parse model JSON response: "
                f"{type(exc).__name__}: {exc}. "
                "finish_reason=None. "
                f"raw_text={raw_text!r}"
            )
            raise InvalidModelResponseError(
                detail,
                raw_text=raw_text,
                finish_reason=None,
            ) from exc
    else:
        # Some OpenRouter providers occasionally return empty content for structured
        # output calls. Retry, then fall back to no-schema JSON prompting.
        attempt_modes = [True]
        if backend == "openrouter":
            attempt_modes.extend([True, False, False])
        else:
            attempt_modes.append(True)

        raw_text = None
        last_response_dump = None
        last_exception = None
        last_parse_exception = None
        for attempt_idx, use_schema in enumerate(attempt_modes, start=1):
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
            }
            if backend == "sglang":
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                if use_schema:
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": "pm_action", "schema": schema},
                    }
            elif backend == "openrouter":
                kwargs["extra_body"] = {"reasoning": {"exclude": True}}
                if use_schema:
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": "pm_action", "schema": schema},
                    }

            mode_label = "schema" if use_schema else "fallback-no-schema"
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                print(
                    f"Model call failed ({mode_label}, attempt {attempt_idx}/{len(attempt_modes)}): "
                    f"{type(exc).__name__}: {exc}"
                )
                if attempt_idx < len(attempt_modes):
                    time.sleep(0.25)
                    continue
                break
            raw_text = extract_chat_text(response)
            finish_reason = extract_chat_finish_reason(response)
            last_response_dump = response.model_dump()
            print(
                f"Raw LLM response ({mode_label}, attempt {attempt_idx}/{len(attempt_modes)}):",
                repr(raw_text),
                f"(finish_reason={finish_reason})",
            )
            if raw_text is not None and raw_text.strip():
                try:
                    return parse_json_payload(raw_text)
                except Exception as exc:
                    last_parse_exception = exc
                    print(
                        f"Failed to parse JSON ({mode_label}, attempt {attempt_idx}/{len(attempt_modes)}): "
                        f"{type(exc).__name__}: {exc}"
                    )
                    if attempt_idx < len(attempt_modes):
                        time.sleep(0.25)
                        continue
                    break
            if attempt_idx < len(attempt_modes):
                time.sleep(0.25)

        detail = (
            "Failed to obtain valid JSON action from backend "
            f"{backend} after {len(attempt_modes)} attempt(s). "
            f"finish_reason={finish_reason!r}. "
            f"raw_text={raw_text!r}. "
            f"response={last_response_dump!r}"
        )
        if last_parse_exception is not None:
            detail += (
                f" parse_error={type(last_parse_exception).__name__}: {last_parse_exception}"
            )
        if last_exception is not None:
            detail += f" last_exception={type(last_exception).__name__}: {last_exception}"
        raise InvalidModelResponseError(
            detail,
            raw_text=raw_text,
            finish_reason=finish_reason,
        )


def extract_chat_text(response):
    if not response.choices:
        return None
    choice = response.choices[0]
    message = getattr(choice, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content:
            return content
    return getattr(choice, "text", None)


def extract_chat_finish_reason(response):
    if not getattr(response, "choices", None):
        return None
    choice = response.choices[0]
    return getattr(choice, "finish_reason", None) or getattr(
        choice, "native_finish_reason", None
    )


class InvalidModelResponseError(ValueError):
    def __init__(self, message, raw_text=None, finish_reason=None):
        super().__init__(message)
        self.raw_text = raw_text
        self.finish_reason = finish_reason


def parse_json_payload(raw_text):
    if raw_text is None:
        raise ValueError("Model response is empty.")
    text = raw_text.strip()
    if "<think>" in text:
        # Drop model thoughts and keep only the visible response.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    decoder = json.JSONDecoder()
    try:
        payload, end = decoder.raw_decode(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
        return payload
    remainder = text[end:].strip()
    if remainder:
        # Ignore trailing content if the model returns extra tokens.
        return payload
    return payload


def normalize_task_ids(action):
    task_ids = action.get("task_ids")
    if task_ids is None:
        task_id = action.get("task_id")
        return [task_id] if task_id else []
    normalized = []
    for task_id in task_ids:
        if not task_id:
            continue
        if task_id not in normalized:
            normalized.append(task_id)
    return normalized


def virtual_clock_minute(day_idx, step_time):
    """Return absolute virtual minutes from week start."""
    return (day_idx * 24 * 60) + time_to_minutes(step_time)


def pending_proactive_channels_for_heartbeat(task_states, state_visibility):
    """Return sorted proactive channels for active, incomplete tasks."""
    channels = set()
    for state in task_states.values():
        if not state.get("active"):
            continue
        if state.get("completed") or state.get("canceled"):
            continue
        current = state.get("current", {})
        if not requires_state_monitoring(current, state_visibility):
            continue
        channel = required_monitor_channel(current)
        if channel:
            channels.add(channel)
    return sorted(channels)


def pending_task_reminders_for_heartbeat(
    task_states,
    state_visibility,
    step_minutes,
    day_start_minutes,
    max_items=4,
):
    """Return reminder entries for active proactive tasks, prioritized by urgency."""
    reminders = []
    for task_id, state in sorted(task_states.items()):
        if not state.get("active"):
            continue
        if state.get("completed") or state.get("canceled"):
            continue
        task = state.get("task", {})
        depends_on = task.get("depends_on")
        if depends_on:
            dependency_state = task_states.get(depends_on)
            if not dependency_state or not dependency_state.get("completed"):
                continue
        current = state.get("current", {})
        if not requires_state_monitoring(current, state_visibility):
            continue
        action_text = runtime_task_action_text(state).strip().rstrip(".")
        task_type = current.get("type")
        if task_type in ("time", "time_check"):
            target = current.get("target_time")
            if not target:
                continue
            target_minutes = time_to_minutes(target)
            delta = target_minutes - step_minutes
            if delta >= 0:
                if delta > HEARTBEAT_REMINDER_TIME_HORIZON_MINUTES:
                    continue
                sort_key = (0, delta, task_id)
                when_text = f"in {delta} min" if delta > 0 else "now"
                detail = f"at {target} ({when_text})"
            else:
                if abs(delta) > HEARTBEAT_REMINDER_OVERDUE_HORIZON_MINUTES:
                    continue
                sort_key = (1, abs(delta), task_id)
                detail = f"target was {target} ({abs(delta)} min ago)"
            reminders.append(
                {
                    "sort_key": sort_key,
                    "text": f"{action_text} ({detail})",
                }
            )
            continue

        channel = current.get("cue_channel", "narrative")
        if channel == "narrative":
            continue
        cue_desc = f"when its {channel} cue appears"
        if current.get("label"):
            cue_desc = f"when the previously mentioned cue appears on {channel}"
        reminders.append(
            {
                "sort_key": (2, task_id),
                "text": f"{action_text} ({cue_desc})",
            }
        )

    reminders.sort(key=lambda item: item["sort_key"])
    return [item["text"] for item in reminders[:max_items]]


def format_heartbeat_message(
    interval_minutes,
    pending_channels=None,
    task_reminders=None,
    message_mode="channel_query",
):
    """Format a heartbeat reminder line for the model."""
    if message_mode == "task_reminder":
        if task_reminders:
            return (
                f"Heartbeat ({interval_minutes} min): periodic reminder window. "
                "Upcoming tasks to keep in mind: "
                + "; ".join(task_reminders)
                + ". Consider executing them when their cue/time condition is met."
            )
        return (
            f"Heartbeat ({interval_minutes} min): periodic reminder window. "
            "No currently pending proactive tasks."
        )
    if pending_channels:
        return (
            f"Heartbeat ({interval_minutes} min): periodic proactive check window. "
            "Consider querying pending channels: "
            + ", ".join(pending_channels)
            + "."
        )
    return (
        f"Heartbeat ({interval_minutes} min): periodic proactive check window. "
        "No currently pending proactive channels."
    )


def run_llm(
    scenario,
    log_path,
    model,
    env_path,
    max_time_requests,
    backend,
    base_url,
    api_key,
    prompt_log_path=None,
    show_task_legend=False,
    out_dir=None,
    enable_heartbeat=False,
    auto_heartbeat_minutes=None,
    heartbeat_message_mode="channel_query",
    max_tokens=256,
):
    """Run an LLM session with optional state channel queries."""
    run_started_at_utc = now_utc_iso()
    run_started_perf = time.perf_counter()
    client = build_llm_client(backend, api_key=api_key, base_url=base_url)
    log_path = normalize_log_path(log_path, model, out_dir=out_dir)
    if prompt_log_path is None and out_dir is not None:
        prompt_log_path = (
            f"{os.path.splitext(os.path.basename(log_path))[0]}.prompt.txt"
        )
    prompt_log_path = resolve_output_path(prompt_log_path, out_dir=out_dir)
    log_entries = []
    state_visibility = normalize_state_visibility(scenario)
    state_channels = normalize_state_channels(scenario)
    allowed_channels = list_state_channels(scenario)
    time_visible_by_default = state_visibility.get("clock", False)
    heartbeat_runtime_enabled = enable_heartbeat or auto_heartbeat_minutes is not None
    print_llm_instructions(
        time_visible_by_default,
        show_task_legend,
        allowed_channels,
        allow_heartbeat=enable_heartbeat,
        auto_heartbeat_minutes=auto_heartbeat_minutes,
        heartbeat_message_mode=heartbeat_message_mode,
    )
    prompt_log_handle = None
    if prompt_log_path:
        prompt_log_handle = open(prompt_log_path, "w", encoding="utf-8", buffering=1)

    system_prompt = build_llm_system_prompt(
        show_task_legend,
        allowed_channels,
        allow_heartbeat=enable_heartbeat,
        auto_heartbeat_minutes=auto_heartbeat_minutes,
        heartbeat_message_mode=heartbeat_message_mode,
    )
    messages = [{"role": "system", "content": system_prompt}]
    heartbeat_state = {
        "enabled": False,
        "interval_minutes": None,
        "next_due_minute": None,
    }
    daily_header = "\n".join(DAILY_TASK_HEADER_LINES)
    print(daily_header)
    messages.append({"role": "user", "content": daily_header})
    updates_by_day = build_updates_by_day(scenario)
    for day_idx, day in enumerate(scenario["days"]):
        day_start_minutes = build_day_start_minutes(day)
        tasks = day["tasks"]
        lure_catalog = normalize_lure_catalog(day.get("lures", []))
        active_task_ids = set()
        for task in tasks:
            enc_type, _ = normalize_encoding(task["encoding"])
            if enc_type == "start":
                active_task_ids.add(task["id"])

        day_updates = updates_by_day.get(day["name"], {"pre": [], "by_step": {}})
        day_task_states = {task["id"]: init_task_state(task) for task in tasks}
        for task in tasks:
            enc_type, _ = normalize_encoding(task["encoding"])
            if enc_type == "start":
                day_task_states[task["id"]]["active"] = True
        for update in day_updates.get("pre", []):
            task_id = update.get("task_id")
            if task_id in day_task_states:
                apply_task_update(
                    day_task_states[task_id],
                    update,
                    task_states=day_task_states,
                )
        id_to_handle, _ = build_day_handle_maps(
            day_task_states,
            lure_catalog,
            seed_key=f"{day['name']}:handles",
        )
        day_header = "\n".join([f"=== {day['name']} ==="] + day.get("start_instructions", []))
        print(f"\n{day_header}")
        messages.append({"role": "user", "content": day_header})
        if auto_heartbeat_minutes is not None:
            first_step_time = day["steps"][0]["time"] if day.get("steps") else "00:00"
            day_start_virtual_minute = virtual_clock_minute(day_idx, first_step_time)
            heartbeat_state["enabled"] = True
            heartbeat_state["interval_minutes"] = auto_heartbeat_minutes
            heartbeat_state["next_due_minute"] = (
                day_start_virtual_minute + auto_heartbeat_minutes
            )
            auto_text = (
                f"Heartbeat auto-enabled for this day every {auto_heartbeat_minutes} "
                "virtual minutes (strictly periodic, not cue-aligned)."
            )
            print(auto_text)
            messages.append({"role": "user", "content": auto_text})
        if show_task_legend:
            legend_entries = []
            for task_id, state in sorted(day_task_states.items()):
                if not state["active"]:
                    continue
                legend_entries.append(
                    {
                        "id": task_id,
                        "handle": id_to_handle[task_id],
                        "action_text": runtime_task_action_text(state),
                    }
                )
            for lure in lure_catalog:
                legend_entries.append(
                    {
                        "id": lure["id"],
                        "handle": id_to_handle[lure["id"]],
                        "action_text": lure["action_text"],
                    }
                )
            legend_text = format_action_menu(
                legend_entries,
                header="Daily action-handle legend",
            )
            print(legend_text)
            messages.append({"role": "user", "content": legend_text})

        last_query_step_by_channel = {}
        last_snapshot_item_by_channel = {}
        for step_idx, step in enumerate(day["steps"]):
            for update in day_updates.get("by_step", {}).get(step["id"], []):
                task_id = update.get("task_id")
                if task_id in day_task_states:
                    apply_task_update(
                        day_task_states[task_id],
                        update,
                        task_states=day_task_states,
                    )
            for task in tasks:
                enc_type, enc_step = normalize_encoding(task["encoding"])
                if enc_type == "step" and enc_step == step["id"]:
                    active_task_ids.add(task["id"])
                    day_task_states[task["id"]]["active"] = True

            step_minutes = time_to_minutes(step["time"])
            step_virtual_minute = virtual_clock_minute(day_idx, step["time"])
            due_now = compute_due_now(
                day_task_states,
                active_task_ids,
                step,
                step_idx,
                step_minutes,
                day_start_minutes,
            )
            menu_entries, step_handle_to_id = build_step_action_menu(
                day_task_states,
                active_task_ids,
                lure_catalog,
                id_to_handle,
                day["name"],
                step["id"],
            )
            menu_text = format_action_menu(menu_entries, header="Step action menu")
            heartbeat_prompted = False
            if (
                heartbeat_runtime_enabled
                and heartbeat_state["enabled"]
                and heartbeat_state["next_due_minute"] is not None
                and step_virtual_minute >= heartbeat_state["next_due_minute"]
            ):
                pending_channels = None
                task_reminders = None
                if heartbeat_message_mode == "task_reminder":
                    task_reminders = pending_task_reminders_for_heartbeat(
                        day_task_states,
                        state_visibility,
                        step_minutes,
                        day_start_minutes,
                    )
                else:
                    pending_channels = pending_proactive_channels_for_heartbeat(
                        day_task_states, state_visibility
                    )
                heartbeat_text = format_heartbeat_message(
                    heartbeat_state["interval_minutes"],
                    pending_channels=pending_channels,
                    task_reminders=task_reminders,
                    message_mode=heartbeat_message_mode,
                )
                print(heartbeat_text)
                messages.append({"role": "user", "content": heartbeat_text})
                heartbeat_prompted = True
                while step_virtual_minute >= heartbeat_state["next_due_minute"]:
                    heartbeat_state["next_due_minute"] += heartbeat_state[
                        "interval_minutes"
                    ]
            time_line = ""
            if time_visible_by_default:
                time_line = (
                    f"\nTime: {step['time']} | Stopwatch: {step_minutes - day_start_minutes} min"
                )
            prompt = (
                f"{step['text']}\n"
                + "\n".join(step["options"])
                + "\n"
                + menu_text
                + "\n"
                + time_line
            )
            print(f"\n{step['text']}")
            for option in step["options"]:
                print(option)
            print(menu_text)
            if time_visible_by_default:
                print(
                    f"Time: {step['time']} | Stopwatch: {step_minutes - day_start_minutes} min"
                )
            messages.append({"role": "user", "content": prompt})

            check_time_count = 0
            state_query_counts = {}
            while True:
                if prompt_log_handle:
                    prompt_log_handle.write(f"=== {day['name']} {step['id']} ===\n")
                    prompt_log_handle.write(json.dumps(messages, indent=2))
                    prompt_log_handle.write("\n\n")
                    prompt_log_handle.flush()
                allowed_ids = sorted(step_handle_to_id.keys())
                try:
                    action = request_model_action(
                        client,
                        model,
                        messages,
                        allowed_ids,
                        allowed_channels,
                        backend,
                        allow_heartbeat=enable_heartbeat,
                        max_tokens=max_tokens,
                    )
                except InvalidModelResponseError as exc:
                    # Keep long sweeps running even when a provider repeatedly
                    # returns empty/truncated structured outputs.
                    fallback_task_ids = sorted(due_now)[:MAX_ACTION_TASK_IDS]
                    action = {
                        "action": "choose",
                        "choice": "A",
                        "task_ids": fallback_task_ids,
                        "channel": "NONE",
                    }
                    print(
                        "Model response invalid after retries; using fallback action.",
                        f"error={exc}",
                        f"fallback={action}",
                    )
                    if prompt_log_handle:
                        prompt_log_handle.write(
                            "FALLBACK_ACTION: "
                            + json.dumps(action)
                            + "\n"
                            + "FALLBACK_REASON: "
                            + str(exc)
                            + "\n\n"
                        )
                        prompt_log_handle.flush()
                if prompt_log_handle:
                    estimated_tokens = estimate_input_tokens(messages)
                    prompt_log_handle.write(f"EST_INPUT_TOKENS: {estimated_tokens}\n\n")
                    prompt_log_handle.flush()
                if action["action"] in ("check_time", "query_state"):
                    channel = action.get("channel")
                    if action["choice"] != "NONE":
                        messages.append({"role": "assistant", "content": json.dumps(action)})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Invalid choice. When action=query_state or action=check_time, "
                                    "choice must be NONE. Respond again with valid JSON."
                                ),
                            }
                        )
                        continue
                    if normalize_task_ids(action):
                        messages.append({"role": "assistant", "content": json.dumps(action)})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Invalid task_ids. When action=query_state or action=check_time, "
                                    "task_ids must be []. Respond again with valid JSON."
                                ),
                            }
                        )
                        continue
                    if action["action"] == "check_time":
                        channel = "clock"
                    if channel not in allowed_channels:
                        messages.append({"role": "assistant", "content": json.dumps(action)})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Invalid channel. Use one of the available state channels "
                                    f"({', '.join(allowed_channels)})."
                                ),
                            }
                        )
                        continue
                    messages.append({"role": "assistant", "content": json.dumps(action)})
                    state_query_counts[channel] = state_query_counts.get(channel, 0) + 1
                    if channel == "clock":
                        check_time_count += 1
                    print(f"Model action: query_state {channel}")
                    items = resolve_state_query_items(
                        channel,
                        day["steps"],
                        step_idx,
                        day["name"],
                        day_start_minutes,
                        state_channels,
                        last_query_step_by_channel,
                        last_snapshot_item_by_channel,
                    )
                    response = build_state_query_response(
                        channel, items, day["name"], step["id"]
                    )
                    response_text = format_state_query_display(
                        response["channel"], response["items"]
                    )
                    print(response_text)
                    messages.append({"role": "user", "content": response_text})
                    continue

                if action["action"] == "set_heartbeat":
                    if not enable_heartbeat:
                        messages.append({"role": "assistant", "content": json.dumps(action)})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Heartbeat is disabled for this run. Respond again with "
                                    "action=choose, action=query_state, or action=check_time."
                                ),
                            }
                        )
                        continue
                    if action["choice"] != "NONE":
                        messages.append({"role": "assistant", "content": json.dumps(action)})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Invalid choice. When action=set_heartbeat, choice must be NONE. "
                                    "Respond again with valid JSON."
                                ),
                            }
                        )
                        continue
                    if normalize_task_ids(action):
                        messages.append({"role": "assistant", "content": json.dumps(action)})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Invalid task_ids. When action=set_heartbeat, task_ids must be []. "
                                    "Respond again with valid JSON."
                                ),
                            }
                        )
                        continue
                    if action.get("channel") not in ("NONE", None):
                        messages.append({"role": "assistant", "content": json.dumps(action)})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Invalid channel. When action=set_heartbeat, channel must be NONE. "
                                    "Respond again with valid JSON."
                                ),
                            }
                        )
                        continue
                    heartbeat_enabled = bool(action.get("heartbeat_enabled", True))
                    requested_interval = action.get("heartbeat_minutes")
                    if heartbeat_enabled:
                        interval = requested_interval or heartbeat_state["interval_minutes"] or 60
                        if interval not in HEARTBEAT_INTERVAL_OPTIONS:
                            valid_options = ", ".join(
                                str(v) for v in HEARTBEAT_INTERVAL_OPTIONS
                            )
                            messages.append(
                                {"role": "assistant", "content": json.dumps(action)}
                            )
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "Invalid heartbeat_minutes. Use one of: "
                                        f"{valid_options}. Respond again with valid JSON."
                                    ),
                                }
                            )
                            continue
                        heartbeat_state["enabled"] = True
                        heartbeat_state["interval_minutes"] = interval
                        heartbeat_state["next_due_minute"] = step_virtual_minute + interval
                        ack_text = (
                            f"Heartbeat enabled every {interval} virtual minutes "
                            "(strictly periodic, not cue-aligned)."
                        )
                    else:
                        heartbeat_state["enabled"] = False
                        heartbeat_state["interval_minutes"] = None
                        heartbeat_state["next_due_minute"] = None
                        ack_text = "Heartbeat disabled."
                    print(f"Model action: set_heartbeat -> {ack_text}")
                    messages.append({"role": "assistant", "content": json.dumps(action)})
                    messages.append({"role": "user", "content": ack_text})
                    continue

                if action.get("channel") not in ("NONE", None):
                    messages.append({"role": "assistant", "content": json.dumps(action)})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Invalid channel. When action=choose, channel must be NONE. "
                                "Respond again with valid JSON."
                            ),
                        }
                    )
                    continue
                choice = action["choice"]
                if choice not in ("A", "B", "C"):
                    messages.append({"role": "assistant", "content": json.dumps(action)})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Invalid choice. When action=choose, "
                                "choice must be A, B, or C. Respond again with valid JSON."
                            ),
                        }
                    )
                    continue
                messages.append({"role": "assistant", "content": json.dumps(action)})
                task_tokens = normalize_task_ids(action)
                task_ids = []
                for token in task_tokens:
                    if token in step_handle_to_id:
                        mapped_id = step_handle_to_id[token]
                    elif token in step_handle_to_id.values():
                        mapped_id = token
                    else:
                        continue
                    if mapped_id in task_ids:
                        continue
                    task_ids.append(mapped_id)
                if task_ids:
                    print(f"Model action: choose {choice} + action(s) {', '.join(task_ids)}")
                else:
                    print(f"Model action: choose {choice}")
                apply_runtime_completions(
                    day_task_states,
                    task_ids,
                    due_now,
                    step,
                    step_idx,
                    step_minutes,
                    day_start_minutes,
                )
                log_entries.append(
                    {
                        "day": day["name"],
                        "step_id": step["id"],
                        "choice": choice,
                        "task_ids": task_ids,
                        "check_time": check_time_count,
                        "state_queries": state_query_counts,
                        "heartbeat_enabled": heartbeat_state["enabled"],
                        "heartbeat_interval_minutes": heartbeat_state["interval_minutes"],
                        "heartbeat_prompted": heartbeat_prompted,
                    }
                )
                break

    if prompt_log_handle:
        prompt_log_handle.close()
    run_metadata = make_run_metadata(
        mode="run-llm",
        started_at_utc=run_started_at_utc,
        finished_at_utc=now_utc_iso(),
        duration_seconds=time.perf_counter() - run_started_perf,
        entry_count=len(log_entries),
        model=model,
        backend=backend,
    )
    write_log(log_path, log_entries, run_metadata=run_metadata)
    return log_entries


def requires_state_monitoring(current, state_visibility):
    """Return whether a task requires proactive state monitoring."""
    task_type = current.get("type")
    if task_type in ("time", "time_check"):
        return not bool((state_visibility or {}).get("clock", False))
    if task_type == "event":
        cue_channel = current.get("cue_channel", "narrative")
        return cue_channel != "narrative"
    return False


def required_monitor_channel(current):
    """Return required query channel for strict monitoring, or None."""
    task_type = current.get("type")
    if task_type in ("time", "time_check"):
        return "clock"
    if task_type == "event":
        cue_channel = current.get("cue_channel", "narrative")
        if cue_channel != "narrative":
            return cue_channel
    return None


def get_action_state_query_counts(action):
    """Return normalized per-channel query counts for a step/action log entry."""
    counts = {}
    state_queries = action.get("state_queries")
    if isinstance(state_queries, dict):
        for channel, value in state_queries.items():
            if not isinstance(channel, str):
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                counts[channel] = counts.get(channel, 0) + parsed
    else:
        try:
            check_time_calls = int(action.get("check_time", 0))
        except (TypeError, ValueError):
            check_time_calls = 0
        if check_time_calls > 0:
            counts["clock"] = check_time_calls
    return counts


def add_channel_counts(target, source):
    """Accumulate source per-channel counts into target."""
    for channel, value in source.items():
        target[channel] = target.get(channel, 0) + value


def get_action_channel_query_count(action, channel):
    """Return channel query count for this step/action log entry."""
    if channel is None:
        return 0
    return get_action_state_query_counts(action).get(channel, 0)


def score_day(day, actions, pre_updates=None, updates_by_step=None, state_visibility=None):
    """Score a day and compute error metrics.

    Error definitions:
    - false_alarm: performed a task when no task was due or the task was inactive.
    - commission: repeated a task after it was already completed.
    - wrong_content: performed a task when a different task was due.
    """
    step_index = build_day_index(day)
    day_start_minutes = build_day_start_minutes(day)
    steps = day["steps"]
    tasks = day["tasks"]
    task_states = {task["id"]: init_task_state(task) for task in tasks}

    active_task_ids = set()
    for task in tasks:
        enc_type, _ = normalize_encoding(task["encoding"])
        if enc_type == "start":
            active_task_ids.add(task["id"])
            task_states[task["id"]]["active"] = True

    metrics = {
        "hit": 0,
        "late": 0,
        "miss": 0,
        "false_alarm": 0,
        "commission": 0,
        "wrong_content": 0,
        "check_time_calls": 0,
        "state_query_calls": 0,
        "overkill_steps": 0,
        "chosen_tasks": 0,
        "dependency_violation": 0,
        "cross_day_total": 0,
        "cross_day_hit": 0,
        "cross_day_late": 0,
        "cross_day_miss": 0,
        "update_total": 0,
        "update_hit": 0,
        "update_late": 0,
        "update_miss": 0,
        "update_violation": 0,
        "update_canceled": 0,
        "canceled_total": 0,
        "exact_set_match_steps": 0,
        "exact_set_mismatch_steps": 0,
        "exact_set_match_reward": 0,
        "set_tp": 0,
        "set_fp": 0,
        "set_fn": 0,
    }
    by_type = {}
    by_regular = {
        "regular": {"hit": 0, "late": 0, "miss": 0, "total": 0},
        "irregular": {"hit": 0, "late": 0, "miss": 0, "total": 0},
    }
    by_monitoring = {
        "no_proactive_monitoring": {"hit": 0, "late": 0, "miss": 0, "total": 0},
        "proactive_monitoring_required": {"hit": 0, "late": 0, "miss": 0, "total": 0},
    }
    by_monitoring_channel = {}
    for task in tasks:
        by_type.setdefault(task["type"], {"hit": 0, "late": 0, "miss": 0, "total": 0})
        key = "regular" if task["regular"] else "irregular"
        by_type[task["type"]]["total"] += 1
        by_regular[key]["total"] += 1
        if task.get("cross_day"):
            metrics["cross_day_total"] += 1

    if pre_updates is None:
        pre_updates = []
    if updates_by_step is None:
        updates_by_step = {}

    updated_task_ids = set(update.get("task_id") for update in pre_updates)
    for updates in updates_by_step.values():
        for update in updates:
            updated_task_ids.add(update.get("task_id"))
    updated_task_ids.discard(None)
    metrics["update_total"] = len(updated_task_ids)
    for task_id in updated_task_ids:
        if task_id in task_states:
            task_states[task_id]["has_update"] = True

    if len(actions) != len(steps):
        raise ValueError("Action log does not match number of steps for the day.")

    for update in pre_updates:
        task_id = update.get("task_id")
        if task_id in task_states:
            apply_task_update(
                task_states[task_id],
                update,
                task_states=task_states,
                by_type=by_type,
                by_regular=by_regular,
                metrics=metrics,
            )

    for step_idx, step in enumerate(steps):
        action = actions[step_idx]
        for update in updates_by_step.get(step["id"], []):
            task_id = update.get("task_id")
            if task_id in task_states:
                apply_task_update(
                    task_states[task_id],
                    update,
                    task_states=task_states,
                    by_type=by_type,
                    by_regular=by_regular,
                    metrics=metrics,
                )
        for task in tasks:
            enc_type, enc_step = normalize_encoding(task["encoding"])
            if enc_type == "step" and enc_step == step["id"]:
                active_task_ids.add(task["id"])
                task_states[task["id"]]["active"] = True

        due_now = set()
        step_minutes = time_to_minutes(step["time"])
        for task_id in active_task_ids:
            state = task_states[task_id]
            task = state["task"]
            current = state["current"]
            if state["completed"]:
                continue
            if state["canceled"]:
                continue
            depends_on = task.get("depends_on")
            dependency_met = True
            if depends_on:
                dependency_state = task_states.get(depends_on)
                dependency_met = bool(dependency_state and dependency_state["completed"])
            if current["type"] == "event" and is_due_event(current, step):
                state["cue_seen"] = True
                if state["cue_step_idx"] is None:
                    state["cue_step_idx"] = step_idx
                if dependency_met:
                    due_now.add(task_id)
            elif current["type"] == "time" and is_due_time(current, step):
                state["cue_seen"] = True
                if state["cue_step_idx"] is None:
                    state["cue_step_idx"] = step_idx
                if dependency_met:
                    due_now.add(task_id)
            elif current["type"] == "time_check":
                status = timecheck_status(current, step_minutes, day_start_minutes)
                if status == "on_time":
                    if dependency_met:
                        due_now.add(task_id)

        query_counts = get_action_state_query_counts(action)
        metrics["state_query_calls"] += sum(query_counts.values())
        metrics["check_time_calls"] += query_counts.get("clock", 0)
        chosen_task_ids = normalize_task_ids(action)

        # Exact-set diagnostic: +1 only when chosen set exactly matches due set, else -1.
        chosen_set = set(chosen_task_ids)
        if chosen_set == due_now:
            metrics["exact_set_match_steps"] += 1
            metrics["exact_set_match_reward"] += 1
        else:
            metrics["exact_set_mismatch_steps"] += 1
            metrics["exact_set_match_reward"] -= 1
        metrics["set_tp"] += len(chosen_set & due_now)
        metrics["set_fp"] += len(chosen_set - due_now)
        metrics["set_fn"] += len(due_now - chosen_set)

        metrics["chosen_tasks"] += len(chosen_task_ids)
        if len(chosen_task_ids) > len(due_now):
            metrics["overkill_steps"] += 1
        for task_id in chosen_task_ids:
            if task_id not in task_states:
                metrics["false_alarm"] += 1
                if due_now:
                    metrics["wrong_content"] += 1
                continue
            state = task_states[task_id]
            task = state["task"]
            current = state["current"]
            required_channel = required_monitor_channel(current)
            had_required_query = (
                get_action_channel_query_count(action, required_channel) > 0
            )
            depends_on = task.get("depends_on")
            if depends_on:
                dependency_state = task_states.get(depends_on)
                if not dependency_state or not dependency_state["completed"]:
                    metrics["dependency_violation"] += 1
                    metrics["false_alarm"] += 1
                    if due_now:
                        metrics["wrong_content"] += 1
                    continue
            if state["canceled"]:
                metrics["false_alarm"] += 1
                metrics["update_violation"] += 1
                if due_now:
                    metrics["wrong_content"] += 1
                continue
            if state["completed"]:
                metrics["commission"] += 1
                if due_now and task_id not in due_now:
                    metrics["wrong_content"] += 1
            elif not state["active"]:
                metrics["false_alarm"] += 1
                if due_now:
                    metrics["wrong_content"] += 1
            else:
                if task_id in due_now:
                    state["completed"] = True
                    state["completed_at"] = step["id"]
                    state["result"] = "hit"
                    state["completion_had_required_query"] = had_required_query
                    metrics["hit"] += 1
                elif current["type"] == "time_check":
                    status = timecheck_status(current, step_minutes, day_start_minutes)
                    target_minutes = time_to_minutes(current["target_time"])
                    delta = step_minutes - target_minutes
                    if status == "late" and 0 < delta <= TIME_LATE_WINDOW_MINUTES:
                        state["completed"] = True
                        state["completed_at"] = step["id"]
                        state["result"] = "late"
                        state["completion_had_required_query"] = had_required_query
                        metrics["late"] += 1
                    else:
                        metrics["false_alarm"] += 1
                elif current["type"] == "time":
                    target_minutes = time_to_minutes(current["target_time"])
                    delta = step_minutes - target_minutes
                    if 0 < delta <= TIME_LATE_WINDOW_MINUTES:
                        state["completed"] = True
                        state["completed_at"] = step["id"]
                        state["result"] = "late"
                        state["completion_had_required_query"] = had_required_query
                        metrics["late"] += 1
                    else:
                        metrics["false_alarm"] += 1
                else:
                    cue_step_idx = state["cue_step_idx"]
                    if cue_step_idx is not None:
                        delta_steps = step_idx - cue_step_idx
                        if 0 < delta_steps <= EVENT_LATE_WINDOW_STEPS:
                            state["completed"] = True
                            state["completed_at"] = step["id"]
                            state["result"] = "late"
                            state["completion_had_required_query"] = had_required_query
                            metrics["late"] += 1
                        else:
                            metrics["false_alarm"] += 1
                    else:
                        metrics["false_alarm"] += 1
                if state["updated"] and task_id not in due_now:
                    metrics["update_violation"] += 1
                if due_now and task_id not in due_now:
                    metrics["wrong_content"] += 1

    for task_id, state in task_states.items():
        if state["canceled"]:
            if state["has_update"]:
                metrics["update_canceled"] += 1
            continue
        required_channel = required_monitor_channel(state["current"])
        monitor_key = (
            "proactive_monitoring_required"
            if requires_state_monitoring(state["current"], state_visibility)
            else "no_proactive_monitoring"
        )
        by_monitoring[monitor_key]["total"] += 1
        if monitor_key == "proactive_monitoring_required" and required_channel:
            by_monitoring_channel.setdefault(
                required_channel, {"hit": 0, "late": 0, "miss": 0, "total": 0}
            )
            by_monitoring_channel[required_channel]["total"] += 1
        if not state["completed"]:
            metrics["miss"] += 1
            task = state["task"]
            if task.get("cross_day"):
                metrics["cross_day_miss"] += 1
            if state["has_update"]:
                metrics["update_miss"] += 1
            if task["regular"]:
                by_regular["regular"]["miss"] += 1
            else:
                by_regular["irregular"]["miss"] += 1
            by_type[state["current"]["type"]]["miss"] += 1
            by_monitoring[monitor_key]["miss"] += 1
            if monitor_key == "proactive_monitoring_required" and required_channel:
                by_monitoring_channel[required_channel]["miss"] += 1
        else:
            task = state["task"]
            if state["result"] == "hit":
                if task["regular"]:
                    by_regular["regular"]["hit"] += 1
                else:
                    by_regular["irregular"]["hit"] += 1
                by_type[state["current"]["type"]]["hit"] += 1
                by_monitoring[monitor_key]["hit"] += 1
                if monitor_key == "proactive_monitoring_required" and required_channel:
                    by_monitoring_channel[required_channel]["hit"] += 1
                if task.get("cross_day"):
                    metrics["cross_day_hit"] += 1
                if state["has_update"]:
                    metrics["update_hit"] += 1
            elif state["result"] == "late":
                if task["regular"]:
                    by_regular["regular"]["late"] += 1
                else:
                    by_regular["irregular"]["late"] += 1
                by_type[state["current"]["type"]]["late"] += 1
                by_monitoring[monitor_key]["late"] += 1
                if monitor_key == "proactive_monitoring_required" and required_channel:
                    by_monitoring_channel[required_channel]["late"] += 1
                if task.get("cross_day"):
                    metrics["cross_day_late"] += 1
                if state["has_update"]:
                    metrics["update_late"] += 1

    return metrics, by_type, by_regular, by_monitoring, by_monitoring_channel


def score_log(scenario, log_entries):
    actions_by_day = {day["name"]: [] for day in scenario["days"]}
    for entry in log_entries:
        actions_by_day[entry["day"]].append(entry)

    summary = {
        "hit": 0,
        "late": 0,
        "miss": 0,
        "false_alarm": 0,
        "commission": 0,
        "wrong_content": 0,
        "check_time_calls": 0,
        "state_query_calls": 0,
        "overkill_steps": 0,
        "chosen_tasks": 0,
        "dependency_violation": 0,
        "cross_day_total": 0,
        "cross_day_hit": 0,
        "cross_day_late": 0,
        "cross_day_miss": 0,
        "update_total": 0,
        "update_hit": 0,
        "update_late": 0,
        "update_miss": 0,
        "update_violation": 0,
        "update_canceled": 0,
        "canceled_total": 0,
        "exact_set_match_steps": 0,
        "exact_set_mismatch_steps": 0,
        "exact_set_match_reward": 0,
        "set_tp": 0,
        "set_fp": 0,
        "set_fn": 0,
    }
    summary_steps = 0
    per_day = {}
    state_visibility = normalize_state_visibility(scenario)
    updates_by_day = build_updates_by_day(scenario)
    for day in scenario["days"]:
        day_updates = updates_by_day.get(day["name"], {"pre": [], "by_step": {}})
        day_actions = actions_by_day[day["name"]]
        metrics, by_type, by_regular, by_monitoring, by_monitoring_channel = score_day(
            day,
            day_actions,
            day_updates.get("pre", []),
            day_updates.get("by_step", {}),
            state_visibility,
        )
        state_query_calls_by_channel = {}
        for action in day_actions:
            add_channel_counts(
                state_query_calls_by_channel, get_action_state_query_counts(action)
            )
        per_day[day["name"]] = {
            "metrics": metrics,
            "by_type": by_type,
            "by_regular": by_regular,
            "by_monitoring": by_monitoring,
            "by_monitoring_channel": by_monitoring_channel,
            "state_query_calls_by_channel": state_query_calls_by_channel,
            "steps": len(day.get("steps", [])),
        }
        for key in summary:
            summary[key] += metrics[key]
        summary_steps += len(day.get("steps", []))
    return summary, per_day, summary_steps


def read_log_with_metadata(path):
    entries = []
    run_metadata = None
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if (
                isinstance(payload, dict)
                and payload.get("record_type") == RUN_METADATA_RECORD_TYPE
            ):
                if run_metadata is None:
                    run_metadata = payload
                continue
            entries.append(payload)
    return entries, run_metadata


def read_log(path):
    entries, _ = read_log_with_metadata(path)
    return entries


def format_rate(value, total):
    if total <= 0:
        return "n/a"
    return f"{(value / total) * 100:.1f}%"


def format_step_average(value, steps):
    if steps <= 0:
        return "n/a"
    return f"{value / steps:.3f}"


def format_set_precision(tp, fp):
    total = tp + fp
    if total <= 0:
        return "n/a"
    return f"{(tp / total) * 100:.1f}%"


def format_set_recall(tp, fn):
    total = tp + fn
    if total <= 0:
        return "n/a"
    return f"{(tp / total) * 100:.1f}%"


def format_set_f1(tp, fp, fn):
    denom = (2 * tp) + fp + fn
    if denom <= 0:
        return "n/a"
    return f"{(2 * tp / denom) * 100:.1f}%"


def format_precision(hit_count, chosen_count):
    if chosen_count <= 0:
        return "n/a"
    return f"{(hit_count / chosen_count) * 100:.1f}%"


def format_precision_any(hit_count, late_count, chosen_count):
    if chosen_count <= 0:
        return "n/a"
    return f"{((hit_count + late_count) / chosen_count) * 100:.1f}%"


def aggregate_type_counts(per_day):
    totals = {}
    for details in per_day.values():
        for task_type, counts in details["by_type"].items():
            totals.setdefault(task_type, {"hit": 0, "total": 0})
            totals[task_type]["hit"] += counts["hit"]
            totals[task_type]["total"] += counts["total"]
    return totals


def aggregate_monitoring_counts(per_day):
    totals = {
        "no_proactive_monitoring": {"hit": 0, "late": 0, "miss": 0, "total": 0},
        "proactive_monitoring_required": {"hit": 0, "late": 0, "miss": 0, "total": 0},
    }
    for details in per_day.values():
        by_monitoring = details.get("by_monitoring", {})
        for key in totals:
            counts = by_monitoring.get(
                key, {"hit": 0, "late": 0, "miss": 0, "total": 0}
            )
            for metric in ("hit", "late", "miss", "total"):
                totals[key][metric] += counts.get(metric, 0)
    return totals


def aggregate_monitoring_channel_counts(per_day):
    totals = {}
    for details in per_day.values():
        by_channel = details.get("by_monitoring_channel", {})
        for channel, counts in by_channel.items():
            totals.setdefault(channel, {"hit": 0, "late": 0, "miss": 0, "total": 0})
            for key in ("hit", "late", "miss", "total"):
                totals[channel][key] += counts.get(key, 0)
    return totals


def aggregate_state_query_call_counts(per_day):
    totals = {}
    for details in per_day.values():
        channel_counts = details.get("state_query_calls_by_channel", {})
        if isinstance(channel_counts, dict):
            add_channel_counts(totals, channel_counts)
    return totals


def compute_event_time_rates(by_type):
    event = by_type.get("event", {"hit": 0, "total": 0})
    time = by_type.get("time", {"hit": 0, "total": 0})
    time_check = by_type.get("time_check", {"hit": 0, "total": 0})
    time_hit = time["hit"] + time_check["hit"]
    time_total = time["total"] + time_check["total"]
    return event["hit"], event["total"], time_hit, time_total


def print_report(summary, per_day, summary_steps, run_metadata=None):
    total_tasks = 0
    for details in per_day.values():
        day_total = sum(
            counts["total"] for counts in details["by_type"].values()
        )
        total_tasks += day_total
    overall_by_type = aggregate_type_counts(per_day)
    overall_by_monitoring = aggregate_monitoring_counts(per_day)
    overall_by_monitoring_channel = aggregate_monitoring_channel_counts(per_day)
    overall_state_query_calls_by_channel = aggregate_state_query_call_counts(per_day)
    event_hit, event_total, time_hit, time_total = compute_event_time_rates(
        overall_by_type
    )
    monitoring_no_proactive = overall_by_monitoring["no_proactive_monitoring"]
    monitoring_proactive = overall_by_monitoring["proactive_monitoring_required"]

    print("=== PM-Bench Report ===")
    if run_metadata:
        print(
            "Run timing: "
            f"start {run_metadata.get('started_at_utc', 'n/a')} | "
            f"end {run_metadata.get('finished_at_utc', 'n/a')} | "
            f"duration {format_duration_seconds(run_metadata.get('duration_seconds'))}"
        )
    print(
        f"Hit: {summary['hit']} | Late: {summary['late']} | Miss: {summary['miss']} | "
        f"False alarms: {summary['false_alarm']} | Commission: {summary['commission']} | "
        f"Wrong-content: {summary['wrong_content']} | Dependency violations: {summary['dependency_violation']} | "
        f"Overkill steps: {summary['overkill_steps']} | "
        f"state query calls: {summary['state_query_calls']} | "
        f"check_time calls: {summary['check_time_calls']} | "
        f"Actions: {summary['chosen_tasks']}"
    )
    print(
        f"Exact-set: matches {summary['exact_set_match_steps']} | "
        f"mismatches {summary['exact_set_mismatch_steps']} | "
        f"reward {summary['exact_set_match_reward']}"
    )
    print(
        f"Set micro: TP {summary['set_tp']} | FP {summary['set_fp']} | FN {summary['set_fn']}"
    )
    print(
        f"Cross-day: hit {summary['cross_day_hit']} | late {summary['cross_day_late']} | "
        f"miss {summary['cross_day_miss']} | total {summary['cross_day_total']}"
    )
    print(
        f"Updates: hit {summary['update_hit']} | late {summary['update_late']} | "
        f"miss {summary['update_miss']} | canceled {summary['update_canceled']} | "
        f"total {summary['update_total']} | violations {summary['update_violation']}"
    )
    print(
        f"Rates: hit {format_rate(summary['hit'], total_tasks)} | "
        f"late {format_rate(summary['late'], total_tasks)} | "
        f"miss {format_rate(summary['miss'], total_tasks)} | "
        f"false alarm/step {format_rate(summary['false_alarm'], summary_steps)} | "
        f"commission {format_rate(summary['commission'], total_tasks)} | "
        f"wrong-content {format_rate(summary['wrong_content'], total_tasks)} | "
        f"dependency/step {format_rate(summary['dependency_violation'], summary_steps)} | "
        f"overkill/step {format_rate(summary['overkill_steps'], summary_steps)} | "
        f"cross-day miss {format_rate(summary['cross_day_miss'], summary['cross_day_total'])} | "
        f"update miss {format_rate(summary['update_miss'], summary['update_total'] - summary['update_canceled'])} | "
        f"precision_hit {format_precision(summary['hit'], summary['chosen_tasks'])} | "
        f"precision_any {format_precision_any(summary['hit'], summary['late'], summary['chosen_tasks'])} | "
        f"exact-set match rate {format_rate(summary['exact_set_match_steps'], summary_steps)} | "
        f"exact-set avg reward {format_step_average(summary['exact_set_match_reward'], summary_steps)} | "
        f"set_precision {format_set_precision(summary['set_tp'], summary['set_fp'])} | "
        f"set_recall {format_set_recall(summary['set_tp'], summary['set_fn'])} | "
        f"set_f1 {format_set_f1(summary['set_tp'], summary['set_fp'], summary['set_fn'])}"
    )
    if overall_state_query_calls_by_channel:
        parts = [
            f"{channel}={overall_state_query_calls_by_channel[channel]}"
            for channel in sorted(overall_state_query_calls_by_channel)
        ]
        print("State query calls by channel (overall): " + " | ".join(parts))
    else:
        print("State query calls by channel (overall): (none)")
    print(
        f"Hit rates (by modality): event {format_rate(event_hit, event_total)} | "
        f"time {format_rate(time_hit, time_total)}"
    )
    print(
        "Monitoring categories (overall): "
        f"no-proactive hit {monitoring_no_proactive['hit']} | "
        f"late {monitoring_no_proactive['late']} | "
        f"miss {monitoring_no_proactive['miss']} | "
        f"total {monitoring_no_proactive['total']} | "
        f"hit-rate {format_rate(monitoring_no_proactive['hit'], monitoring_no_proactive['total'])} | "
        f"any-rate {format_rate(monitoring_no_proactive['hit'] + monitoring_no_proactive['late'], monitoring_no_proactive['total'])}"
    )
    print(
        "Monitoring categories (overall): "
        f"proactive-required hit {monitoring_proactive['hit']} | "
        f"late {monitoring_proactive['late']} | "
        f"miss {monitoring_proactive['miss']} | "
        f"total {monitoring_proactive['total']} | "
        f"hit-rate(no-late-credit) {format_rate(monitoring_proactive['hit'], monitoring_proactive['total'])} | "
        f"any-rate {format_rate(monitoring_proactive['hit'] + monitoring_proactive['late'], monitoring_proactive['total'])}"
    )
    print("Proactive-required by channel (no late credit):")
    if overall_by_monitoring_channel:
        for channel in sorted(overall_by_monitoring_channel):
            counts = overall_by_monitoring_channel[channel]
            print(
                f"  {channel}: hit {counts['hit']} | late {counts['late']} | "
                f"miss {counts['miss']} | total {counts['total']} | "
                f"hit-rate(no-late-credit) {format_rate(counts['hit'], counts['total'])} | "
                f"any-rate {format_rate(counts['hit'] + counts['late'], counts['total'])}"
            )
    else:
        print("  (none)")
    for day_name, details in per_day.items():
        metrics = details["metrics"]
        day_total = sum(counts["total"] for counts in details["by_type"].values())
        day_steps = details.get("steps", 0)
        day_monitoring = details.get("by_monitoring", {})
        day_monitoring_channel = details.get("by_monitoring_channel", {})
        day_state_query_calls_by_channel = details.get(
            "state_query_calls_by_channel", {}
        )
        day_event_hit, day_event_total, day_time_hit, day_time_total = (
            compute_event_time_rates(details["by_type"])
        )
        day_monitor_no_proactive = day_monitoring.get(
            "no_proactive_monitoring", {"hit": 0, "late": 0, "miss": 0, "total": 0}
        )
        day_monitor_proactive = day_monitoring.get(
            "proactive_monitoring_required",
            {"hit": 0, "late": 0, "miss": 0, "total": 0},
        )
        print(f"\n{day_name}")
        print(
            f"  Hit: {metrics['hit']} | Late: {metrics['late']} | Miss: {metrics['miss']} | "
            f"False alarms: {metrics['false_alarm']} | Commission: {metrics['commission']} | "
            f"Wrong-content: {metrics['wrong_content']} | Dependency violations: {metrics['dependency_violation']} | "
            f"Overkill steps: {metrics['overkill_steps']} | check_time calls: {metrics['check_time_calls']} | "
            f"Actions: {metrics['chosen_tasks']}"
        )
        print(
            f"  Exact-set: matches {metrics['exact_set_match_steps']} | "
            f"mismatches {metrics['exact_set_mismatch_steps']} | "
            f"reward {metrics['exact_set_match_reward']}"
        )
        print(
            f"  Set micro: TP {metrics['set_tp']} | FP {metrics['set_fp']} | FN {metrics['set_fn']}"
        )
        print(
            f"  Cross-day: hit {metrics['cross_day_hit']} | late {metrics['cross_day_late']} | "
            f"miss {metrics['cross_day_miss']} | total {metrics['cross_day_total']}"
        )
        print(
            f"  Updates: hit {metrics['update_hit']} | late {metrics['update_late']} | "
            f"miss {metrics['update_miss']} | canceled {metrics['update_canceled']} | "
            f"total {metrics['update_total']} | violations {metrics['update_violation']}"
        )
        print(
            f"  Rates: hit {format_rate(metrics['hit'], day_total)} | "
            f"late {format_rate(metrics['late'], day_total)} | "
            f"miss {format_rate(metrics['miss'], day_total)} | "
            f"false alarm/step {format_rate(metrics['false_alarm'], day_steps)} | "
            f"commission {format_rate(metrics['commission'], day_total)} | "
            f"wrong-content {format_rate(metrics['wrong_content'], day_total)} | "
            f"dependency/step {format_rate(metrics['dependency_violation'], day_steps)} | "
            f"overkill/step {format_rate(metrics['overkill_steps'], day_steps)} | "
            f"cross-day miss {format_rate(metrics['cross_day_miss'], metrics['cross_day_total'])} | "
            f"update miss {format_rate(metrics['update_miss'], metrics['update_total'] - metrics['update_canceled'])} | "
            f"precision_hit {format_precision(metrics['hit'], metrics['chosen_tasks'])} | "
            f"precision_any {format_precision_any(metrics['hit'], metrics['late'], metrics['chosen_tasks'])} | "
            f"exact-set match rate {format_rate(metrics['exact_set_match_steps'], day_steps)} | "
            f"exact-set avg reward {format_step_average(metrics['exact_set_match_reward'], day_steps)} | "
            f"set_precision {format_set_precision(metrics['set_tp'], metrics['set_fp'])} | "
            f"set_recall {format_set_recall(metrics['set_tp'], metrics['set_fn'])} | "
            f"set_f1 {format_set_f1(metrics['set_tp'], metrics['set_fp'], metrics['set_fn'])}"
        )
        if day_state_query_calls_by_channel:
            parts = [
                f"{channel}={day_state_query_calls_by_channel[channel]}"
                for channel in sorted(day_state_query_calls_by_channel)
            ]
            print("  State query calls by channel: " + " | ".join(parts))
        else:
            print("  State query calls by channel: (none)")
        print(
            f"  Hit rates (by modality): event {format_rate(day_event_hit, day_event_total)} | "
            f"time {format_rate(day_time_hit, day_time_total)}"
        )
        print(
            "  Monitoring categories: "
            f"no-proactive hit {day_monitor_no_proactive['hit']} | "
            f"late {day_monitor_no_proactive['late']} | "
            f"miss {day_monitor_no_proactive['miss']} | "
            f"total {day_monitor_no_proactive['total']} | "
            f"hit-rate {format_rate(day_monitor_no_proactive['hit'], day_monitor_no_proactive['total'])} | "
            f"any-rate {format_rate(day_monitor_no_proactive['hit'] + day_monitor_no_proactive['late'], day_monitor_no_proactive['total'])}"
        )
        print(
            "  Monitoring categories: "
            f"proactive-required hit {day_monitor_proactive['hit']} | "
            f"late {day_monitor_proactive['late']} | "
            f"miss {day_monitor_proactive['miss']} | "
            f"total {day_monitor_proactive['total']} | "
            f"hit-rate(no-late-credit) {format_rate(day_monitor_proactive['hit'], day_monitor_proactive['total'])} | "
            f"any-rate {format_rate(day_monitor_proactive['hit'] + day_monitor_proactive['late'], day_monitor_proactive['total'])}"
        )
        print("  By type:")
        for task_type, counts in details["by_type"].items():
            print(
                f"    {task_type}: hit {counts['hit']} | late {counts['late']} | "
                f"miss {counts['miss']} | total {counts['total']}"
            )
        print("  By regularity:")
        for key, counts in details["by_regular"].items():
            print(
                f"    {key}: hit {counts['hit']} | late {counts['late']} | "
                f"miss {counts['miss']} | total {counts['total']}"
            )
        print("  By monitoring:")
        for key, counts in day_monitoring.items():
            print(
                f"    {key}: hit {counts['hit']} | late {counts['late']} | "
                f"miss {counts['miss']} | total {counts['total']}"
            )
        print("  Proactive required by channel:")
        if day_monitoring_channel:
            for channel in sorted(day_monitoring_channel):
                counts = day_monitoring_channel[channel]
                print(
                    f"    {channel}: hit {counts['hit']} | late {counts['late']} | "
                    f"miss {counts['miss']} | total {counts['total']}"
                )
        else:
            print("    (none)")


def build_markdown_table(headers, rows):
    """Build a GitHub-flavored markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def build_markdown_report(summary, per_day, summary_steps, run_metadata=None):
    """Build a markdown score report with renderable tables."""
    total_tasks = 0
    for details in per_day.values():
        day_total = sum(counts["total"] for counts in details["by_type"].values())
        total_tasks += day_total

    overall_by_type = aggregate_type_counts(per_day)
    overall_by_monitoring = aggregate_monitoring_counts(per_day)
    overall_by_monitoring_channel = aggregate_monitoring_channel_counts(per_day)
    overall_state_query_calls_by_channel = aggregate_state_query_call_counts(per_day)
    event_hit, event_total, time_hit, time_total = compute_event_time_rates(
        overall_by_type
    )
    monitoring_no_proactive = overall_by_monitoring["no_proactive_monitoring"]
    monitoring_proactive = overall_by_monitoring["proactive_monitoring_required"]

    counts_line = (
        f"Hit: {summary['hit']} | Late: {summary['late']} | Miss: {summary['miss']} | "
        f"False alarms: {summary['false_alarm']} | Commission: {summary['commission']} | "
        f"Wrong-content: {summary['wrong_content']} | Dependency violations: {summary['dependency_violation']} | "
        f"Overkill steps: {summary['overkill_steps']} | "
        f"state query calls: {summary['state_query_calls']} | "
        f"check_time calls: {summary['check_time_calls']} | "
        f"Actions: {summary['chosen_tasks']}"
    )
    exact_set_line = (
        f"Exact-set: matches {summary['exact_set_match_steps']} | "
        f"mismatches {summary['exact_set_mismatch_steps']} | "
        f"reward {summary['exact_set_match_reward']}"
    )
    set_micro_line = (
        f"Set micro: TP {summary['set_tp']} | FP {summary['set_fp']} | FN {summary['set_fn']}"
    )
    cross_day_line = (
        f"Cross-day: hit {summary['cross_day_hit']} | late {summary['cross_day_late']} | "
        f"miss {summary['cross_day_miss']} | total {summary['cross_day_total']}"
    )
    updates_line = (
        f"Updates: hit {summary['update_hit']} | late {summary['update_late']} | "
        f"miss {summary['update_miss']} | canceled {summary['update_canceled']} | "
        f"total {summary['update_total']} | violations {summary['update_violation']}"
    )
    rates_line = (
        f"Rates: hit {format_rate(summary['hit'], total_tasks)} | "
        f"late {format_rate(summary['late'], total_tasks)} | "
        f"miss {format_rate(summary['miss'], total_tasks)} | "
        f"false alarm/step {format_rate(summary['false_alarm'], summary_steps)} | "
        f"commission {format_rate(summary['commission'], total_tasks)} | "
        f"wrong-content {format_rate(summary['wrong_content'], total_tasks)} | "
        f"dependency/step {format_rate(summary['dependency_violation'], summary_steps)} | "
        f"overkill/step {format_rate(summary['overkill_steps'], summary_steps)} | "
        f"cross-day miss {format_rate(summary['cross_day_miss'], summary['cross_day_total'])} | "
        f"update miss {format_rate(summary['update_miss'], summary['update_total'] - summary['update_canceled'])} | "
        f"precision_hit {format_precision(summary['hit'], summary['chosen_tasks'])} | "
        f"precision_any {format_precision_any(summary['hit'], summary['late'], summary['chosen_tasks'])} | "
        f"exact-set match rate {format_rate(summary['exact_set_match_steps'], summary_steps)} | "
        f"exact-set avg reward {format_step_average(summary['exact_set_match_reward'], summary_steps)} | "
        f"set_precision {format_set_precision(summary['set_tp'], summary['set_fp'])} | "
        f"set_recall {format_set_recall(summary['set_tp'], summary['set_fn'])} | "
        f"set_f1 {format_set_f1(summary['set_tp'], summary['set_fp'], summary['set_fn'])}"
    )
    modality_line = (
        f"Hit rates (by modality): event {format_rate(event_hit, event_total)} | "
        f"time {format_rate(time_hit, time_total)}"
    )

    lines = ["# PM-Bench score report", "", "## Summary", ""]
    # Keep these canonical lines for backward-compatible parsing.
    lines.extend(
        [
            counts_line,
            exact_set_line,
            set_micro_line,
            cross_day_line,
            updates_line,
            rates_line,
            modality_line,
            "",
        ]
    )

    if run_metadata:
        timing_rows = [
            ["Started (UTC)", run_metadata.get("started_at_utc", "n/a")],
            ["Finished (UTC)", run_metadata.get("finished_at_utc", "n/a")],
            ["Duration", format_duration_seconds(run_metadata.get("duration_seconds"))],
        ]
        lines.extend(
            [
                "## Run Timing",
                "",
                build_markdown_table(["Field", "Value"], timing_rows),
                "",
            ]
        )

    overall_counts_rows = [
        ["Hit", summary["hit"]],
        ["Late", summary["late"]],
        ["Miss", summary["miss"]],
        ["False alarms", summary["false_alarm"]],
        ["Commission", summary["commission"]],
        ["Wrong-content", summary["wrong_content"]],
        ["Dependency violations", summary["dependency_violation"]],
        ["Overkill steps", summary["overkill_steps"]],
        ["State query calls", summary["state_query_calls"]],
        ["Check_time calls", summary["check_time_calls"]],
        ["Actions", summary["chosen_tasks"]],
        ["Exact-set matches", summary["exact_set_match_steps"]],
        ["Exact-set mismatches", summary["exact_set_mismatch_steps"]],
        ["Exact-set reward", summary["exact_set_match_reward"]],
        ["Set TP", summary["set_tp"]],
        ["Set FP", summary["set_fp"]],
        ["Set FN", summary["set_fn"]],
    ]
    lines.extend(
        [
            "## Overall Counts",
            "",
            build_markdown_table(["Metric", "Value"], overall_counts_rows),
            "",
        ]
    )

    overall_state_query_rows = []
    for channel in sorted(overall_state_query_calls_by_channel):
        overall_state_query_rows.append(
            [channel, overall_state_query_calls_by_channel[channel]]
        )
    if not overall_state_query_rows:
        overall_state_query_rows.append(["(none)", 0])
    lines.extend(
        [
            "## State Query Calls by Channel (Overall)",
            "",
            build_markdown_table(["Channel", "Calls"], overall_state_query_rows),
            "",
        ]
    )

    rate_rows = [
        ["Hit rate", format_rate(summary["hit"], total_tasks)],
        ["Late rate", format_rate(summary["late"], total_tasks)],
        ["Miss rate", format_rate(summary["miss"], total_tasks)],
        ["False alarm/step", format_rate(summary["false_alarm"], summary_steps)],
        ["Commission rate", format_rate(summary["commission"], total_tasks)],
        ["Wrong-content rate", format_rate(summary["wrong_content"], total_tasks)],
        ["Dependency/step", format_rate(summary["dependency_violation"], summary_steps)],
        ["Overkill/step", format_rate(summary["overkill_steps"], summary_steps)],
        ["Cross-day miss rate", format_rate(summary["cross_day_miss"], summary["cross_day_total"])],
        [
            "Update miss rate",
            format_rate(
                summary["update_miss"],
                summary["update_total"] - summary["update_canceled"],
            ),
        ],
        ["Precision hit", format_precision(summary["hit"], summary["chosen_tasks"])],
        [
            "Precision any",
            format_precision_any(
                summary["hit"], summary["late"], summary["chosen_tasks"]
            ),
        ],
        [
            "Exact-set match rate",
            format_rate(summary["exact_set_match_steps"], summary_steps),
        ],
        [
            "Exact-set avg reward",
            format_step_average(summary["exact_set_match_reward"], summary_steps),
        ],
        [
            "Set precision",
            format_set_precision(summary["set_tp"], summary["set_fp"]),
        ],
        [
            "Set recall",
            format_set_recall(summary["set_tp"], summary["set_fn"]),
        ],
        [
            "Set F1",
            format_set_f1(summary["set_tp"], summary["set_fp"], summary["set_fn"]),
        ],
    ]
    lines.extend(
        ["## Overall Rates", "", build_markdown_table(["Metric", "Value"], rate_rows), ""]
    )

    lines.extend(
        [
            "## Modality Hit Rates",
            "",
            build_markdown_table(
                ["Modality", "Hit", "Total", "Hit rate"],
                [
                    ["Event", event_hit, event_total, format_rate(event_hit, event_total)],
                    ["Time (time + time_check)", time_hit, time_total, format_rate(time_hit, time_total)],
                ],
            ),
            "",
        ]
    )

    monitoring_rows = [
        [
            "no_proactive_monitoring",
            monitoring_no_proactive["hit"],
            monitoring_no_proactive["late"],
            monitoring_no_proactive["miss"],
            monitoring_no_proactive["total"],
            format_rate(
                monitoring_no_proactive["hit"], monitoring_no_proactive["total"]
            ),
            format_rate(
                monitoring_no_proactive["hit"] + monitoring_no_proactive["late"],
                monitoring_no_proactive["total"],
            ),
        ],
        [
            "proactive_monitoring_required",
            monitoring_proactive["hit"],
            monitoring_proactive["late"],
            monitoring_proactive["miss"],
            monitoring_proactive["total"],
            format_rate(monitoring_proactive["hit"], monitoring_proactive["total"]),
            format_rate(
                monitoring_proactive["hit"] + monitoring_proactive["late"],
                monitoring_proactive["total"],
            ),
        ],
    ]
    lines.extend(
        [
            "## Monitoring Categories",
            "",
            build_markdown_table(
                [
                    "Category",
                    "Hit",
                    "Late",
                    "Miss",
                    "Total",
                    "Hit rate",
                    "Any rate (hit+late)",
                ],
                monitoring_rows,
            ),
            "",
            "Note: `proactive_monitoring_required` hit rate is no-late-credit by design.",
            "",
        ]
    )

    channel_rows = []
    for channel in sorted(overall_by_monitoring_channel):
        counts = overall_by_monitoring_channel[channel]
        channel_rows.append(
            [
                channel,
                counts["hit"],
                counts["late"],
                counts["miss"],
                counts["total"],
                format_rate(counts["hit"], counts["total"]),
                format_rate(
                    counts["hit"] + counts["late"],
                    counts["total"],
                ),
            ]
        )
    if not channel_rows:
        channel_rows.append(["(none)", 0, 0, 0, 0, "n/a", "n/a"])
    lines.extend(
        [
            "## Proactive Required by Channel",
            "",
            build_markdown_table(
                [
                    "Channel",
                    "Hit",
                    "Late",
                    "Miss",
                    "Total",
                    "Hit rate (no late credit)",
                    "Any rate (hit+late)",
                ],
                channel_rows,
            ),
            "",
        ]
    )

    per_day_rows = []
    for day_name, details in per_day.items():
        metrics = details["metrics"]
        day_total = sum(counts["total"] for counts in details["by_type"].values())
        day_steps = details.get("steps", 0)
        day_event_hit, day_event_total, day_time_hit, day_time_total = (
            compute_event_time_rates(details["by_type"])
        )
        day_monitoring = details.get("by_monitoring", {})
        day_monitor_no_proactive = day_monitoring.get(
            "no_proactive_monitoring", {"hit": 0, "late": 0, "miss": 0, "total": 0}
        )
        day_monitor_proactive = day_monitoring.get(
            "proactive_monitoring_required",
            {"hit": 0, "late": 0, "miss": 0, "total": 0},
        )
        per_day_rows.append(
            [
                day_name,
                metrics["hit"],
                metrics["late"],
                metrics["miss"],
                format_rate(metrics["hit"], day_total),
                format_rate(metrics["late"], day_total),
                format_rate(metrics["miss"], day_total),
                format_rate(metrics["false_alarm"], day_steps),
                format_rate(metrics["overkill_steps"], day_steps),
                format_rate(day_event_hit, day_event_total),
                format_rate(day_time_hit, day_time_total),
                format_rate(
                    day_monitor_no_proactive["hit"],
                    day_monitor_no_proactive["total"],
                ),
                format_rate(
                    day_monitor_proactive["hit"],
                    day_monitor_proactive["total"],
                ),
                format_rate(metrics["exact_set_match_steps"], day_steps),
                format_step_average(metrics["exact_set_match_reward"], day_steps),
                format_set_precision(metrics["set_tp"], metrics["set_fp"]),
                format_set_recall(metrics["set_tp"], metrics["set_fn"]),
                format_set_f1(metrics["set_tp"], metrics["set_fp"], metrics["set_fn"]),
            ]
        )
    lines.extend(
        [
            "## Per-Day Summary",
            "",
            build_markdown_table(
                [
                    "Day",
                    "Hit",
                    "Late",
                    "Miss",
                    "Hit rate",
                    "Late rate",
                    "Miss rate",
                    "False alarm/step",
                    "Overkill/step",
                    "Event hit rate",
                    "Time hit rate",
                    "No-proactive hit rate",
                    "Proactive hit rate (no late credit)",
                    "Exact-set match rate",
                    "Exact-set avg reward",
                    "Set precision",
                    "Set recall",
                    "Set F1",
                ],
                per_day_rows,
            ),
            "",
        ]
    )

    per_day_state_query_rows = []
    for day_name, details in per_day.items():
        day_counts = details.get("state_query_calls_by_channel", {})
        if isinstance(day_counts, dict) and day_counts:
            for channel in sorted(day_counts):
                per_day_state_query_rows.append([day_name, channel, day_counts[channel]])
        else:
            per_day_state_query_rows.append([day_name, "(none)", 0])
    lines.extend(
        [
            "## State Query Calls by Channel (Per Day)",
            "",
            build_markdown_table(["Day", "Channel", "Calls"], per_day_state_query_rows),
            "",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"


def validate_day(
    day,
    allowed_channels,
    require_action_text=False,
    require_structured_lures=False,
    min_lures_per_day=0,
):
    """Validate day structure, cues, task definitions, and state-channel events."""
    errors = []
    warnings = []

    step_ids = set()
    task_ids = set()
    cue_ids = set()
    state_cue_ids_by_channel = {channel: set() for channel in allowed_channels}
    state_event_records = {channel: [] for channel in allowed_channels}
    step_times = []
    step_minutes = []
    action_text_to_task_ids = {}

    for step_idx, step in enumerate(day.get("steps", [])):
        step_id = step.get("id")
        if step_id in step_ids:
            errors.append(f"{day['name']}: duplicate step id '{step_id}'")
        step_ids.add(step_id)

        step_time = step.get("time")
        step_times.append(step_time)
        step_minutes.append(time_to_minutes(step_time))

        cue_ids.update(step.get("cues", []))
        state_events = step.get("state_events", {})
        if state_events:
            if not isinstance(state_events, dict):
                errors.append(f"{day['name']} {step_id}: state_events must be an object")
            else:
                for channel, events in state_events.items():
                    if channel not in allowed_channels:
                        errors.append(
                            f"{day['name']} {step_id}: unknown state channel '{channel}'"
                        )
                        continue
                    if not isinstance(events, list):
                        errors.append(
                            f"{day['name']} {step_id}: state_events.{channel} must be a list"
                        )
                        continue
                    for event in events:
                        if isinstance(event, str):
                            if channel in PROCESS_CHANNELS:
                                errors.append(
                                    f"{day['name']} {step_id}: process channel '{channel}' events must be objects"
                                )
                                continue
                            state_cue_ids_by_channel[channel].add(event)
                            continue
                        if not isinstance(event, dict):
                            errors.append(
                                f"{day['name']} {step_id}: state event must be an object or string"
                            )
                            continue
                        event_id = event.get("id")
                        event_text = event.get("text")
                        if not event_id:
                            errors.append(
                                f"{day['name']} {step_id}: state event missing id"
                            )
                            continue
                        if not event_text:
                            errors.append(
                                f"{day['name']} {step_id}: state event '{event_id}' missing text"
                            )
                        if channel in PROCESS_CHANNELS:
                            phase = event.get("meta", {}).get("phase")
                            if phase not in PROCESS_PHASES - {"idle"}:
                                errors.append(
                                    f"{day['name']} {step_id}: process channel '{channel}' event '{event_id}' missing phase"
                                )
                            else:
                                state_event_records[channel].append(
                                    {"step_idx": step_idx, "id": event_id, "phase": phase}
                                )
                        elif channel in state_event_records:
                            state_event_records[channel].append(
                                {"step_idx": step_idx, "id": event_id, "phase": None}
                            )
                        state_cue_ids_by_channel[channel].add(event_id)

        options = step.get("options", [])
        if len(options) != 3:
            errors.append(f"{day['name']} {step_id}: options must have 3 items")
        else:
            prefixes = ("A)", "B)", "C)")
            for idx, prefix in enumerate(prefixes):
                if not options[idx].startswith(prefix):
                    warnings.append(
                        f"{day['name']} {step_id}: option {idx + 1} should start with '{prefix}'"
                    )
        for update in step.get("updates", []):
            action = update.get("action")
            if action not in ("cancel", "reschedule", "override"):
                errors.append(
                    f"{day['name']} {step_id}: unknown update action '{action}'"
                )
            cue_id = update.get("cue_id")
            if cue_id and cue_id not in step.get("cues", []):
                errors.append(
                    f"{day['name']} {step_id}: update cue '{cue_id}' not in step cues"
                )
            if action in ("reschedule", "override"):
                has_change = any(
                    update.get(field)
                    for field in ("new_cue_id", "new_target_time", "new_label", "new_type")
                )
                if not has_change:
                    errors.append(
                        f"{day['name']} {step_id}: update action '{action}' missing new_* fields"
                    )
                new_type = update.get("new_type")
                if new_type == "event" and not update.get("new_cue_id"):
                    errors.append(
                        f"{day['name']} {step_id}: event update missing new_cue_id"
                    )
                if new_type in ("time", "time_check") and not update.get("new_target_time"):
                    errors.append(
                        f"{day['name']} {step_id}: time update missing new_target_time"
                    )

    if step_minutes:
        for prev, curr in zip(step_minutes, step_minutes[1:]):
            if curr < prev:
                errors.append(f"{day['name']}: step times must be non-decreasing")
                break

    day_start_minutes = build_day_start_minutes(day)
    step_offsets = [m - day_start_minutes for m in step_minutes]

    for task in day.get("tasks", []):
        task_id = task.get("id")
        if task_id in task_ids:
            errors.append(f"{day['name']}: duplicate task id '{task_id}'")
        task_ids.add(task_id)

        action_text = task.get("action_text")
        if require_action_text and not action_text:
            errors.append(f"{day['name']} {task_id}: missing required action_text")
        if action_text is not None:
            if not isinstance(action_text, str) or not action_text.strip():
                errors.append(f"{day['name']} {task_id}: action_text must be a non-empty string")
            else:
                action_key = action_text.strip().lower()
                action_text_to_task_ids.setdefault(action_key, []).append(task_id)

        enc_type, enc_step = normalize_encoding(task["encoding"])
        if enc_type == "step" and enc_step not in step_ids:
            errors.append(
                f"{day['name']} {task_id}: encoding step '{enc_step}' not found"
            )

        task_type = task.get("type")
        if task_type == "event":
            cue_id = task.get("cue_id")
            cue_channel = task.get("cue_channel", "narrative")
            if not cue_id:
                errors.append(f"{day['name']} {task_id}: missing cue_id")
            elif cue_channel == "narrative":
                if cue_id not in cue_ids:
                    errors.append(
                        f"{day['name']} {task_id}: cue_id '{cue_id}' not present in any step cues"
                    )
            else:
                if cue_channel not in allowed_channels:
                    errors.append(
                        f"{day['name']} {task_id}: unknown cue_channel '{cue_channel}'"
                    )
                elif cue_id not in state_cue_ids_by_channel.get(cue_channel, set()):
                    errors.append(
                        f"{day['name']} {task_id}: cue_id '{cue_id}' not present in state_events for {cue_channel}"
                    )
                elif cue_channel in PROCESS_CHANNELS:
                    records = [r for r in state_event_records.get(cue_channel, []) if r["id"] == cue_id]
                    if not records:
                        errors.append(
                            f"{day['name']} {task_id}: process cue '{cue_id}' missing completion event"
                        )
                    else:
                        complete_steps = [r["step_idx"] for r in records if r["phase"] == "complete"]
                        if not complete_steps:
                            errors.append(
                                f"{day['name']} {task_id}: process cue '{cue_id}' missing complete phase"
                            )
                        else:
                            completion_step = min(complete_steps)
                            has_prior_progress = any(
                                r["phase"] == "in_progress" and r["step_idx"] < completion_step
                                for r in state_event_records.get(cue_channel, [])
                            )
                            if not has_prior_progress:
                                errors.append(
                                    f"{day['name']} {task_id}: process cue '{cue_id}' missing prior in_progress state"
                                )
        elif task_type == "time":
            target_time = task.get("target_time")
            if target_time not in step_times:
                errors.append(
                    f"{day['name']} {task_id}: target_time '{target_time}' not found in steps"
                )
        elif task_type == "time_check":
            target_time = task.get("target_time")
            window_before = task.get("window_before", 0)
            window_after = task.get("window_after", 0)
            target_offset = time_to_minutes(target_time) - day_start_minutes
            window_start = target_offset - window_before
            window_end = target_offset + window_after
            if not any(
                window_start <= offset <= window_end for offset in step_offsets
            ):
                errors.append(
                    f"{day['name']} {task_id}: time_check window does not overlap any step"
                )
        else:
            errors.append(f"{day['name']} {task_id}: unknown task type '{task_type}'")

    depends_map = {}
    for task in day.get("tasks", []):
        task_id = task.get("id")
        depends_on = task.get("depends_on")
        if not depends_on:
            continue
        if depends_on == task_id:
            errors.append(f"{day['name']} {task_id}: depends_on cannot reference itself")
            continue
        if depends_on not in task_ids:
            errors.append(f"{day['name']} {task_id}: depends_on '{depends_on}' not found")
            continue
        depends_map[task_id] = depends_on

    visiting = set()
    visited = set()

    def visit(node):
        if node in visited:
            return
        if node in visiting:
            errors.append(f"{day['name']}: dependency cycle detected at '{node}'")
            return
        visiting.add(node)
        dep = depends_map.get(node)
        if dep:
            visit(dep)
        visiting.remove(node)
        visited.add(node)

    for node in depends_map:
        visit(node)

    if require_action_text:
        for action_key, task_list in action_text_to_task_ids.items():
            if len(task_list) <= 1:
                continue
            errors.append(
                f"{day['name']}: duplicate action_text '{action_key}' across tasks {', '.join(sorted(task_list))}"
            )

    lures = day.get("lures", [])
    if lures and not isinstance(lures, list):
        errors.append(f"{day['name']}: lures must be a list")
        lures = []
    lure_seen = set()
    if lures:
        for lure in lures:
            if isinstance(lure, str):
                lure_id = lure
                if require_structured_lures:
                    errors.append(
                        f"{day['name']}: lure '{lure_id}' must be an object with id/action_text"
                    )
            elif isinstance(lure, dict):
                lure_id = lure.get("id")
                if not lure_id:
                    errors.append(f"{day['name']}: lure object missing id")
                    continue
                action_text = lure.get("action_text")
                if require_structured_lures and not action_text:
                    errors.append(
                        f"{day['name']}: lure '{lure_id}' missing required action_text"
                    )
                if action_text is not None:
                    if not isinstance(action_text, str) or not action_text.strip():
                        errors.append(
                            f"{day['name']}: lure '{lure_id}' action_text must be a non-empty string"
                        )
            else:
                errors.append(f"{day['name']}: lure must be a string or object")
                continue

            if lure_id in lure_seen:
                errors.append(f"{day['name']}: duplicate lure id '{lure_id}'")
            lure_seen.add(lure_id)
        overlap = task_ids.intersection(lure_seen)
        for lure in sorted(overlap):
            errors.append(f"{day['name']}: lure id '{lure}' overlaps a task id")
    if min_lures_per_day and len(lure_seen) < min_lures_per_day:
        errors.append(
            f"{day['name']}: expected at least {min_lures_per_day} lures, got {len(lure_seen)}"
        )

    for channel in PROCESS_CHANNELS:
        records = state_event_records.get(channel, [])
        if not records:
            continue
        in_progress_steps = [r["step_idx"] for r in records if r["phase"] == "in_progress"]
        for record in records:
            if record["phase"] != "complete":
                continue
            if not any(step < record["step_idx"] for step in in_progress_steps):
                errors.append(
                    f"{day['name']}: process channel '{channel}' has complete event '{record['id']}' without prior in_progress"
                )

    return errors, warnings


def base_instruction_label(label):
    cleaned = label.rstrip(".")
    if " after " in cleaned:
        cleaned = cleaned.split(" after ", 1)[0].rstrip()
    cleaned = re.sub(r"\s+when\s+.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(at|around)\s+\d{1,2}:\d{2}$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip().lower()


def find_today_only_lines(lines):
    if not lines:
        return None, []
    header_idx = None
    for idx, line in enumerate(lines):
        if isinstance(line, str) and not line.lstrip().startswith("- "):
            header_idx = idx
            break
    if header_idx is None:
        return None, []
    indices = []
    idx = header_idx + 1
    while idx < len(lines) and lines[idx].lstrip().startswith("- "):
        stripped = lines[idx].strip().lower()
        if not re.match(
            r"^-\s+on\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            stripped,
        ):
            indices.append(idx)
        idx += 1
    return header_idx, indices


def match_task_for_instruction(line, task_states):
    text = line.lower()
    best_id = None
    best_len = 0
    for task_id, state in task_states.items():
        label = base_instruction_label(state["task"]["label"])
        if not label:
            continue
        if label in text and len(label) > best_len:
            best_id = task_id
            best_len = len(label)
    return best_id


def infer_target_day_from_sentence(sentence, day_names, default_day):
    lowered = sentence.lower()
    for name in day_names:
        if name and name.lower() in lowered:
            return name
    return default_day


def sentence_is_reminder(sentence):
    lowered = sentence.strip().lower()
    return (
        "you remind yourself" in lowered
        or "you jot a note" in lowered
    )


def sentence_matches_label(sentence, label):
    if not label:
        return False
    sentence_lower = sentence.strip().lower()
    label_lower = label.rstrip(".").lower()
    return label_lower in sentence_lower


def instruction_mentions_full_label(line, label):
    if not line or not label:
        return False
    line_lower = line.strip().lower()
    label_lower = label.strip().rstrip(".").lower()
    return label_lower in line_lower


def validate_start_instructions_updates(scenario):
    errors = []
    updates_by_day = build_updates_by_day(scenario)
    for day in scenario.get("days", []):
        day_name = day.get("name")
        pre_updates = updates_by_day.get(day_name, {}).get("pre", [])
        if not pre_updates:
            continue
        header_idx, indices = find_today_only_lines(day.get("start_instructions", []))
        if header_idx is None or not indices:
            continue
        task_states = {task["id"]: init_task_state(task) for task in day.get("tasks", [])}
        for update in pre_updates:
            task_id = update.get("task_id")
            if task_id in task_states:
                apply_task_update(
                    task_states[task_id], update, task_states=task_states
                )
        for idx in indices:
            line = day["start_instructions"][idx]
            task_id = match_task_for_instruction(line, task_states)
            if not task_id:
                continue
            if task_states[task_id]["canceled"]:
                errors.append(
                    f"{day_name}: start_instructions mention canceled task '{task_id}'"
                )
    return errors


def validate_new_task_headers(scenario):
    errors = []
    first_mentions = {}
    for day in scenario.get("days", []):
        task_states = {task["id"]: init_task_state(task) for task in day.get("tasks", [])}
        for line in day.get("start_instructions", []):
            task_id = match_task_for_instruction(line, task_states)
            if task_id and task_id not in first_mentions:
                first_mentions[task_id] = day.get("name")
        for step in day.get("steps", []):
            text = step.get("text", "")
            if "You jot a note:" not in text:
                continue
            for sentence in text.split("."):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if "You jot a note:" in sentence:
                    sentence = sentence.split("You jot a note:", 1)[1].strip()
                task_id = match_task_for_instruction(sentence, task_states)
                if task_id and task_id not in first_mentions:
                    first_mentions[task_id] = day.get("name")

    for day in scenario.get("days", []):
        header_idx, indices = find_today_only_lines(day.get("start_instructions", []))
        if header_idx is None or not indices:
            continue
        for idx in indices:
            line = day["start_instructions"][idx]
            if line.strip().lower() == "- none.":
                continue
            task_states = {task["id"]: init_task_state(task) for task in day.get("tasks", [])}
            task_id = match_task_for_instruction(line, task_states)
            if not task_id:
                continue
            if first_mentions.get(task_id) != day.get("name"):
                errors.append(
                    f"{day['name']}: new-task header repeats previously announced task '{task_id}'"
                )
    return errors


def validate_no_canceled_task_mentions(scenario):
    errors = []
    day_names = [day.get("name") for day in scenario.get("days", [])]
    task_states_by_day = {
        day.get("name"): {
            task["id"]: init_task_state(task) for task in day.get("tasks", [])
        }
        for day in scenario.get("days", [])
    }
    for day in scenario.get("days", []):
        day_name = day.get("name")
        for step in day.get("steps", []):
            for update in step.get("updates", []):
                target_day = update.get("target_day", day_name)
                states = task_states_by_day.get(target_day)
                if not states:
                    continue
                task_id = update.get("task_id")
                if not task_id or task_id not in states:
                    continue
                apply_task_update(states[task_id], update, task_states=states)
            text = step.get("text", "")
            for sentence in text.split("."):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if not sentence_is_reminder(sentence):
                    continue
                target_day = infer_target_day_from_sentence(
                    sentence, day_names, day_name
                )
                states = task_states_by_day.get(target_day, {})
                if not states:
                    continue
                task_id = match_task_for_instruction(sentence, states)
                if not task_id:
                    continue
                if states[task_id]["canceled"]:
                    errors.append(
                        f"{day_name} {step['id']}: reminder mentions canceled task '{task_id}'"
                    )
                    continue
                state = states[task_id]
                if state["updated"] and not sentence_matches_label(
                    sentence, state["current"]["label"]
                ):
                    errors.append(
                        f"{day_name} {step['id']}: reminder mentions outdated task '{task_id}'"
                    )
    return errors


def validate_updates(scenario):
    errors = []
    tasks_by_day = {}
    cues_by_day = {}
    cue_step_idx_by_day = {}
    times_by_day = {}
    time_step_idx_by_day = {}
    start_minutes_by_day = {}
    offsets_by_day = {}

    for day in scenario.get("days", []):
        day_name = day["name"]
        tasks_by_day[day_name] = {task["id"]: task for task in day.get("tasks", [])}
        cues = set()
        cue_step_idx = {}
        times = []
        time_step_idx = {}
        minutes = []
        for step_idx, step in enumerate(day.get("steps", [])):
            cues.update(step.get("cues", []))
            for cue_id in step.get("cues", []):
                cue_step_idx.setdefault(cue_id, step_idx)
            times.append(step.get("time"))
            time_step_idx.setdefault(step.get("time"), step_idx)
            minutes.append(time_to_minutes(step.get("time")))
        cues_by_day[day_name] = cues
        cue_step_idx_by_day[day_name] = cue_step_idx
        times_by_day[day_name] = times
        time_step_idx_by_day[day_name] = time_step_idx
        start_minutes = build_day_start_minutes(day)
        start_minutes_by_day[day_name] = start_minutes
        offsets_by_day[day_name] = [m - start_minutes for m in minutes]

    for day in scenario.get("days", []):
        day_name = day["name"]
        for step_idx, step in enumerate(day.get("steps", [])):
            step_minutes = time_to_minutes(step.get("time"))
            for update in step.get("updates", []):
                target_day = update.get("target_day", day_name)
                if target_day not in tasks_by_day:
                    errors.append(
                        f"{day_name} {step['id']}: update target_day '{target_day}' not found"
                    )
                    continue
                task_id = update.get("task_id")
                if not task_id or task_id not in tasks_by_day[target_day]:
                    errors.append(
                        f"{day_name} {step['id']}: update task_id '{task_id}' not found in {target_day}"
                    )
                    continue
                task = tasks_by_day[target_day][task_id]
                new_cue_id = update.get("new_cue_id")
                if new_cue_id and new_cue_id not in cues_by_day[target_day]:
                    errors.append(
                        f"{day_name} {step['id']}: update cue '{new_cue_id}' not in {target_day} cues"
                    )
                if (
                    new_cue_id
                    and target_day == day_name
                    and new_cue_id in cue_step_idx_by_day[target_day]
                    and cue_step_idx_by_day[target_day][new_cue_id] <= step_idx
                ):
                    errors.append(
                        f"{day_name} {step['id']}: update cue '{new_cue_id}' must occur after the update step"
                    )
                new_target_time = update.get("new_target_time")
                if new_target_time:
                    if (
                        target_day == day_name
                        and time_to_minutes(new_target_time) <= step_minutes
                    ):
                        errors.append(
                            f"{day_name} {step['id']}: update time '{new_target_time}' must be later than the update step time '{step['time']}'"
                        )
                    effective_type = update.get("new_type", task.get("type"))
                    if effective_type == "time":
                        if new_target_time not in times_by_day[target_day]:
                            errors.append(
                                f"{day_name} {step['id']}: update time '{new_target_time}' not in {target_day} steps"
                            )
                        elif (
                            target_day == day_name
                            and time_step_idx_by_day[target_day][new_target_time] <= step_idx
                        ):
                            errors.append(
                                f"{day_name} {step['id']}: update time '{new_target_time}' must point to a later step in {target_day}"
                            )
                    elif effective_type == "time_check":
                        window_before = update.get(
                            "new_window_before", task.get("window_before", 0)
                        )
                        window_after = update.get(
                            "new_window_after", task.get("window_after", 0)
                        )
                        target_offset = (
                            time_to_minutes(new_target_time)
                            - start_minutes_by_day[target_day]
                        )
                        window_start = target_offset - window_before
                        window_end = target_offset + window_after
                        if not any(
                            window_start <= offset <= window_end
                            for offset in offsets_by_day[target_day]
                        ):
                            errors.append(
                                f"{day_name} {step['id']}: update time_check window does not overlap steps in {target_day}"
                            )

    return errors


def validate_groundtruth_labels(scenario):
    errors = []
    has_groundtruth = any(
        "groundtruth" in step
        for day in scenario.get("days", [])
        for step in day.get("steps", [])
    )
    if not has_groundtruth:
        return errors

    report = compute_scenario_groundtruth(scenario)
    if not report["solvable"]:
        errors.extend(report["issues"])

    for day in scenario.get("days", []):
        day_report = report["days"].get(day["name"], {})
        expected_by_step = day_report.get("step_groundtruth", {})
        for step in day.get("steps", []):
            expected = expected_by_step.get(step["id"], {"status": "none", "actions": []})
            actual = step.get("groundtruth")
            if not isinstance(actual, dict):
                errors.append(f"{day['name']} {step['id']}: groundtruth must be an object")
                continue
            if actual.get("status") != expected["status"]:
                errors.append(
                    f"{day['name']} {step['id']}: groundtruth status mismatch "
                    f"(expected {expected['status']}, got {actual.get('status')})"
                )
            actual_actions = actual.get("actions")
            if not isinstance(actual_actions, list):
                errors.append(f"{day['name']} {step['id']}: groundtruth.actions must be a list")
                continue
            if actual_actions != expected["actions"]:
                errors.append(
                    f"{day['name']} {step['id']}: groundtruth actions mismatch "
                    f"(expected {expected['actions']}, got {actual_actions})"
                )
    return errors


def validate_observable_tasks(scenario):
    errors = []
    updates_by_day = build_updates_by_day(scenario)
    all_days = scenario.get("days", [])
    prior_visible_text_by_day = []
    accumulated_visible = []
    for day in all_days:
        prior_visible_text_by_day.append("\n".join(accumulated_visible).lower())
        accumulated_visible.extend(day.get("start_instructions", []))
        for step in day.get("steps", []):
            text = step.get("text")
            if text:
                accumulated_visible.append(text)

    for day_index, day in enumerate(all_days):
        day_name = day.get("name")
        task_states = {task["id"]: init_task_state(task) for task in day.get("tasks", [])}
        for update in updates_by_day.get(day_name, {}).get("pre", []):
            task_id = update.get("task_id")
            if task_id in task_states:
                apply_task_update(task_states[task_id], update, task_states=task_states)

        _, indices = find_today_only_lines(day.get("start_instructions", []))
        visible_lines = [day["start_instructions"][idx] for idx in indices]

        for task_id, state in task_states.items():
            task = state["task"]
            if task.get("regular"):
                continue
            if state.get("canceled"):
                continue
            current_label = state.get("current", {}).get("label") or task.get("label")
            if not current_label:
                continue
            if task.get("cross_day"):
                if any(
                    instruction_mentions_full_label(line, current_label)
                    for line in visible_lines
                ):
                    errors.append(
                        f"{day_name}: cross-day task '{task_id}' should not be repeated in the due-day start instructions"
                    )
                if current_label.rstrip(".").lower() not in prior_visible_text_by_day[day_index]:
                    errors.append(
                        f"{day_name}: cross-day task '{task_id}' was not exposed on an earlier day before it became due"
                    )
                continue
            if not any(
                instruction_mentions_full_label(line, current_label)
                for line in visible_lines
            ):
                errors.append(
                    f"{day_name}: non-regular task '{task_id}' is not exposed with its full participant-visible label before it can become due"
                )
    return errors


def validate_scenario(scenario):
    """Validate scenario structure, including state channel metadata."""
    errors = []
    warnings = []
    require_action_text = bool(scenario.get("state_channels"))
    require_structured_lures = require_action_text
    min_lures_per_day = 10 if require_structured_lures else 0
    allowed_channels = list_state_channels(scenario)
    state_visibility = scenario.get("state_visibility")
    if state_visibility is not None and not isinstance(state_visibility, dict):
        errors.append("state_visibility must be an object if provided")
    state_channels = scenario.get("state_channels")
    if state_channels is not None and not isinstance(state_channels, dict):
        errors.append("state_channels must be an object if provided")
    if isinstance(state_channels, dict):
        for channel, config in state_channels.items():
            if not isinstance(config, dict):
                errors.append(f"state_channels.{channel} must be an object")
                continue
            mode = config.get("mode", "delta")
            if mode not in ("delta", "snapshot"):
                errors.append(
                    f"state_channels.{channel}.mode must be 'delta' or 'snapshot'"
                )
            if channel in PROCESS_CHANNELS and mode != "snapshot":
                errors.append(
                    f"state_channels.{channel}.mode must be 'snapshot' for process channels"
                )
            default_item = config.get("default_item")
            if default_item is not None:
                if not isinstance(default_item, dict):
                    errors.append(
                        f"state_channels.{channel}.default_item must be an object"
                    )
                elif not default_item.get("id") or not default_item.get("text"):
                    errors.append(
                        f"state_channels.{channel}.default_item must include id and text"
                    )
                elif channel in PROCESS_CHANNELS:
                    phase = default_item.get("meta", {}).get("phase")
                    if phase != "idle":
                        errors.append(
                            f"state_channels.{channel}.default_item meta.phase must be 'idle'"
                        )
            elif channel in PROCESS_CHANNELS:
                errors.append(
                    f"state_channels.{channel} must define default_item for process channels"
                )
    for day in scenario.get("days", []):
        day_errors, day_warnings = validate_day(
            day,
            allowed_channels,
            require_action_text=require_action_text,
            require_structured_lures=require_structured_lures,
            min_lures_per_day=min_lures_per_day,
        )
        errors.extend(day_errors)
        warnings.extend(day_warnings)
    errors.extend(validate_updates(scenario))
    errors.extend(validate_start_instructions_updates(scenario))
    errors.extend(validate_new_task_headers(scenario))
    errors.extend(validate_no_canceled_task_mentions(scenario))
    errors.extend(validate_observable_tasks(scenario))
    errors.extend(validate_groundtruth_labels(scenario))
    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="PM-Bench mini timeline simulator")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run an interactive session")
    run_parser.add_argument("--scenario", required=True)
    run_parser.add_argument("--log", default=None)
    run_parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for generated artifacts (logs, sidecars).",
    )
    run_parser.add_argument(
        "--task-legend",
        action="store_true",
        dest="task_legend",
        default=False,
        help="Show a daily action-handle legend (active tasks + lures).",
    )

    run_llm_parser = subparsers.add_parser("run-llm", help="Run a model session")
    run_llm_parser.add_argument("--scenario", required=True)
    run_llm_parser.add_argument("--log", default=None)
    run_llm_parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for generated artifacts (logs, sidecars).",
    )
    run_llm_parser.add_argument("--model", default="gpt-5")
    run_llm_parser.add_argument("--env", default=".env")
    run_llm_parser.add_argument(
        "--max-time-requests",
        type=int,
        default=5,
        help="Deprecated no-op. State-query caps are disabled.",
    )
    run_llm_parser.add_argument(
        "--backend",
        choices=["openai", "openrouter", "sglang"],
        default="openai",
        help="Select which API backend to use for inference.",
    )
    run_llm_parser.add_argument(
        "--base-url",
        default=None,
        help="Override the OpenAI-compatible base URL (useful for sglang).",
    )
    run_llm_parser.add_argument(
        "--api-key",
        default=None,
        help="API key override; for sglang use any string (default: None).",
    )
    run_llm_parser.add_argument("--prompt-log", default=None)
    run_llm_parser.add_argument(
        "--task-legend",
        action="store_true",
        dest="task_legend",
        default=False,
        help="Show a daily action-handle legend (active tasks + lures).",
    )
    run_llm_parser.add_argument(
        "--enable-heartbeat",
        action="store_true",
        dest="enable_heartbeat",
        default=False,
        help=(
            "Allow optional action=set_heartbeat so the model can enable periodic "
            "proactive-check reminders (30 or 60 virtual minutes)."
        ),
    )
    run_llm_parser.add_argument(
        "--auto-heartbeat-minutes",
        type=int,
        choices=list(HEARTBEAT_INTERVAL_OPTIONS),
        default=None,
        help=(
            "Auto-enable heartbeat at the start of each day with a fixed interval "
            "(30 or 60). Does not change default behavior unless provided."
        ),
    )
    run_llm_parser.add_argument(
        "--heartbeat-message-mode",
        choices=list(HEARTBEAT_MESSAGE_MODES),
        default="channel_query",
        help=(
            "Heartbeat prompt style. channel_query keeps the original "
            "\"consider querying channels\" nudge; task_reminder sends "
            "upcoming-task reminders."
        ),
    )
    run_llm_parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Chat-completions token budget for non-OpenAI backend action generation.",
    )

    score_parser = subparsers.add_parser("score", help="Score a log file")
    score_parser.add_argument("--scenario", required=True)
    score_parser.add_argument("--log", required=True)
    score_parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for the generated markdown score report.",
    )

    validate_parser = subparsers.add_parser("validate", help="Validate a scenario file")
    validate_parser.add_argument("--scenario", required=True)

    args = parser.parse_args()
    if args.command == "run":
        scenario = load_scenario(args.scenario)
        run_interactive(scenario, args.log, args.task_legend, args.out_dir)
    elif args.command == "run-llm":
        scenario = load_scenario(args.scenario)
        run_llm(
            scenario,
            args.log,
            args.model,
            args.env,
            args.max_time_requests,
            args.backend,
            args.base_url,
            args.api_key,
            args.prompt_log,
            args.task_legend,
            args.out_dir,
            args.enable_heartbeat,
            args.auto_heartbeat_minutes,
            args.heartbeat_message_mode,
            args.max_tokens,
        )
    elif args.command == "score":
        scenario = load_scenario(args.scenario)
        log_entries, run_metadata = read_log_with_metadata(args.log)
        summary, per_day, summary_steps = score_log(scenario, log_entries)
        buffer = StringIO()
        with redirect_stdout(buffer):
            print_report(summary, per_day, summary_steps, run_metadata=run_metadata)
        report_text = buffer.getvalue()
        print(report_text, end="")
        report_md = build_markdown_report(
            summary, per_day, summary_steps, run_metadata=run_metadata
        )
        log_path = os.path.abspath(args.log)
        report_filename = f"{os.path.splitext(os.path.basename(log_path))[0]}.score.md"
        report_dir = args.out_dir or os.path.dirname(log_path)
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, report_filename)
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(report_md)
    elif args.command == "validate":
        scenario = load_scenario(args.scenario)
        errors, warnings = validate_scenario(scenario)
        for warning in warnings:
            print(f"warning: {warning}")
        for error in errors:
            print(f"error: {error}")
        if errors:
            return 1
        print("Scenario OK")
    else:
        parser.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
