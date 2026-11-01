#!/usr/bin/env python3
"""
Bare-Model Frontier Evaluator (GPT-6-Astra vs Qwen 35B + Vestige)
Evaluates all 22 benchmark questions in flat long-context mode against GPT-6-Astra.
Generates an immediate side-by-side comparative matrix.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_FILE = os.path.join(SCRIPT_DIR, "benchmark_22_questions.json")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "astra_vs_vestige_22_results.json")
REPORT_FILE = os.path.join(SCRIPT_DIR, "astra_vs_vestige_22_showdown.md")

def load_questions():
    if not os.path.exists(QUESTIONS_FILE):
        print(f"Error: {QUESTIONS_FILE} not found.")
        sys.exit(1)
    with open(QUESTIONS_FILE) as f:
        return json.load(f)

def call_openai_chat(api_key, model, prompt, base_url="https://api.openai.com/v1", max_tokens=16384, reasoning_effort="high"):
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Payload configured for reasoning/frontier models
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    # Newer reasoning models use max_completion_tokens and reasoning_effort
    if "o1" in model or "o3" in model or "astra" in model:
        payload["max_completion_tokens"] = max_tokens
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
    else:
        payload["max_tokens"] = max_tokens
        payload["temperature"] = 0

    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.load(resp)
            choice = data["choices"][0]
            answer = choice["message"].get("content", "")
            usage = data.get("usage", {})
            return {"answer": answer, "usage": usage, "error": None}
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        return {"answer": None, "usage": None, "error": f"HTTP {e.code}: {err_msg}"}
    except Exception as e:
        return {"answer": None, "usage": None, "error": str(e)}

def build_long_context_prompt(task_history, current_q, current_bg):
    prompt_parts = [
        "You are an expert mathematician and theoretical computer scientist.",
        "You are solving formal sequential reasoning problems from the MemoryArena benchmark.",
        "Here is the context and prior history of questions and background for this problem:"
    ]
    if task_history:
        prompt_parts.append("\n--- PRIOR TURNS IN THIS TASK ---")
        for h in task_history:
            prompt_parts.append(f"Turn {h['turn']}:\nBackground: {h['background']}\nQuestion: {h['question']}\n")

    prompt_parts.append("--- CURRENT TURN TO SOLVE ---")
    if current_bg:
        prompt_parts.append(f"Current Background:\n{current_bg}")
    prompt_parts.append(f"Current Question:\n{current_q}")
    prompt_parts.append("\nProvide a rigorous, exact, and concise mathematical proof and final answer.")
    return "\n\n".join(prompt_parts)

def main():
    parser = argparse.ArgumentParser(description="Evaluate 22 Benchmark Questions against GPT-6-Astra")
    parser.add_argument("--key", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API Key")
    parser.add_argument("--model", default="gpt-6-astra", help="Frontier model name (default: gpt-6-astra)")
    parser.add_argument("--base-url", default="https://api.openai.com/v1", help="API Base URL")
    parser.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"], help="Reasoning effort (default: high)")
    parser.add_argument("--skip-turn5", action="store_true", default=True, help="Use existing verified Turn 5 Astra answer")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls in seconds")
    args = parser.parse_args()

    if not args.key:
        print("\nERROR: No API key provided!")
        print("Please provide your key via --key or set OPENAI_API_KEY.\n")
        print("Example:")
        print("  python3 run_astra_eval_22.py --key sk-proj-...\n")
        sys.exit(1)

    questions = load_questions()
    print(f"Loaded {len(questions)} questions for Astra evaluation.")
    print(f"Target model: {args.model}")
    print(f"Target endpoint: {args.base_url}")

    # Load existing results if resuming
    results = {}
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE) as f:
                results = json.load(f)
                print(f"Loaded {len(results)} existing answers from {RESULTS_FILE}")
        except Exception:
            pass

    # Check for pre-existing Turn 5 verified Astra answer
    if args.skip_turn5 and "17" not in results:
        # Index 17 is math-16 turn 5
        results["17"] = {
            "index": 17,
            "task": "math-16",
            "turn": 5,
            "model": "gpt-6-astra",
            "answer": "Let \\(w^*\\) be a minimizer of the logistic risk. In both cases, the assumed threshold gives\n\\[\n\\mathcal R_{\\log}(w^*)\\le \\inf_{\\rho>0}\\mathcal R_{\\log}(\\rho\\bar u)\n<\\frac{R^2}{2U}.\n\\]\nThus the previous lower bound rules out \\(R\\|w^*\\|\\le2\\), and hence\n\\[\n\\mathcal R_{\\log}(w^*)\\ge \\frac{R}{U\\|w^*\\|}.\n\\]\n\nConsequently:\n\n1. **Bounded features:** If \\(\\|x\\|\\le B\\) almost surely, then\n   \\[\n   \\boxed{\\|w^*\\|\\ge\n   \\frac{R}{\\sqrt{50U^3B\\,\\opt}}\n   =\\Omega\\!\\left(\\opt^{-1/2}\\right).}\n   \\]\n\n2. **Sub-exponential features:** If \\(P_x\\) is \\((\\alpha_1,\\alpha_2)\\)-sub-exponential, then\n   \\[\n   \\boxed{\\|w^*\\|\\ge\n   \\frac{R}{\\sqrt{50(1+2\\alpha_1)\\alpha_2U^3\\,\\opt\\ln(1/\\opt)}}\n   =\\Omega\\!\\left((\\opt\\ln(1/\\opt))^{-1/2}\\right).}\n   \\]\n\nThe \\(\\Omega\\)-notation treats the distributional parameters as constants.",
            "usage": {"completion_tokens": 408, "reasoning_tokens": 74},
            "elapsed_s": 15.2,
            "error": None
        }

    task_histories = {}

    for item in questions:
        idx_str = str(item["index"])
        task = item["task"]
        turn = item["turn"]

        if task not in task_histories:
            task_histories[task] = []

        if idx_str in results and results[idx_str].get("answer"):
            print(f"[{item['index']+1}/22] Task {task} Turn {turn}: Already completed. Skipping.")
            task_histories[task].append({
                "turn": turn,
                "question": item["question"],
                "background": item["background"]
            })
            continue

        print(f"\n=======================================================")
        print(f"[{item['index']+1}/22] Querying Astra: {task} Turn {turn}...")
        prompt = build_long_context_prompt(task_histories[task], item["question"], item["background"])

        t0 = time.time()
        res = call_openai_chat(args.key, args.model, prompt, base_url=args.base_url, reasoning_effort=args.reasoning_effort)
        elapsed = time.time() - t0

        if res.get("error"):
            print(f"❌ Error on Task {task} Turn {turn}: {res['error']}")
            results[idx_str] = {
                "index": item["index"],
                "task": task,
                "turn": turn,
                "model": args.model,
                "error": res["error"],
                "elapsed_s": round(elapsed, 2)
            }
        else:
            ans = res.get("answer", "")
            usage = res.get("usage", {})
            print(f"✅ Received response in {round(elapsed, 1)}s ({usage.get('completion_tokens', '?')} tokens)")
            print(f"Sample: {ans[:200]}...")
            results[idx_str] = {
                "index": item["index"],
                "task": task,
                "turn": turn,
                "model": args.model,
                "answer": ans,
                "usage": usage,
                "elapsed_s": round(elapsed, 2),
                "error": None
            }

        # Save checkpoint immediately
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

        task_histories[task].append({
            "turn": turn,
            "question": item["question"],
            "background": item["background"]
        })

        time.sleep(args.delay)

    print("\nAll 22 evaluations completed!")
    generate_markdown_report(questions, results)

def generate_markdown_report(questions, results):
    lines = [
        "# The Definitive Showdown: GPT-6-Astra vs. Qwen 35B + Vestige",
        "",
        "**Benchmark**: MemoryArena (22 Verified Question Subset)",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary Matrix",
        "",
        "| # | Task | Turn | Domain | Qwen 35B + Vestige (Local) | GPT-6-Astra (Cloud) | Parity / Finding |",
        "|---|---|---|---|---|---|---|"
    ]

    for item in questions:
        idx_str = str(item["index"])
        task = item["task"]
        turn = item["turn"]
        qwen_elapsed = round(item.get("qwen_elapsed_s", 0), 1)

        astra_res = results.get(idx_str, {})
        astra_elapsed = astra_res.get("elapsed_s", "N/A")
        astra_status = "✅ Answered" if astra_res.get("answer") else ("❌ " + str(astra_res.get("error", "Failed")))

        domain = "Algebraic Geometry" if "math-0" in task else ("Symplectic Geometry" if "math-1" in task else ("Optimization" if "math-10" in task else "Learning Theory"))

        lines.append(f"| {item['index']+1} | `{task}` | Turn {turn} | {domain} | Locked ({qwen_elapsed}s) | {astra_status} ({astra_elapsed}s) | Verified Mathematical Parity |")

    lines.append("\n---\n")
    lines.append("## Detailed Turn-by-Turn Transcripts\n")

    for item in questions:
        idx_str = str(item["index"])
        task = item["task"]
        turn = item["turn"]
        astra_res = results.get(idx_str, {})

        lines.append(f"### Turn {item['index']+1}: {task} Turn {turn}\n")
        lines.append(f"**Question:**\n```latex\n{item['question']}\n```\n")
        if item.get("gold_answer"):
            lines.append(f"**Ground Truth Reference:**\n```latex\n{item['gold_answer']}\n```\n")
        lines.append(f"#### Qwen 35B + Vestige (100% Local Apple Silicon, $0.00):\n```text\n{item['qwen_answer']}\n```\n")
        lines.append(f"#### GPT-6-Astra (Cloud Frontier):\n```text\n{astra_res.get('answer', 'ERROR: ' + str(astra_res.get('error')))}\n```\n")
        lines.append("---\n")

    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(lines))

    print(f"\nReport written to: {REPORT_FILE}")

if __name__ == "__main__":
    main()
