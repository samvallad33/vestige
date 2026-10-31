#!/usr/bin/env python3
"""
Standalone Bare-Model Benchmark Test for Math-16 Turn 5.
Runs the exact question against an OpenAI frontier model (zero-vestige, zero-RAG, bare model).
Compares the output directly against Ground Truth and Vestige + Qwen 35B.
"""

import os
import sys
import json
import argparse
import urllib.request
import urllib.error

GROUND_TRUTH = (
    "For the first case, the lower bound is ||w*|| = \\Omega(1 / \\sqrt{\\opt}). "
    "For the second case, the lower bound is ||w*|| = \\Omega(1 / \\sqrt{\\opt \\cdot \\ln(1/\\opt)})."
)

TURN_5_QUESTION = """Under Assumption 3.2, give an asymptotic \\Omega lower bound for ||w*|| under the following two cases:
(1) if ||x|| <= B almost surely and \\opt < \\frac{R^4}{200 U^3 B},
(2) if P_x is (\\alpha_1, \\alpha_2)-sub-exponential and \\opt \\cdot \\ln(1/\\opt) < \\frac{R^4}{200 (1 + 2\\alpha_1) \\alpha_2 U^3}"""

HISTORY_TURNS = [
    {
        "turn": 0,
        "question": "Let $r,\\rho>0$ be given, then upper bound $\\frac{2}{\\rho}(1-e^{-r\\rho})$ with an integral and then a function of $\\rho$.",
        "answer": "A tight enough upper bound is $\\frac{2}{\\rho}(1-e^{-r\\rho})\\le\\int_0^{2\\pi}\\llog\\del{r\\rho\\envert{\\cos(\\theta)}}r\\dif\\theta\\le \\frac{8\\sqrt{2}}{\\rho}$"
    },
    {
        "turn": 1,
        "question": "Given $w,w'\\in\\R^d$, suppose $\\pr_{(x,y)\\sim P}\\del{y\\ne\\sign\\del{\\langle w,x\\rangle}}=\\opt$. If $\|x\|\\le B$ almost surely, then upper bound \\bbE\\sbr{\\llog\\del{\\envert{\\langle w',x\\rangle}} \\mathbb{I}_{y \\ne \\sign(\\langle w,x \\rangle)}}.",
        "answer": "A tight enough upper bound is $B\|w'\|\\cdot\\opt$"
    },
    {
        "turn": 2,
        "question": "Under Assumption 3.2, for any $w\\in\\R^d$, upper bound $\\bbE\\sbr{\\llog\\del{\\envert{\\langle w,x\\rangle}}}$.",
        "answer": "A tight enough upper bound is $\\frac{12U}{\\|w\\|}$"
    },
    {
        "turn": 3,
        "question": "Given $\\rho>0$, under Assumption 3.2, upper bound \\cRlog(\\rho\\baru) and \\inf_{\\rho>0}\\cRlog(\\rho\\baru) under the following two cases: if $\|x\|\\le B$ almost surely; or if $P_x$ is $(\\alpha_1,\\alpha_2)$-sub-exponential.",
        "answer": "For the first case, we have $\\cRlog(\\rho\\baru)\\le \\frac{12U}{\\rho}+\\rho B\\cdot\\opt$, with $\\inf_{\\rho>0}\\cRlog(\\rho\\baru)\\le\\sqrt{50UB\\cdot\\opt}$. For the second case, $\\inf_{\\rho>0}\\cRlog(\\rho\\baru)\\le\\sqrt{50U(1+2\\alpha_1)\\alpha_2\\cdot\\opt\\ln(1/\\opt)}$."
    },
    {
        "turn": 4,
        "question": "Under Assumption 3.2, given $w\\in\\R^d$, lower bound $\\cRlog(w)$ under the following two cases: (1) $R\|w\|\\le2$, or (2) $R\|w\|\\ge2$.",
        "answer": "If $R\|w\|\\le2$, then $\\cRlog(w)\\ge \\frac{R^2}{2U}$. If $R\|w\|\\ge2$, then $\\cRlog(w)\\ge \\frac{R}{U\|w\|}$."
    }
]

def build_messages(mode: str):
    messages = []
    if mode == "isolated":
        # Pure isolated test: Turn 5 completely bare with zero prior memory/context
        messages.append({
            "role": "user",
            "content": TURN_5_QUESTION
        })
    elif mode == "long_context":
        # Standard flat long-context benchmark arm: full prior Q&A turns in context
        for t in HISTORY_TURNS:
            messages.append({"role": "user", "content": t["question"]})
            messages.append({"role": "assistant", "content": t["answer"]})
        messages.append({"role": "user", "content": TURN_5_QUESTION})
    return messages

def run_test(api_key: str, model: str, mode: str, base_url: str = "https://api.openai.com/v1"):
    messages = build_messages(mode)
    print(f"\n======================================================================")
    print(f"RUNNING BARE-MODEL BENCHMARK TEST (Math-16 Turn 5)")
    print(f"Model: {model}")
    print(f"Mode:  {mode} ({'Pure Zero-Memory Isolated Question' if mode == 'isolated' else 'Flat Chat History Turns 0-4'})")
    print(f"Endpoint: {base_url}/chat/completions")
    print(f"======================================================================\n")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0 if not model.startswith("o1") else 1 # o1 requires default/1
    }

    # For reasoning models, adjust parameters if needed
    if model.startswith("o1") or "astra" in model.lower():
        payload.pop("temperature", None)
        payload["max_completion_tokens"] = 65536
    else:
        payload["max_tokens"] = 16384

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    )

    print("Sending request to API... (streaming/waiting for response)")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choice = data["choices"][0]
            answer = choice["message"]["content"]
            usage = data.get("usage", {})

            print("\n------------------------- MODEL RESPONSE -------------------------")
            print(answer)
            print("------------------------------------------------------------------")
            print(f"Usage: {json.dumps(usage, indent=2)}")

            # Save result
            out_file = f"bare_model_{model.replace('/', '_')}_{mode}.json"
            with open(out_file, "w") as f:
                json.dump({
                    "model": model,
                    "mode": mode,
                    "question": TURN_5_QUESTION,
                    "answer": answer,
                    "usage": usage,
                    "ground_truth": GROUND_TRUTH
                }, f, indent=2)

            print(f"\nSaved raw result to: {out_file}")
            print("\n======================= COMPARISON AUDIT =======================")
            print(f"GROUND TRUTH: {GROUND_TRUTH}")
            print("================================================================")
            return answer

    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")
        print(f"API Error {e.code}: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Bare Model Math-16 Turn 5 Benchmark")
    parser.add_argument("--model", default="gpt-6-astra", help="Model name (default: gpt-6-astra)")
    parser.add_argument("--mode", choices=["isolated", "long_context"], default="long_context",
                        help="'isolated' = Turn 5 only (zero memory); 'long_context' = Flat Turns 0-4 context")
    parser.add_argument("--key", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key (or set OPENAI_API_KEY env)")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="API Base URL")

    args = parser.parse_args()

    api_key = args.key
    if not api_key:
        api_key = input("Enter your OpenAI API key: ").strip()
        if not api_key:
            print("Error: API key is required.", file=sys.stderr)
            sys.exit(1)

    run_test(api_key, args.model, args.mode, args.base_url)

if __name__ == "__main__":
    main()
