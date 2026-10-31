#!/usr/bin/env python3
"""Score the JD career classifier against a fixed, labelled set of job descriptions.

Run this before changing FIREWORKS_MODEL (or after Fireworks retires one), so a
cheaper or newer model can't quietly start sending people to the wrong track.

Usage:
    set -a && . ./.env && set +a
    python3 scripts/eval_career_classifier.py                        # current FIREWORKS_MODEL
    python3 scripts/eval_career_classifier.py --model accounts/fireworks/models/glm-5p3-flash
    python3 scripts/eval_career_classifier.py --min-accuracy 0.9

Exit 0 when accuracy >= --min-accuracy, 1 below it, 2 on missing prerequisites.
With LANGSMITH_API_KEY set, every call is traced under run name
"eval:career_classifier" with the case id in metadata, so failures can be
inspected in LangSmith next to production traces.

Uses the exact classifier prompt from app.py (read from source, not copied), so
the eval can't drift from what production sends. Imports nothing from app.py,
so it needs no MONGO_URI.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = Path(__file__).with_name("eval_career_cases.json")


def load_prompt_and_careers() -> tuple[str, set[str]]:
    src = (ROOT / "app.py").read_text()
    found: dict[str, object] = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("_CAREER_CLASSIFIER_PROMPT", "_VALID_CAREERS"):
                    found[t.id] = ast.literal_eval(node.value)
    if len(found) != 2:
        raise SystemExit("could not find _CAREER_CLASSIFIER_PROMPT / _VALID_CAREERS in app.py")
    return str(found["_CAREER_CLASSIFIER_PROMPT"]), set(found["_VALID_CAREERS"])  # type: ignore[arg-type]


def parse_choice(content: str) -> str:
    """Same parsing rule as app._classify_career_from_jd."""
    words = (content or "").strip().lower().split()
    return words[0].strip(".,:;()\"'") if words else ""


def run(llm, cases: list[dict], prompt: str, valid: set[str], verbose: bool = True) -> dict:
    results = []
    for c in cases:
        snippet = (c.get("title", "") + "\n\n" + c["jd"])[:3000].strip()
        t0 = time.time()
        try:
            resp = llm.invoke(
                [("system", prompt), ("human", snippet)],
                config={"run_name": "eval:career_classifier", "metadata": {"case_id": c["id"]}},
            )
            got = parse_choice(getattr(resp, "content", "") or "")
            err = None
        except Exception as ex:  # a provider error is a failed case, not a crash
            got, err = "", f"{type(ex).__name__}: {str(ex)[:120]}"
        ok = got == c["expected"]
        results.append({"id": c["id"], "expected": c["expected"], "got": got or "(none)",
                        "ok": ok, "valid": got in valid, "ms": int((time.time() - t0) * 1000), "error": err})
        if verbose:
            mark = "PASS" if ok else "FAIL"
            extra = f"  [{err}]" if err else ("" if got in valid else "  [not a valid career]")
            print(f"{mark}  {c['id']:<28} expected={c['expected']:<14} got={got or '(none)'}{extra}")
    n = len(results)
    correct = sum(r["ok"] for r in results)
    return {
        "n": n, "correct": correct, "accuracy": correct / n if n else 0.0,
        "errors": sum(1 for r in results if r["error"]),
        "invalid": sum(1 for r in results if not r["valid"] and not r["error"]),
        "p50_ms": sorted(r["ms"] for r in results)[n // 2] if n else 0,
        "results": results,
    }


def main(argv: list[str] | None = None, llm=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=os.environ.get("FIREWORKS_MODEL", "accounts/fireworks/models/deepseek-v4p1-flash"))
    ap.add_argument("--min-accuracy", type=float, default=0.85)
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = ap.parse_args(argv)

    prompt, valid = load_prompt_and_careers()
    cases = json.loads(CASES_PATH.read_text())
    unknown = {c["expected"] for c in cases} - valid
    if unknown:
        print(f"PREREQ: cases expect unknown careers {sorted(unknown)}")
        return 2

    if llm is None:
        key = os.environ.get("FIREWORKS_API_KEY")
        if not key:
            print("PREREQ: FIREWORKS_API_KEY not set. Run: set -a && . ./.env && set +a")
            return 2
        from langchain_fireworks import ChatFireworks
        llm = ChatFireworks(model=args.model, api_key=key, temperature=0, timeout=60)

    print(f"model={args.model} cases={len(cases)}")
    summary = run(llm, cases, prompt, valid, verbose=not args.json)
    if args.json:
        print(json.dumps({k: v for k, v in summary.items() if k != "results"}))
    else:
        print(f"\naccuracy {summary['correct']}/{summary['n']} = {summary['accuracy']:.0%}  "
              f"provider_errors={summary['errors']} invalid_replies={summary['invalid']}  p50={summary['p50_ms']}ms")
    return 0 if summary["accuracy"] >= args.min_accuracy else 1


if __name__ == "__main__":
    sys.exit(main())
