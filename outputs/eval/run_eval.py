"""Multi-agent vs single-agent baseline on the frozen answer key.

Usage (from project root):
    python -m outputs.eval.run_eval --runs 3
"""
import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import llm
from outputs.agents import agents
from outputs.graph.build_graph import analyze_scenario
from outputs.retrieval import Retriever, load_persona

ROOT = Path(__file__).resolve().parents[2]
KEY = ROOT / "outputs" / "data" / "answer_key.json"
RESULTS = ROOT / "results"


def baseline(scenario, profile, chunks):
    r = Retriever(chunks)
    ctx = r.search(agents.flagger_queries(scenario) + agents.checker_queries(scenario))
    return agents.single_agent_baseline(scenario, profile, ctx).verdict


def retrieval_recall(case, chunks):
    r = Retriever(chunks)
    s = case["scenario"]
    got = {c["ref"] for c in r.search(agents.flagger_queries(s) + agents.checker_queries(s))}
    return sum(k in got for k in case["key_evidence"]) / len(case["key_evidence"])


def score(preds, cases):
    by_type = defaultdict(lambda: [0, 0])
    fp = fn = correct = 0
    for c in cases:
        p, e = preds[c["case_id"]], c["expected_verdict"]
        ok = p == e
        correct += ok
        by_type[c["case_type"]][0] += ok
        by_type[c["case_type"]][1] += 1
        fp += (p == "confirmed" and e == "dismissed")
        fn += (p != "confirmed" and e == "confirmed")
    errors = sum(1 for x in preds.values() if x == "error")
    return {"accuracy": correct / len(cases), "false_positives": fp, "missed_gaps": fn, "errors": errors,
            "by_type": {t: f"{a}/{n}" for t, (a, n) in by_type.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--only", default="", help="comma-separated case ids")
    args = ap.parse_args()

    cases = json.loads(KEY.read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["case_id"] in args.only.split(",")]
    personas = {p: load_persona(p) for p in {c["persona"] for c in cases}}
    recall = {c["case_id"]: retrieval_recall(c, personas[c["persona"]]["chunks"]) for c in cases}

    runs, traces = [], []
    for run in range(args.runs):
        multi, base = {}, {}
        for c in cases:
            p = personas[c["persona"]]
            t0 = time.time()
            try:
                tr = analyze_scenario(c["scenario"], p["profile"], p["chunks"])
            except RuntimeError as e:      # all providers failed: count as wrong, keep going
                tr = {"verdict": "error", "error": str(e)[:300]}
            multi[c["case_id"]] = tr["verdict"]
            try:
                base[c["case_id"]] = baseline(c["scenario"], p["profile"], p["chunks"])
            except RuntimeError:
                base[c["case_id"]] = "error"
            traces.append({"run": run, "case_id": c["case_id"], "expected": c["expected_verdict"],
                           "baseline": base[c["case_id"]], **tr})
            print(f"run {run} {c['case_id']:3} expected={c['expected_verdict']:10} "
                  f"multi={multi[c['case_id']]:12} baseline={base[c['case_id']]:12} "
                  f"({time.time() - t0:.1f}s)", flush=True)
        runs.append({"multi": score(multi, cases), "baseline": score(base, cases),
                     "multi_preds": multi, "baseline_preds": base})

    calls = defaultdict(lambda: {"calls": 0, "tokens": 0})
    providers = defaultdict(int)
    for e in llm.CALL_LOG:
        calls[e["agent"]]["calls"] += 1
        calls[e["agent"]]["tokens"] += e["tokens"]
        providers[e["provider"]] += 1

    def rng(system, key):
        vals = [r[system][key] for r in runs]
        return {"mean": round(statistics.mean(vals), 3), "min": min(vals), "max": max(vals)}

    summary = {
        "cases": len(cases), "runs": args.runs,
        "multi_agent": {"accuracy": rng("multi", "accuracy"), "false_positives": rng("multi", "false_positives"),
                        "missed_gaps": rng("multi", "missed_gaps"),
                        "by_type_per_run": [r["multi"]["by_type"] for r in runs]},
        "single_agent_baseline": {"accuracy": rng("baseline", "accuracy"),
                                  "false_positives": rng("baseline", "false_positives"),
                                  "missed_gaps": rng("baseline", "missed_gaps"),
                                  "by_type_per_run": [r["baseline"]["by_type"] for r in runs]},
        "retrieval_recall_of_key_evidence": round(statistics.mean(recall.values()), 3),
        "calls_by_agent": calls, "providers_used": providers,
        "per_run": runs,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    (RESULTS / "eval_traces.json").write_text(json.dumps(traces, indent=2))
    print("\n" + json.dumps({k: summary[k] for k in
                             ["multi_agent", "single_agent_baseline", "retrieval_recall_of_key_evidence",
                              "calls_by_agent", "providers_used"]}, indent=2))


if __name__ == "__main__":
    main()
