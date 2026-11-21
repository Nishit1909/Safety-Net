"""Run the full agent on one persona.

    python main.py persona_a                                   # AI proposes 3 scenarios
    python main.py persona_a --scenario "Job loss for 8 months" # your own scenario
    python main.py persona_a --scenario "Job loss" --scenario "Death of earning member"
    python main.py persona_a --approve                         # release held fix-cost estimates
"""
import argparse
import json

from outputs.graph.build_graph import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("persona", nargs="?", default="persona_a")
    ap.add_argument("--scenario", action="append", default=[],
                    help="custom scenario text; repeat the flag for several")
    ap.add_argument("--approve", action="store_true")
    args = ap.parse_args()

    custom = [{"scenario_id": f"U{i + 1}", "description": s} for i, s in enumerate(args.scenario)]
    out = run(args.persona, approved=args.approve, scenarios=custom or None)

    for sid, t in out["traces"].items():
        print(f"\n[{sid}] {t['scenario']}\n  Flagger: {t['flagger']['position']} | Checker: {t['checker']['position']}"
              f"\n  Resolver: {t['verdict'].upper()} - {t['ruling']['justification']}")
    print("\nMETRICS:", json.dumps(out["metrics"], indent=2))
    print("\nREPORT:", json.dumps(out["report"], indent=2))
    print("\nPENDING APPROVAL:", out.get("pending_approval"))
