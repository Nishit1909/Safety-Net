import json
import os

os.environ["SAFETYNET_MOCK"] = "1"

import llm
from outputs import metrics
from outputs.agents.common import validate_citations
from outputs.retrieval import Retriever, load_persona
from outputs.agents import agents
from schema import Claim, Citation


def test_metrics_persona_a():
    p = load_persona("persona_a")["profile"]
    assert metrics.monthly_outgo(p) == 70000
    assert metrics.runway_months(p) == 4.3
    assert metrics.runway_months(p, include_locked=True) == 7.1   # FD net of 1% penalty, PPF excluded


def test_locked_with_lockin_excluded():
    p = load_persona("persona_b")["profile"]
    assert metrics.runway_months(p, True) == metrics.runway_months(p) == 4.0


def test_chunking_refs():
    refs = {c["ref"] for c in load_persona("persona_c")["chunks"]}
    assert "loan_agreement.md#7.1" in refs and "term_policy.md#6.1" in refs


def test_citation_guard_drops_fabricated():
    chunks = load_persona("persona_a")["chunks"]
    ctx = [c for c in chunks if c["ref"] == "term_policy.md#5.5"]
    claim = Claim(position="gap", argument="x", citations=[
        Citation(ref="term_policy.md#5.5", quote="does not cover loss of income due to illness"),
        Citation(ref="term_policy.md#9.9", quote="invented clause"),
        Citation(ref="term_policy.md#5.5", quote="pays full salary forever"),
    ])
    claim, dropped = validate_citations(claim, ctx)
    assert [c.ref for c in claim.citations] == ["term_policy.md#5.5"]
    assert len(dropped) == 2


def test_retrieval_recall_on_answer_key():
    cases = json.load(open("outputs/data/answer_key.json", encoding="utf-8"))
    hits = total = 0
    for c in cases:
        r = Retriever(load_persona(c["persona"])["chunks"])
        got = {x["ref"] for x in r.search(agents.flagger_queries(c["scenario"]) + agents.checker_queries(c["scenario"]))}
        hits += sum(k in got for k in c["key_evidence"]); total += len(c["key_evidence"])
    assert hits / total >= 0.8, hits / total


def test_full_graph_with_mock():
    def mock(prompt, schema, agent):
        if agent == "scenario_generator":
            return {"scenarios": [{"scenario_id": "S1", "description": "job loss 6 months"},
                                  {"scenario_id": "S2", "description": "death of earning member"}]}
        if agent in ("flagger", "checker"):
            return {"position": "gap", "argument": "mock", "citations": []}
        if agent == "resolver":
            return {"verdict": "confirmed", "justification": "mock", "decisive_refs": []}
        return {"summary": "mock", "findings": ["f"]}
    llm.set_mock_handler(mock)
    from outputs.graph.build_graph import run
    out = run("persona_a")
    assert set(out["traces"]) == {"S1", "S2"}
    assert out["metrics"]["severity"] == "medium"
    assert out["pending_approval"] and "fix_costs" not in out["report"]
    out2 = run("persona_a", approved=True)
    assert out2["report"]["fix_costs"] and not out2["pending_approval"]
