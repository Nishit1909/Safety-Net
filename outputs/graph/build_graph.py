"""LangGraph wiring: Intake -> Scenario Generator -> fan-out per scenario
[Flagger || Checker -> citation check -> Resolver] -> fan-in -> Metrics -> Writer -> Approval Gate."""
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from outputs import metrics
from outputs.agents import agents
from outputs.agents.common import validate_citations
from outputs.retrieval import Retriever, load_persona


def merge(a: dict, b: dict) -> dict:
    return {**a, **b}


class SafetyNetState(TypedDict, total=False):
    persona_id: str
    profile: dict
    chunks: list
    scenarios: list
    traces: Annotated[dict, merge]   # scenario_id -> full courtroom trace
    metrics: dict
    report: dict
    approved: bool
    pending_approval: list


def analyze_scenario(scenario: str, profile: dict, chunks: list[dict]) -> dict:
    """One courtroom lane. Flagger and Checker run concurrently and never see each other."""
    retriever = Retriever(chunks)
    flag_ctx = retriever.search(agents.flagger_queries(scenario))
    check_ctx = retriever.search(agents.checker_queries(scenario))

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_fut = pool.submit(agents.risk_flagger, scenario, profile, flag_ctx)
        c_fut = pool.submit(agents.coverage_checker, scenario, profile, check_ctx)
        flag, check = f_fut.result(), c_fut.result()

    flag, f_drop = validate_citations(flag, flag_ctx)
    check, c_drop = validate_citations(check, check_ctx)

    seen, judge_ctx = set(), []
    for c in flag_ctx + check_ctx:
        if c["ref"] not in seen:
            seen.add(c["ref"])
            judge_ctx.append(c)

    ruling = agents.resolver(scenario, profile, flag, check, judge_ctx,
                             {"flagger": f_drop, "checker": c_drop})
    return {"scenario": scenario, "flagger": flag.model_dump(), "checker": check.model_dump(),
            "dropped_citations": {"flagger": f_drop, "checker": c_drop},
            "verdict": ruling.verdict, "ruling": ruling.model_dump()}


def intake(state):
    p = load_persona(state["persona_id"])
    return {"profile": p["profile"], "chunks": p["chunks"]}


def generate_scenarios(state):
    if state.get("scenarios"):          # eval mode: scenarios supplied
        return {}
    out = agents.scenario_generator(state["profile"])
    return {"scenarios": [s.model_dump() for s in out.scenarios]}


def fan_out(state):
    return [Send("scenario_lane", {"scenario": s, "profile": state["profile"], "chunks": state["chunks"]})
            for s in state["scenarios"]]


def scenario_lane(payload):
    s = payload["scenario"]
    trace = analyze_scenario(s["description"], payload["profile"], payload["chunks"])
    return {"traces": {s["scenario_id"]: trace}}


def compute_metrics(state):
    return {"metrics": metrics.compute(state["profile"], state["traces"])}


def write_report(state):
    rulings = {sid: t["ruling"] for sid, t in state["traces"].items()}
    scen = {s["scenario_id"]: s["description"] for s in state["scenarios"]}
    rep = agents.writer(state["profile"], rulings, scen, state["metrics"])
    return {"report": rep.model_dump()}


def approval_gate(state):
    """Fix-cost estimates are advisory-adjacent: held until explicitly approved."""
    held = []
    m = state["metrics"]
    if m["confirmed_gaps"]:
        shortfall_months = max(0, m["benchmark_months"] - m["runway_months"])
        held.append({"item": "emergency-fund shortfall to reach benchmark",
                     "amount_rs": round(shortfall_months * m["monthly_outgo"])})
    if state.get("approved"):
        return {"pending_approval": [], "report": {**state["report"], "fix_costs": held}}
    return {"pending_approval": held}


def build_graph():
    g = StateGraph(SafetyNetState)
    g.add_node("intake", intake)
    g.add_node("scenario_generator", generate_scenarios)
    g.add_node("scenario_lane", scenario_lane)
    g.add_node("metrics", compute_metrics)
    g.add_node("writer", write_report)
    g.add_node("approval_gate", approval_gate)
    g.add_edge(START, "intake")
    g.add_edge("intake", "scenario_generator")
    g.add_conditional_edges("scenario_generator", fan_out, ["scenario_lane"])
    g.add_edge("scenario_lane", "metrics")
    g.add_edge("metrics", "writer")
    g.add_edge("writer", "approval_gate")
    g.add_edge("approval_gate", END)
    return g.compile()


def run(persona_id: str, approved: bool = False, scenarios: list | None = None) -> dict:
    init = {"persona_id": persona_id, "approved": approved, "traces": {}}
    if scenarios:
        init["scenarios"] = scenarios
    return build_graph().invoke(init)
