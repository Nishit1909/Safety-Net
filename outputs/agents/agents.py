"""The five LLM roles. Each is one prompt + one call_llm."""
from llm import call_llm
from schema import Claim, Report, Ruling, ScenarioList
from outputs.agents.common import format_chunks, profile_facts

DISCLAIMER = ("Documents are synthetic. You are diagnosing coverage, never recommending products.")


def scenario_generator(profile: dict) -> ScenarioList:
    prompt = f"""You propose life-shock scenarios for a financial coverage review. {DISCLAIMER}
Profile:
{profile_facts(profile)}
Documents held: {profile.get('documents')}

Propose 3 scenarios that are realistic and SPECIFIC to this person's obligations and documents
(e.g. job loss, accident causing inability to work, hospitalisation, death of earning member).
Ids S1, S2, S3."""
    return call_llm(prompt, ScenarioList, model="small", temperature=0.4, agent="scenario_generator")


# Flagger and Checker get DIFFERENT retrieval queries -> independent evidence.
def flagger_queries(scenario: str) -> list[str]:
    return [scenario, f"exclusion not covered {scenario}", "waiting period exclusion limit cap",
            "loan default requirement insurance condition", "outstanding principal EMI"]


def checker_queries(scenario: str) -> list[str]:
    return [scenario, f"benefit payable rider covers {scenario}", "rider monthly benefit income",
            "sum assured sum insured coverage", "lender insurance assigned cover"]


def risk_flagger(scenario: str, profile: dict, chunks: list[dict]) -> Claim:
    prompt = f"""ROLE: Risk Flagger. Build the STRONGEST evidence-based case that this person has a
financial coverage GAP under the scenario. {DISCLAIMER}
Look hard for exclusions, waiting periods, caps, conditions, and obligations that coverage fails to meet.
Combine facts across documents when needed. Use the pre-computed numbers; do not do new arithmetic.
If, after honest effort, the evidence shows no gap, set position "no_gap".

SCENARIO: {scenario}
PROFILE:
{profile_facts(profile)}
POLICY EXCERPTS (cite using the <ref> exactly; quote must be copied verbatim from that excerpt):
{format_chunks(chunks)}"""
    return call_llm(prompt, Claim, model="small", agent="flagger")


def coverage_checker(scenario: str, profile: dict, chunks: list[dict]) -> Claim:
    prompt = f"""ROLE: Coverage Checker. Independently try to show the person IS adequately covered
under the scenario: find riders, benefits, other policies, or assets that close any apparent gap. {DISCLAIMER}
Check that each benefit actually applies to THIS scenario's cause (read exclusions carefully).
Use the pre-computed numbers; do not do new arithmetic.
If, after honest effort, coverage is insufficient, set position "gap".

SCENARIO: {scenario}
PROFILE:
{profile_facts(profile)}
POLICY EXCERPTS (cite using the <ref> exactly; quote must be copied verbatim from that excerpt):
{format_chunks(chunks)}"""
    return call_llm(prompt, Claim, model="small", agent="checker")


def resolver(scenario: str, profile: dict, flag: Claim, check: Claim, cited_chunks: list[dict],
             dropped: dict) -> Ruling:
    prompt = f"""ROLE: Resolver (judge). Two analysts argued independently about whether a coverage gap exists.
Decide using ONLY the source text below, not the analysts' summaries. {DISCLAIMER}

Rules:
- "confirmed": the source text shows coverage/assets do not meet the scenario's needs.
- "dismissed": the source text shows the need is met (e.g. a benefit that applies to this exact cause).
- "inconclusive": the text genuinely does not settle it.
- Verify each benefit applies to this scenario's cause; exclusions override benefits.
- Citations removed by the verifier (could not be found in source) must be ignored.

SCENARIO: {scenario}
PROFILE (numbers computed by code, trust them):
{profile_facts(profile)}

RISK FLAGGER ({flag.position}): {flag.argument}
Valid refs: {[c.ref for c in flag.citations]} | removed: {dropped.get('flagger', [])}

COVERAGE CHECKER ({check.position}): {check.argument}
Valid refs: {[c.ref for c in check.citations]} | removed: {dropped.get('checker', [])}

SOURCE TEXT:
{format_chunks(cited_chunks)}"""
    return call_llm(prompt, Ruling, model="large", agent="resolver")


def writer(profile: dict, rulings: dict, scenarios: dict, metrics_out: dict) -> Report:
    prompt = f"""Write a short diagnostic coverage report. {DISCLAIMER}
Only restate numbers given below; never compute new ones. No product recommendations.
Benchmark: commonly cited guidance is an emergency fund of {metrics_out['benchmark_months']} months of expenses.

METRICS: {metrics_out}
SCENARIOS: {scenarios}
RULINGS: {rulings}"""
    return call_llm(prompt, Report, model="small", agent="writer")


def single_agent_baseline(scenario: str, profile: dict, chunks: list[dict]) -> Ruling:
    """Fair baseline: same documents, union of BOTH retrieval sets, strongest model, one pass."""
    prompt = f"""You are a financial coverage analyst. Decide whether this person has a coverage GAP under the
scenario. {DISCLAIMER}
- "confirmed": coverage/assets do not meet the scenario's needs.
- "dismissed": the need is met (e.g. a benefit that applies to this exact cause).
- "inconclusive": the text genuinely does not settle it.
Check exclusions, waiting periods, caps, riders, and requirements across ALL documents.
Use the pre-computed numbers; do not do new arithmetic.

SCENARIO: {scenario}
PROFILE:
{profile_facts(profile)}
POLICY EXCERPTS:
{format_chunks(chunks)}"""
    return call_llm(prompt, Ruling, model="large", agent="baseline")
