import re

from outputs import metrics


def profile_facts(profile: dict) -> str:
    """Numbers pre-computed in code, so agents never do the arithmetic."""
    lines = [
        f"Age {profile.get('age')}, {profile.get('occupation')}, dependents: {profile.get('dependents')}",
        f"Monthly income: Rs {profile['monthly_income']:,}",
        f"Monthly outgo (all expenses incl. EMIs): Rs {metrics.monthly_outgo(profile):,.0f} "
        f"-> breakdown {profile['monthly_expenses']}",
        f"Liquid assets: Rs {metrics.liquid_total(profile):,.0f} {profile['liquid_assets']}",
        f"Locked assets: {profile.get('locked_assets', {})}",
        f"Obligations: {profile.get('obligations', {})}",
        f"Runway from liquid assets alone: {metrics.runway_months(profile)} months",
        f"Runway if breakable locked assets are also used: {metrics.runway_months(profile, True)} months",
    ]
    if profile.get("known_conditions_note"):
        lines.append(f"Health note: {profile['known_conditions_note']}")
    return "\n".join(lines)


def format_chunks(chunks: list[dict]) -> str:
    return "\n".join(f"<{c['ref']}> {c['text']}" for c in chunks)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def validate_citations(claim, retrieved: list[dict]):
    """Deterministic guard: keep only citations whose ref was retrieved AND whose quote appears in it."""
    by_ref = {c["ref"]: _norm(c["text"]) for c in retrieved}
    kept, dropped = [], []
    for cit in claim.citations:
        text = by_ref.get(cit.ref)
        q = _norm(cit.quote)
        if text is not None and q and (q in text or _overlap(q, text) >= 0.8):
            kept.append(cit)
        else:
            dropped.append(cit.ref)
    claim.citations = kept
    return claim, dropped


def _overlap(q: str, text: str) -> float:
    qw = q.split()
    tw = set(text.split())
    return sum(w in tw for w in qw) / max(len(qw), 1)
