"""Deterministic money math. No LLM touches these numbers."""

EMERGENCY_FUND_BENCHMARK_MONTHS = 6


def monthly_outgo(profile: dict) -> float:
    return float(sum(profile["monthly_expenses"].values()))


def liquid_total(profile: dict) -> float:
    return float(sum(profile["liquid_assets"].values()))


def locked_accessible_total(profile: dict) -> float:
    """Locked assets that can be broken early, net of penalty. Assets with a lock-in date are excluded."""
    total = 0.0
    for asset in profile.get("locked_assets", {}).values():
        if "premature_withdrawal_penalty_pct" in asset:
            total += asset["amount"] * (1 - asset["premature_withdrawal_penalty_pct"] / 100)
    return total


def runway_months(profile: dict, include_locked: bool = False) -> float:
    outgo = monthly_outgo(profile)
    if outgo <= 0:
        return float("inf")
    cash = liquid_total(profile) + (locked_accessible_total(profile) if include_locked else 0)
    return round(cash / outgo, 1)


def severity(runway: float) -> str:
    if runway < 3:
        return "high"
    if runway < EMERGENCY_FUND_BENCHMARK_MONTHS:
        return "medium"
    return "low"


def compute(profile: dict, verdicts: dict) -> dict:
    r = runway_months(profile)
    r_locked = runway_months(profile, include_locked=True)
    confirmed = [sid for sid, v in verdicts.items() if v["verdict"] == "confirmed"]
    return {
        "monthly_outgo": monthly_outgo(profile),
        "liquid_assets": liquid_total(profile),
        "runway_months": r,
        "runway_months_with_locked": r_locked,
        "benchmark_months": EMERGENCY_FUND_BENCHMARK_MONTHS,
        "severity": severity(r) if confirmed else "none",
        "confirmed_gaps": confirmed,
    }
