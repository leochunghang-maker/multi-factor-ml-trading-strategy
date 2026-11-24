"""Interview guide must describe the current platform, not the pre-November-2026 stack."""

from __future__ import annotations

from pathlib import Path

GUIDE = Path("docs/quant_interview_guide.md")


def _folded(text: str) -> str:
    return " ".join(text.split())


def test_interview_guide_matches_current_platform() -> None:
    text = GUIDE.read_text()
    folded = _folded(text)

    required = (
        "point-in-time eligibility",
        "residual survivorship",
        "nested",
        "one-SE",
        "purge",
        "trial_log",
        "Deflated Sharpe",
        "half-spread",
        "borrow",
        "ADV participation",
        "beta-neutral",
        "sector-neutral",
        "completed NYSE session",
        "no broker client",
        "statistically significant alpha",
    )
    missing = [phrase for phrase in required if phrase not in folded]
    assert not missing, f"interview guide missing current-platform phrases: {missing}"

    stale = (
        "Historical constituents are not point-in-time.",
        "I would improve the data layer first, especially point-in-time universe membership and data quality checks.",
        "A production-quality version would model spread, market impact, liquidity, borrow costs, financing, and order size.",
    )
    present = [phrase for phrase in stale if phrase in folded]
    assert not present, f"interview guide still has stale claims: {present}"

    assert "live-money" in folded
    assert "not production-ready" in folded
