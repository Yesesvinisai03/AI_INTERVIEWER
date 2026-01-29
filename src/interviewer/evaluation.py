from __future__ import annotations
from collections import defaultdict
from typing import List, Dict, Any

from .state import AnswerRecord


# -------------------------------------------------
# Scoring Dimensions
# -------------------------------------------------

DIMENSIONS = [
    "technical_depth",
    "clarity",
    "confidence",
    "communication",
    "correctness",
]

# -------------------------------------------------
# Per-answer scoring logic
# -------------------------------------------------

# Score a single answer across multiple dimensions
def _score_answer(answer: str, quality: str, hints_used: int) -> Dict[str, int]:
    """
    Returns per-dimension score (0–2 per dimension)
    """
    scores = {d: 0 for d in DIMENSIONS}
    text = (answer or "").lower()
    words = text.split()
    length = len(words)

    # ---------------- Communication ----------------
    if length >= 25:
        scores["communication"] = 2
    elif length >= 12:
        scores["communication"] = 1

    # ---------------- Clarity ----------------
    if any(k in text for k in ["because", "therefore", "for example", "trade-off", "however"]):
        scores["clarity"] = 2
    elif length >= 10:
        scores["clarity"] = 1

    # ---------------- Confidence ----------------
    if any(k in text for k in ["i think", "maybe", "not sure", "probably", "i guess"]):
        scores["confidence"] = 0
    elif length >= 10:
        scores["confidence"] = 2
    else:
        scores["confidence"] = 1

    # ---------------- Technical Depth ----------------
    if any(k in text for k in [
        "algorithm", "api", "database", "latency", "scalability",
        "memory", "thread", "complexity", "architecture", "cache"
    ]):
        scores["technical_depth"] = 2
    elif length >= 15:
        scores["technical_depth"] = 1

    # ---------------- Correctness ----------------
    scores["correctness"] = 2 if quality == "good" else 0

    # ---------------- Hint Penalty (IMPORTANT) ----------------
    if hints_used > 0:
        scores["confidence"] = max(0, scores["confidence"] - hints_used)
        scores["clarity"] = max(0, scores["clarity"] - hints_used)

    return scores

# -------------------------------------------------
# Main Evaluation Function
# -------------------------------------------------

# Computes dimension scores, strengths, weaknesses, and summary
def evaluate_interview(records: List[AnswerRecord]) -> Dict[str, Any]:
    if not records:
        return {
            "score": 0,
            "dimensions": {},
            "strengths": [],
            "improvements": [],
            "topic_breakdown": {},
            "summary": "No answers were recorded.",
            "total_hints": 0,
        }

    dimension_totals = defaultdict(int)
    dimension_max = defaultdict(int)
    topic_breakdown = defaultdict(lambda: {"good": 0, "weak": 0})
    strengths: List[str] = []
    improvements: List[str] = []
    total_hints = 0

    # ---------------- Aggregate Scores ----------------
    for r in records:
        total_hints += r.hints_used
        scores = _score_answer(r.answer, r.quality, r.hints_used)

        for d, v in scores.items():
            dimension_totals[d] += v
            dimension_max[d] += 2

        topic_breakdown[r.topic]["good" if r.quality == "good" else "weak"] += 1
        
        # ---------------- HR-specific insights ----------------
    hr_records = [r for r in records if r.round_name == "hr"]

    if hr_records:
        avg_len = sum(len(r.answer.split()) for r in hr_records) / len(hr_records)

        if avg_len >= 25:
            strengths.append("HR Communication: clear and confident responses")
        else:
            improvements.append("HR Communication: answers need more structure")

        if any("conflict" in r.answer.lower() or "challenge" in r.answer.lower() for r in hr_records):
            strengths.append("HR Situational Handling: demonstrated maturity")

    # ---------------- Normalize to /10 ----------------
    dimension_scores = {}
    for d in DIMENSIONS:
        if dimension_max[d] == 0:
            dimension_scores[d] = 0
        else:
            dimension_scores[d] = round((dimension_totals[d] / dimension_max[d]) * 10)

    overall_score = round(sum(dimension_scores.values()) / len(DIMENSIONS))

    # ---------------- Strengths & Improvements ----------------
    strengths = []
    improvements = []

    for d, s in dimension_scores.items():
        label = d.replace("_", " ").title()
        if s >= 7:
            strengths.append(f"{label}: strong performance")
        elif s <= 4:
            improvements.append(f"{label}: needs improvement")

    summary = (
        f"Overall performance score: **{overall_score}/10**.\n\n"
        f"Strong areas: {', '.join(strengths) or '—'}.\n"
        f"Focus areas: {', '.join(improvements) or '—'}."
    )

    return {
        "score": overall_score,
        "dimensions": dimension_scores,
        "strengths": strengths,
        "improvements": improvements,
        "topic_breakdown": dict(topic_breakdown),
        "summary": summary,
        "total_hints": total_hints,  # ✅ REQUIRED FOR UI BADGE
    }

# Score a single interview round out of 10
def score_round(
    records,
    round_name: str,
    whiteboard_bonus: int = 0,
) -> float:
    """
    Scores a single round out of 10.
    Adds a small bonus for effective whiteboard usage.
    """

    relevant = [r for r in records if r.round_name == round_name]
    if not relevant:
        return 0.0

    good = sum(1 for r in relevant if r.quality == "good")
    base_score = (good / len(relevant)) * 10

    # ✅ Whiteboard bonus (max +1)
    final_score = min(10.0, round(base_score + whiteboard_bonus, 1))
    return final_score


# Generate final downloadable report using interview state
# Uses state.final_evaluation as the single source of truth
def generate_downloadable_report(state) -> str:
    """
    Generates a plain-text downloadable interview report
    including the Final Evaluation paragraph.
    """

    lines = []
    lines.append("AI INTERVIEW FEEDBACK REPORT")
    lines.append("=" * 30)
    lines.append(f"Candidate: {state.candidate.name}")
    lines.append(f"Role: {state.candidate.role}")
    lines.append("")

    # Round scores
    for r, s in state.round_scores.items():
        lines.append(f"{r.upper()} ROUND SCORE: {s}/10")

    lines.append("")
    lines.append(f"FINAL VERDICT: {state.final_verdict}")
    lines.append("")

    # ✅ FINAL EVALUATION (from state, single source of truth)
    if state.final_evaluation:
        lines.append("FINAL EVALUATION")
        lines.append("-" * 20)
        lines.append(state.final_evaluation)
    else:
        lines.append("FINAL EVALUATION")
        lines.append("-" * 20)
        lines.append("Final evaluation was not generated.")

    return "\n".join(lines)
