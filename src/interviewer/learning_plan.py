from __future__ import annotations
from typing import Dict, Any, List
import json



# -------------------------------------------------
# Resource Map (Normalized)
# Used as a safe fallback when LLM-based resource generation fails
# -------------------------------------------------

RESOURCE_MAP = {
    "python": [
        "Official Python Tutorial: https://docs.python.org/3/tutorial/",
        "Real Python (best practices): https://realpython.com/",
    ],
    "dsa": [
        "NeetCode Roadmap: https://neetcode.io/roadmap",
        "Blind 75 LeetCode List: https://leetcode.com/discuss/general-discussion/460599/blind-75-leetcode-questions",
    ],
    "apis": [
        "REST API Design: https://restfulapi.net/",
        "FastAPI Documentation: https://fastapi.tiangolo.com/",
    ],
    "databases": [
        "SQLBolt Interactive Lessons: https://sqlbolt.com/",
        "Use The Index, Luke: https://use-the-index-luke.com/",
    ],
    "system design": [
        "System Design Primer: https://github.com/donnemartin/system-design-primer",
        "Grokking System Design Interview: https://www.designgurus.io/",
    ],
    "communication": [
        "Explaining Solutions Clearly: https://www.youtube.com/watch?v=VZ8R7k8o2mQ",
    ],
}


# -------------------------------------------------
# Personality Boost (IMPORTANT)
# -------------------------------------------------

PERSONALITY_PRIORITY = {
    "Strict FAANG": ["dsa", "system design", "databases"],
    "Professor": ["system design", "databases", "python"],
    "Startup Hiring Manager": ["apis", "system design", "databases"],
    "Friendly Coach": ["python", "dsa", "communication"],
}


# -------------------------------------------------
# Helpers
# -------------------------------------------------

# Normalize raw topic names into predefined resource categories
def _normalize_topic(t: str) -> str:
    t = t.lower()
    if "api" in t:
        return "apis"
    if "db" in t or "sql" in t:
        return "databases"
    if "system" in t:
        return "system design"
    if "python" in t:
        return "python"
    if "dsa" in t or "algorithm" in t:
        return "dsa"
    if "communicat" in t:
        return "communication"
    return t



# Generate topic-specific learning resources using LLM
def fetch_grounded_courses(llm, weak_topics: List[str], role: str) -> List[Dict[str, Any]]:
    """
    Returns topic-aware, LLM-generated learning resources.
    """

    topics = ", ".join(weak_topics)

    prompt = f"""
You are an interview coach.

Role: {role}
Weak topics: {topics}

For EACH weak topic, recommend 2–3 HIGH-QUALITY learning resources.

Rules:
- Resources MUST be topic-specific (not platform homepages)
- Use real courses/tutorials (Coursera, Udemy, freeCodeCamp, YouTube, official docs)
- Include topic name explicitly
- Include direct working URLs
- No explanations
- Return ONLY valid JSON

JSON FORMAT:
[
  {{
    "topic": "Databases",
    "resources": [
      "SQL Basics – freeCodeCamp: https://www.freecodecamp.org/learn/sql/",
      "Database Fundamentals – Coursera: https://www.coursera.org/learn/database-management"
    ]
  }}
]
"""

    try:
        raw = llm.invoke_text(prompt)
        data = json.loads(raw.strip())
        if isinstance(data, list):
            return data
    except Exception:
        pass

    return []


# Generate concrete, actionable practice tasks using LLM
def fetch_practice_topics(llm, weak_topics: List[str], role: str) -> List[str]:
    """
    Uses LLM to generate topic-specific practice tasks and next steps.
    """

    topics = ", ".join(weak_topics)

    prompt = f"""
You are an interview coach.

Role: {role}
Weak areas: {topics}

Suggest 4–6 concrete practice tasks the candidate should do next.

Rules:
- Be specific and actionable
- Focus on interview preparation
- No explanations
- Return ONLY a JSON array of strings

Example:
[
  "Practice SQL JOIN and GROUP BY problems using real datasets",
  "Design a token-based authentication system supporting multiple devices"
]
"""

    try:
        raw = llm.invoke_text(prompt)
        data = json.loads(raw.strip())
        if isinstance(data, list):
            return data
    except Exception:
        pass

    return []





# -------------------------------------------------
# Resource Formatter (UI / DOCX Safe)
# -------------------------------------------------

# Format learning resources into plain-text lines
def format_resources_for_display(resource_blocks: List[Dict[str, Any]]) -> List[str]:
    """
    Plain-text output ONLY (no bullets, no markdown, no icons).
    Matches first screenshot exactly.
    """
    lines: List[str] = []

    for block in resource_blocks:
        topic = block.get("topic", "General")
        lines.append(f"{topic}")

        for res in block.get("resources", []):
            if ":" in res:
                title, link = res.split(":", 1)
                lines.append(f"{title.strip()}: {link.strip()}")
            else:
                lines.append(res)

        lines.append("")  # spacing

    return lines



# -------------------------------------------------
# Learning Plan Builder
# -------------------------------------------------

# Build a personalized learning plan based on interview evaluation
def build_learning_plan(
    role: str,
    evaluation: Dict[str, Any],
    personality: str = "Friendly Coach",
) -> Dict[str, Any]:

    score = int(evaluation.get("score", 0))
    breakdown = evaluation.get("topic_breakdown", {}) or {}

    overall_focus = (
    f"This learning plan is based on your overall interview performance "
    f"({score}/10), considering all rounds and observed weaknesses."
    )
    # ---------------- Weak Topics ----------------
    weak_topics = []
    for topic, stats in breakdown.items():
        if stats.get("weak", 0) > 0:
            weak_topics.append(_normalize_topic(topic))

    if not weak_topics:
        weak_topics = PERSONALITY_PRIORITY.get(personality, ["dsa", "python"])

    weak_topics = list(dict.fromkeys(weak_topics))

    # ---------------- Roadmap Steps (DIMENSION-DRIVEN) ----------------
    steps: List[str] = []



    # ---------------- LLM-generated Practice Topics ----------------
    try:
        from .llm import LLM
        llm = LLM()
        practice_topics = fetch_practice_topics(llm, weak_topics, role)
        steps.extend(practice_topics)
    except Exception:
        pass


    dimensions = evaluation.get("dimensions", {})

    # ---- Weakness-based steps (NOT score-based) ----
    if dimensions.get("correctness", 10) <= 5:
        steps.append("Reinforce correctness by revising fundamentals and validating logic step by step.")

    if dimensions.get("clarity", 10) <= 5:
        steps.append("Practice structured explanations using problem → approach → solution format.")

    if dimensions.get("communication", 10) <= 5:
        steps.append("Improve verbal communication by explaining solutions aloud and summarizing answers.")

    if dimensions.get("technical_depth", 10) <= 5:
        steps.append("Strengthen technical depth by revisiting core concepts and internal mechanisms.")

    if dimensions.get("confidence", 10) <= 5:
        steps.append("Build confidence through mock interviews and timed practice.")

    # ---- Intensity modifier (ONLY depth, not content) ----
    if score <= 4:
        steps.append("Start with easy problems and gradually increase difficulty.")
    elif score <= 7:
        steps.append("Focus on medium-level problems with time constraints.")
    else:
        steps.append("Practice advanced problems, edge cases, and optimizations.")


    # Personality emphasis
    if personality == "Strict FAANG":
        steps.append("Daily DSA practice with strict time limits.")
    elif personality == "Professor":
        steps.append("Focus on theoretical correctness and formal explanations.")
    elif personality == "Startup Hiring Manager":
        steps.append("Emphasize real-world design decisions and trade-offs.")

    # ---------------- Resources (INTERVIEW-WIDE, NON-STACKING) ----------------
    resources: List[Dict[str, Any]] = []

    # Absolute safety fallback (STRUCTURED)
    if not resources:
        for t in weak_topics:
            if t in RESOURCE_MAP:
                resources.append({
                    "topic": t.title(),
                    "resources": RESOURCE_MAP[t]
                })

        if not resources:
            resources.append({
                "topic": "DSA",
                "resources": RESOURCE_MAP["dsa"]
            })




    return {
        "overall_focus": overall_focus,
        "weak_topics": weak_topics,
        "steps": steps,
        "resources": format_resources_for_display(resources),
    }



