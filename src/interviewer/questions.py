from __future__ import annotations

import json
import random
import re
from typing import List, Dict, Any, Optional

from .rag import retrieve_resume_context


# -------------------------------------------------
# Constants
# -------------------------------------------------

LANGS = ["python", "java", "javascript", "typescript", "sql", "c++", "c#", "go"]

ROLE_FOCUS = {
    "backend": ["api", "authentication", "database", "scalability", "caching"],
    "frontend": ["react", "state management", "performance", "accessibility"],
    "data": ["sql", "etl", "statistics", "pipelines", "data modeling"],
    "full stack": ["api", "frontend", "database", "system design"],
}

EDU_STOPWORDS = [
    "education", "b.tech", "btech", "degree", "cgpa", "gpa",
    "college", "university", "school", "course", "coursework",
    "semester", "syllabus", "curriculum", "certification", "nptel",
]

YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
YEAR_RANGE_RE = re.compile(r"\b(19\d{2}|20\d{2})\s*[-–]\s*(19\d{2}|20\d{2})\b")


# -------------------------------------------------
# Personality Behavior (QUESTION STYLE)
# -------------------------------------------------

PERSONALITY_RULES = {
    "Friendly Coach": [
        "Use encouraging, learning-oriented wording.",
        "Prefer explanation and understanding over trickiness.",
    ],
    "Strict FAANG": [
        "Make questions deep and challenging.",
        "Focus on edge cases, complexity, and trade-offs.",
        "Avoid hand-holding language.",
    ],
    "Startup Hiring Manager": [
        "Focus on real-world decisions and production constraints.",
        "Ask how things break and how to fix them.",
    ],
    "Professor": [
        "Focus on theory, definitions, and correctness.",
        "Encourage structured reasoning.",
    ],
}


# -------------------------------------------------
# Personality Difficulty & Expectations (STEP 5A)
# -------------------------------------------------

PERSONALITY_DIFFICULTY = {
    "Friendly Coach": {
        "difficulty": "easy",
        "expectation": "basic understanding and clear explanation",
    },
    "Professor": {
        "difficulty": "theory-heavy",
        "expectation": "formal definitions, correctness, and proofs",
    },
    "Startup Hiring Manager": {
        "difficulty": "practical",
        "expectation": "real-world impact, trade-offs, and scalability",
    },
    "Strict FAANG": {
        "difficulty": "hard",
        "expectation": "optimal solutions, edge cases, and complexity analysis",
    },
}



# -------------------------------------------------
# Numeric Difficulty → LLM Difficulty Mapping
# -------------------------------------------------

DIFFICULTY_LEVEL_MAP = {
    1: "easy",
    2: "medium",
    3: "medium-hard",
    4: "hard",
    5: "very hard",
}


# -------------------------------------------------
# Resume Parsing
# -------------------------------------------------

def extract_languages(resume_text: str) -> List[str]:
    text = (resume_text or "").lower()
    langs = [l for l in LANGS if re.search(rf"\b{re.escape(l)}\b", text)]
    return langs or ["python"]


def _looks_like_education(line: str) -> bool:
    low = line.lower()
    if any(w in low for w in EDU_STOPWORDS):
        return True
    if YEAR_RANGE_RE.search(low):
        return True
    if YEAR_RE.search(low) and any(w in low for w in ["college", "university", "school"]):
        return True
    return False


def extract_project_names(resume_text: str) -> List[str]:
    projects: List[str] = []
    lines = [l.strip() for l in (resume_text or "").splitlines() if l.strip()]

    in_projects = False
    in_education = False

    for line in lines:
        low = line.lower()

        if "education" in low:
            in_education = True
        if in_education and any(k in low for k in ["projects", "experience", "skills"]):
            in_education = False

        if any(k in low for k in ["projects", "project experience"]):
            in_projects = True
            continue
        if in_projects and any(k in low for k in ["education", "skills", "certifications"]):
            in_projects = False

        if in_education or _looks_like_education(line):
            continue

        cleaned = re.sub(r"^[•\-\*\d\.\)\s]+", "", line)

        if in_projects and 6 <= len(cleaned) <= 90:
            projects.append(cleaned[:80])
        elif any(k in low for k in ["system", "dashboard", "platform", "portal", "application"]):
            projects.append(cleaned[:80])

    return list(dict.fromkeys(projects))[:6]


# -------------------------------------------------
# Role Requirements
# -------------------------------------------------

def role_requirements(role: str) -> List[str]:
    r = (role or "").lower()
    for key in ROLE_FOCUS:
        if key in r:
            return ROLE_FOCUS[key]
    return ["dsa", "apis", "databases"]


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def _ensure_hints(q: Dict[str, Any]) -> Dict[str, Any]:
    if not q.get("hints"):
        q["hints"] = [
            "Define the concept clearly",
            "Explain how it works internally",
            "Give a real-world example",
        ]
    return q


def _is_vague_question(q) -> bool:
    if not isinstance(q, str):
        return True
    return len(q.split()) < 7




def build_hr_questions(
    llm,
    state,
    count: int = 1,  # generate ONE question at a time
) -> List[Dict[str, Any]]:
    """
    Generate HR questions using LLM ONLY.
    No fallback. No hardcoded questions.
    """

    prompt = {
        "task": "Generate HR interview questions",
        "role": state.candidate.role,
        "experience": state.candidate.experience_level,
        "personality": state.candidate.personality,
        "rules": [
            f"Generate HR interview questions in the style of a {state.candidate.personality}.",
            "Questions must be situational or reflective.",
            "Do NOT ask generic HR questions.",
            "Do NOT reuse common templates.",
            "Each question must feel distinct.",
            "Return ONLY valid JSON.",
        ],
        "schema": {
            "questions": [
                {"question": "string"}
            ]
        }
    }

    for _ in range(3):  # retry LLM, never fallback
        raw = llm.invoke_text(json.dumps(prompt))

        match = re.search(
            r"\{\s*\"questions\"\s*:\s*\[.*?\]\s*\}",
            raw,
            re.DOTALL
        )
        if not match:
            continue

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return []  # 🔒 HARD FAIL SAFE — no HR question


        for q in data.get("questions", []):
            if q.get("question"):
                return [{
                    "question": q["question"],
                    "topic": "HR",
                    "qtype": "hr",
                }]


    # 🚨 HARD FAIL — no silent fallback
    raise RuntimeError("LLM failed to generate HR questions")


   

# -------------------------------------------------
# HR Follow-up Question Generator (Answer-based)
# -------------------------------------------------

def build_hr_followup(
    llm,
    prev_question: str,
    prev_answer: str,
    personality: str,
    run_nonce: str,
) -> Optional[str]:
    """
    Generate ONE HR follow-up question based on the candidate's answer.
    """

    prompt = {
        "task": "Generate ONE HR follow-up interview question",
        "nonce": run_nonce,
        "personality": personality,
        "previous_question": prev_question,
        "candidate_answer": prev_answer,
        "rules": [
            "Question must match interviewer personality:",
            "- Friendly Coach → supportive reflection",
            "- Professor → ethical or reasoning-based",
            "- Startup Hiring Manager → ownership and ambiguity",
            "- Strict FAANG → pressure and accountability",
        ],
        "schema": {
            "question": "string"
        },
    }

    try:
        raw = llm.invoke_text(json.dumps(prompt))
        data = json.loads(raw)
        q = data.get("question")
        if q and len(q.split()) >= 6:
            return q
    except Exception:
        pass

    return None



# -------------------------------------------------
# Follow-up Question Generator (PERSONALITY AWARE)
# -------------------------------------------------

def build_followup_question(
    llm,
    role: str,
    resume_text: str,
    prev_question: str,
    prev_answer: str,
    run_nonce: str,
    personality: str = "Friendly Coach",
) -> Optional[Dict[str, Any]]:

    hits = retrieve_resume_context(
        user_id="unused",
        query=f"deep technical details for {role}",
        k=5,
    )

    resume_context = "\n".join(
        h["text"] if isinstance(h, dict) else str(h)
        for h in hits
    )[:1500]

    prompt = {
        "task": "Generate ONE deeper technical follow-up question",
        "nonce": run_nonce,
        "personality": personality,
        "style_rules": PERSONALITY_RULES.get(personality, []),
        "role": role,
        "resume_context": resume_context,
        "previous_question": prev_question,
        "candidate_answer": prev_answer,
        "rules": [
            "Must go deeper than before.",
            "Avoid yes/no questions.",
            "Avoid STAR or behavioral questions.",
            "Return ONLY JSON.",
        ],
        "schema": {
            "topic": "string",
            "qtype": "resume|theory|coding|system",
            "question": "string",
            "hints": ["string"],
        },
    }

    try:
        raw = llm.invoke_text(json.dumps(prompt))
        data = json.loads(raw)

        if (
            isinstance(data, dict)
            and data.get("question")
            and not _is_vague_question(data["question"])
        ):
            return _ensure_hints(data)

    except Exception:
        return None




def safe_json_object(text: str) -> dict:
    """
    Extract the FIRST valid JSON object from LLM output.
    Handles extra text, markdown, explanations.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group())




# -------------------------------------------------
# Interview Plan Generator (PERSONALITY AWARE)
# -------------------------------------------------

def build_technical_question(
    llm,
    user_id: str,
    role: str,
    resume_text: str,
    experience: str,
    run_nonce: str,
    n_questions: int = 8,
    personality: str = "Friendly Coach",
    difficulty: str | None = None,
    difficulty_level: int | None = None,
    focus: str | None = None,
):




    # -------------------------------------------------
    # STEP 5B: Resolve personality difficulty & expectation
    # -------------------------------------------------
    profile = PERSONALITY_DIFFICULTY.get(
        personality,
        PERSONALITY_DIFFICULTY["Friendly Coach"]
    )

    # 🔥 Resolve numeric → textual difficulty
    if difficulty_level is not None:
        safe_level = max(1, min(5, difficulty_level))
        difficulty = DIFFICULTY_LEVEL_MAP[safe_level]

    else:
        difficulty = difficulty or profile["difficulty"]

    expectation = profile["expectation"]


    langs = extract_languages(resume_text)
    projects = extract_project_names(resume_text)
    reqs = role_requirements(role)

    hits = retrieve_resume_context(
        user_id=user_id,
        query=f"projects, technologies, APIs related to {role}",
        k=12,
    )

    resume_context = "\n".join(
        h["text"] if isinstance(h, dict) else str(h)
        for h in hits
    )[:4000]

    prompt = {
        "nonce": run_nonce,
        "personality": personality,
        "style_rules": PERSONALITY_RULES.get(personality, []),
        "role": role,
        "experience": experience,
        "languages": langs,
        "question_type": focus,
        "projects": projects,
        "focus_topics": reqs,
        "resume_context": resume_context,
        "difficulty": difficulty,
        "expectation": expectation,

       "rules": [
            f"Difficulty level: {difficulty}",
            f"Expected depth: {expectation}",
            f"Generate ONE {focus}-based technical interview question.",
            "resume → must reference candidate projects or experience",
            "skill → focus on specific tools or technologies",
            "coding → must require code or pseudocode",
            "system → must discuss design, trade-offs, failures",
            "Match interviewer personality strictly",
            "Return ONLY valid JSON"
        ],

        "schema": {
            "topic": "string",
            "qtype": "resume|theory|coding|system",
            "question": "string",
            "hints": ["string"]
        }

    }

    # 🔁 RETRY LLM 3 TIMES
    for attempt in range(3):
        try:
            raw = llm.invoke_text(json.dumps(prompt))
            data = safe_json_object(raw)

            if (
                isinstance(data, dict)
                and data.get("question")
                and not _is_vague_question(data["question"])
            ):
                return _ensure_hints(data)

        except Exception as e:
            print(f"⚠️ Technical LLM attempt {attempt + 1} failed:", e)


    # 🔁 CONTROLLED FALLBACK (NOT SILENT)
    return {
        "topic": "Technical",
        "qtype": focus or "theory",
        "question": (
            "Based on your experience, explain a technical problem you worked on, "
            "why it was challenging, and how you solved it."
        ),
        "hints": [
            "Describe the system or context",
            "Explain the root cause",
            "Explain your solution and trade-offs",
        ],
        "_fallback": True,   # 👈 VERY IMPORTANT
    }
