from __future__ import annotations

import json
import uuid
import re
import random
from datetime import datetime
from typing import List

from langgraph.graph import StateGraph, END

from .state import InterviewState, Turn, AnswerRecord
from .questions import build_technical_question, build_followup_question, build_hr_questions, build_hr_followup
from .evaluation import score_round

# --------------------------------------------------
# Constants
# --------------------------------------------------

#while stopping interview
END_PHRASES = {
    "no", "nope", "nothing", "nothing else",
    "that's all", "that is all", "all good",
    "no questions", "none", "done", "stop", "stop interview"
}


#interview rounds
ROUNDS = ["hr", "technical", "behavioral"]


#score required to pass each round
PASS_MARK = {
    "hr": 5.0,
    "technical": 6.0,     # 🔒 Gatekeeper
    "behavioral": 5.0,
}


# --------------------------------------------------
# Interviewer Personalities (Behavior)
# each personality controls tone, flow
# --------------------------------------------------

PERSONALITIES = {
    "Friendly Coach": {
        "weak_limit": 3,
        "max_depth": 2,
        "msg_need_time": "No worries 🙂 Take your time. Tell me when you’re ready.",
        "closing_prefix": "Good effort overall 🙂",
    },
    "Strict FAANG": {
        "weak_limit": 2,
        "max_depth": 3,
        "msg_need_time": "Take 30–60 seconds. Reply when ready.",
        "closing_prefix": "We’ll stop here.",
    },
    "Startup Hiring Manager": {
        "weak_limit": 2,
        "max_depth": 3,
        "msg_need_time": "Sure — take a moment and reply when ready.",
        "closing_prefix": "Let’s pause here.",
    },
    "Professor": {
        "weak_limit": 2,
        "max_depth": 2,
        "msg_need_time": "Take your time and organize your thoughts. Reply when ready.",
        "closing_prefix": "We’ll conclude here.",
    },
}



# --------------------------------------------------
# Hint limits per personality (REAL INTERVIEW)
# --------------------------------------------------
HINT_LIMITS = {
    "Friendly Coach": 3,
    "Startup Hiring Manager": 2,
    "Professor": 1,
    "Strict FAANG": 0,
}


PERSONALITY_START_LEVEL = {
    "Friendly Coach": 1,
    "Startup Hiring Manager": 2,
    "Professor": 3,
    "Strict FAANG": 4,
}

PERSONALITY_DIFFICULTY_BOUNDS = {
    "Friendly Coach": (1, 3),
    "Startup Hiring Manager": (2, 4),
    "Professor": (2, 4),
    "Strict FAANG": (3, 5),
}



# --------------------------------------------------
# Personality-based Difficulty Jump Size
# --------------------------------------------------

PERSONALITY_DIFFICULTY_JUMP = {
    "Friendly Coach": {
        "up": 1,
        "down": 1,
    },
    "Startup Hiring Manager": {
        "up": 1,
        "down": 1,
    },
    "Professor": {
        "up": 2,   # 🔥 Professor escalates faster
        "down": 1,
    },
    "Strict FAANG": {
        "up": 1,
        "down": 2,  # 🔥 FAANG punishes weak answers harder
    },
}


# --------------------------------------------------
# Personality → Question Strategy
# Strategy defining focus and question limits per personality
# --------------------------------------------------

PERSONALITY_STRATEGY = {
    "Friendly Coach": {
        "difficulty": "easy",
        "focus": "fundamentals, intuition, learning mindset",
        "hr_min": 5,
        "hr_max": 6,
        "behavioral_min": 5,
        "behavioral_max": 6,
    },
    "Professor": {
        "difficulty": "theory-heavy",
        "focus": "formal definitions, models, correctness",
        "hr_min": 5,
        "hr_max": 6,
        "behavioral_min": 5,
        "behavioral_max": 6,
    },
    "Startup Hiring Manager": {
        "difficulty": "practical",
        "focus": "scalability, trade-offs, impact, metrics",
        "hr_min": 5,
        "hr_max": 6,
        "behavioral_min": 5,
        "behavioral_max": 6,
    },
    "Strict FAANG": {
        "difficulty": "hard",
        "focus": "edge cases, optimization, precision",
        "hr_min": 5,
        "hr_max": 6,
        "behavioral_min": 4,
        "behavioral_max": 5,
    },
}



# --------------------------------------------------
# Interviewer Identity
# --------------------------------------------------

INTERVIEWER_PROFILES = {
    "Friendly Coach": {"name": "Alex", "intro": "your interview coach"},
    "Strict FAANG": {"name": "Smith", "intro": "a Senior Software Engineer at Google"},
    "Startup Hiring Manager": {"name": "Ryan", "intro": "an Engineering Manager at a fast-growing startup"},
    "Professor": {"name": "Dr. Raj", "intro": "a Professor of Computer Science"},
}

# --------------------------------------------------
# Helpers
# --------------------------------------------------

# Generate greeting based on current system time
def time_based_greeting() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning"
    if 12 <= hour < 17:
        return "Good afternoon"
    return "Good evening"


def _norm(text: str) -> str:
    return (text or "").strip().lower()



# Prevents duplicate summaries or continued questioning
def hard_stop_if_rejected(state):
    # 🔒 NEVER re-run summary
    if (
        getattr(state, "final_verdict", None) == "NOT SELECTED"
        and state.phase != "closing"
        and not getattr(state, "summary_emitted", False)
    ):
        return finalize_interview(state)
    return None


# DIFFICULTY ADJUSTMENT LOGIC
def adjust_difficulty(state: InterviewState, direction: str):
    personality = state.candidate.personality

    min_d, max_d = PERSONALITY_DIFFICULTY_BOUNDS[personality]
    jump = PERSONALITY_DIFFICULTY_JUMP[personality][direction]

    if direction == "up":
        state.difficulty_level = min(
            state.difficulty_level + jump,
            max_d
        )
    elif direction == "down":
        state.difficulty_level = max(
            state.difficulty_level - jump,
            min_d
        )

    state.last_difficulty_change = direction


# Detects if the current question matches a previously answered concept
def find_similar_concept_record(state: InterviewState, question: str):
    """
    Finds a previous GOOD answer to the SAME CONCEPT.
    Ignores dont-know / weak answers.
    """
    q_words = set(_norm(question).split())

    for r in reversed(state.records):
        # ✅ only consider GOOD answers
        if r.quality != "good":
            continue

        if r.answer.strip() == "":
            continue


        prev_q_words = set(_norm(r.question).split())
        overlap = len(q_words & prev_q_words) / max(len(prev_q_words), 1)

        if overlap > 0.6:
            return r

    return None




# Fast keyword-based intent override (skip, hint, dont_know)
def quick_intent_override(answer: str) -> str | None:
    a = answer.strip().lower()

    if a in {"next", "skip", "move on"}:
        return "skip"

    if a in {"hint", "help", "can i get hint", "can i get help"}:
        return "hint"

    if a in {"i dont know", "i don't know", "not sure", "no idea"}:
        return "dont_know"

    return None




# LLM-based intent detection with structured JSON output
# Determines how to handle the user's response

def llm_detect_intent(llm, question: str, answer: str) -> dict:
    """
    Returns structured intent + confidence.
    """
    prompt = f"""
You are an interview evaluator.

Question:
{question}

Candidate Answer:
{answer}

Classify the response strictly into ONE intent:

- answer
- needs_time
- dont_know
- hint
- filler
- clarification (asking for more details about the question)


Also say whether the candidate attempted to answer.

Return ONLY valid JSON:
{{
  "intent": "answer",
  "attempted": true
}}
"""

    try:
        raw = llm.invoke_text(prompt)
        data = json.loads(raw.strip())
        if data.get("intent") in {"answer","needs_time","dont_know","hint","filler","clarification"}:
            return data
    except Exception:
        pass

    return {"intent": "answer", "attempted": True}


# Detect semantic repetition between two answers using LLM
def llm_detect_repetition(llm, prev_answer: str, current_answer: str) -> bool:
    prompt = f"""
Do these two answers express the same idea semantically?

Previous answer:
{prev_answer}

Current answer:
{current_answer}

Return ONLY valid JSON:
{{"repeated": true}} or {{"repeated": false}}
"""
    try:
        data = json.loads(llm.invoke_text(prompt))
        return bool(data.get("repeated", False))
    except Exception:
        return False

# Detects whether the candidate reused the same answer anywhere
def repeated_answer_anywhere(llm, state: InterviewState, current_question: str, answer: str) -> bool:
    ans = _norm(answer)
    words = ans.split()

    # Too short → never block
    if len(words) < 12:
        return False

    for r in state.records:
        # Ignore same question
        if _norm(r.question) == _norm(current_question):
            continue

        prev = _norm(r.answer)

        # ✅ Case 1: near-exact copy (strong signal)
        if ans == prev:
            return True

        # ✅ Case 2: very high token overlap (copy with tiny edits)
        overlap = len(set(words) & set(prev.split())) / max(len(set(prev.split())), 1)

        if overlap > 0.85:
            return True

        # ❌ DO NOT do semantic LLM comparison here

    return False


# Generate a short, personality-aligned hint without revealing the answer
def generate_dynamic_hint(llm, question, answer, topic, personality):
    prompt = f"""
You are acting as a {personality} interviewer.

Interview Question:
{question}

Candidate Answer:
{answer}

Topic:
{topic}

Task:
Give ONE short HINT to guide the candidate toward the correct answer.

Rules:
- Do NOT give the full answer
- Do NOT judge the candidate
- Be appropriate to the interviewer style
- 1–2 sentences max
"""

    try:
        return llm.invoke_text(prompt).strip()
    except Exception:
        return "Think about the core concept and try again."


# Generate a one-sentence interviewer reaction
def llm_generate_reaction(
    llm,
    personality: str,
    round_name: str,
    question: str,
    answer: str,
    evaluation: str,
):
    prompt = f"""
You are an interviewer reacting to a candidate's answer.

Interviewer style: {personality}
Round: {round_name}

Candidate answer:
{answer}

Evaluation label: {evaluation}

STRICT RULES (VIOLATION = INVALID):
- DO NOT ask questions
- DO NOT instruct the candidate
- DO NOT prompt for answers
- DO NOT say "please explain", "go ahead", "can you"
- DO NOT change roles
- Reaction only
- One short sentence only
- End with a period

Examples of VALID reactions:
- "Good explanation."
- "That shows clear understanding."
- "This answer lacks sufficient depth."

Return ONLY the reaction text.
"""

    try:
        response = llm.invoke_text(prompt).strip()

        # 🔒 ABSOLUTE SAFETY FILTER
        response = re.sub(
            r"(please|can you|could you|go ahead|share|provide|explain).*",
            "",
            response,
            flags=re.IGNORECASE,
        ).strip()

        # remove trailing questions if any
        response = response.split("?")[0].strip()

        return response if response else "Okay."
    except Exception:
        return "Okay."



#initialize and return the LLM interface
def get_llm():
    from .llm import LLM
    return LLM()

# Convert internal round name to display title
def _round_title(r: str) -> str:
    return {"hr": "HR Round", "technical": "Technical Round", "behavioral": "Behavioral Round"}.get(r, r.title())

# Generate role-specific behavioral questions using LLM
def build_role_based_behavioral_questions(llm, role, experience, personality, count=3):
    """
    Dynamically generate ROLE-SPECIFIC behavioral questions using LLM.
    """

    prompt = f"""
    You are a {personality} interviewer.

    Candidate Role: {role}
    Experience Level: {experience}

    Generate {count} ROLE-SPECIFIC BEHAVIORAL interview questions.

    PERSONALITY GUIDELINES:
    - Friendly Coach → growth mindset, learning, reflection
    - Professor → reasoning, principles, ethics, theory
    - Startup Hiring Manager → ownership, ambiguity, speed, impact
    - Strict FAANG → conflict, trade-offs, pressure, decision quality

    STRICT RULES:
    - Output ONLY valid JSON
    - Do NOT include explanations
    - Do NOT include markdown
    - Do NOT include extra text
    - JSON must start with '[' and end with ']'

    JSON FORMAT:
    [
    {{
        "question": "string",
        "topic": "Behavioral",
        "qtype": "role-based"
    }}
    ]
    """

    try:
        raw = llm.invoke_text(prompt)
        print("LLM RAW BEHAVIORAL OUTPUT:", raw)  # DEBUG
        questions = safe_json_array(raw)


        if isinstance(questions, list):
            return questions
    except Exception as e:
        print("Behavioral LLM failed:", e)


    # 🔒 HARD SAFE FALLBACK — NEVER CRASH INTERVIEW
    return [
        {
            "question": "Can you describe a challenging situation you faced at work and how you handled it?",
            "topic": "Behavioral",
            "qtype": "role-based",
        },
        {
            "question": "Tell me about a time you had to work under pressure and meet a tight deadline.",
            "topic": "Behavioral",
            "qtype": "role-based",
        },
    ]


# Safely extract JSON array from LLM output
def safe_json_array(text: str):
    """
    Safely extract the first JSON array from LLM output.
    Prevents fallback questions from being used incorrectly.
    """
    match = re.search(r"\[\s*{.*?}\s*\]", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found")
    return json.loads(match.group())


# Dynamically select and generate the next technical question
# Based on performance, resume, difficulty, and personality
def generate_next_technical_question(state, llm):
    """
    Decide NEXT technical question dynamically
    based on resume, role, performance, personality
    """

    personality = state.candidate.personality
    weak = state.weak_streak
    good = state.good_streak

    # ⛔ Stop early if weak
    # Require at least 3 technical questions before stopping
    if weak >= PERSONALITIES[personality]["weak_limit"] and state.index >= 3:
        return None


    # 🎯 Decide question type
    if state.index == 0:
        qtype = "resume"
    elif good >= 2:
        qtype = "coding"
    elif weak == 1:
        qtype = "skill"
    else:
        qtype = "system"

    personality = state.candidate.personality
    min_d, max_d = PERSONALITY_DIFFICULTY_BOUNDS[personality]

    safe_level = max(min(state.difficulty_level, max_d), min_d)

    return build_technical_question(
        llm=llm,
        user_id=state.candidate.user_id,
        role=state.candidate.role,
        resume_text=state.candidate.resume_text,
        experience=state.candidate.experience_level,
        run_nonce=uuid.uuid4().hex[:6],
        personality=personality,
        difficulty_level=safe_level,   # 🔒 CLAMPED
        focus=qtype,
    )




# Entry point to generate questions for each interview round
def _get_round_questions(llm, state, round_name):
    """
    Entry point for each round.
    HR → fixed + dynamic HR
    Technical → handled dynamically per question
    Behavioral → batch generated
    """

    # ---------------- HR ROUND ----------------
    if round_name == "hr":
        questions = [{
            "question": "Tell me about yourself.",
            "topic": "HR",
            "qtype": "hr",
        }]

        # HR questions are generated dynamically during the round
        return questions


    # ---------------- TECHNICAL ROUND ----------------
    if round_name == "technical":
        # Start with ONE technical question only
        q = build_technical_question(
            llm=llm,
            user_id=state.candidate.user_id,
            role=state.candidate.role,
            resume_text=state.candidate.resume_text,
            experience=state.candidate.experience_level,
            run_nonce=uuid.uuid4().hex[:6],
            personality=state.candidate.personality,
        )

        q.setdefault("topic", "Technical")
        q.setdefault("qtype", "theory")

        return [q]

    # ---------------- BEHAVIORAL ROUND ----------------
    if round_name == "behavioral":
        return build_role_based_behavioral_questions(
            llm=llm,
            role=state.candidate.role,
            experience=state.candidate.experience_level,
            personality=state.candidate.personality,
            count=6,
        )

    return []


# Push the current question into conversation history
# Marks state as awaiting user answer
def ask_question(state: InterviewState) -> InterviewState:
    if state.index >= len(state.questions):
        return state  # control handled in interview_step


    q = state.questions[state.index]
    qtype_label = q.get("qtype", "theory")

    # Normalize labels
    if qtype_label == "coding":
        bracket = "coding"
    elif qtype_label == "role-based":
        bracket = "role-based"
    else:
        bracket = "theory"

    state.history.append(
        Turn(
            speaker="agent",
            text=(
                f"**Q{state.index + 1} — {q.get('topic','General')} "
                f"({bracket}):**\n\n"
                f"{q['question']}\n\n"
            ),
        )
    )

    state.awaiting_answer = True
    return state

# Safely store an answer record with round metadata
def _safe_add_record(state: InterviewState, q: dict, answer: str, quality: str, hints_used: int, round_name: str):
    payload = dict(
        question=q.get("question", ""),
        answer=answer,
        topic=q.get("topic", "General"),
        qtype=q.get("qtype", "theory"),
        quality=quality,
        hints_used=1 if q.get("_hint_given") else 0,

    )
    # Your AnswerRecord has round_name, so include it safely
    payload["round_name"] = round_name
    state.records.append(AnswerRecord(**payload))

# Retrieve all records belonging to a specific round
def _round_records(state: InterviewState, round_name: str) -> List[AnswerRecord]:
    return [r for r in state.records if getattr(r, "round_name", None) == round_name]



# Returns: correct | partial | alternative | irrelevant
def judge_answer(llm, question: str, answer: str, personality: str) -> str:
    """
    Returns: correct | partial | alternative | irrelevant
    Personality affects strictness.
    """

    answer = (answer or "").strip()
    lower = answer.lower()
    words = answer.split()


    # ✅ QUICK ACCEPT: short but valid code answers
    if answer:
        # code-like answers for coding / SQL questions
        if re.search(r"(function|def|return|select\s+.*from|class\s+)", answer, re.I):
            if re.search(r"(code|function|implement|write|sql|query)", question.lower()):
                return "correct"


    # 🚫 Reject if answer is basically the question itself
    q_norm = question.lower().strip()
    a_norm = answer.lower().strip()

    

    # ---------- HARD FILTER ----------
    # Too short → definitely not answering
    

    strict = personality == "Strict FAANG"

    # ---------- LLM JUDGMENT ----------
    prompt = f"""
You are a technical interviewer.

Interviewer style: {personality}

Question:
{question}

Candidate Answer:
{answer}

Classify the answer into EXACTLY ONE category:

- correct: clearly answers the question with correct reasoning
- partial: related but missing clarity, depth, or precision
- alternative: correct but explains a different valid approach
- irrelevant: does not answer the question
-If the question is a coding or SQL question and the answer is not code,
classify it as "irrelevant".

Be {"VERY STRICT" if strict else "REASONABLE"}.

Return ONLY valid JSON:
{{"label":"correct"}}
"""

    try:
        raw = llm.invoke_text(prompt)
        data = json.loads(raw.strip())
        label = data.get("label", "").lower()
        if label in {"correct", "partial", "alternative", "irrelevant"}:
            return label
    except Exception:
        pass  # fall back to heuristics

    return "partial"


# Apply interviewer personality rules to final answer label
def apply_personality_rules(state: InterviewState, q: dict, answer: str, label: str, personality: str) -> str:
    # ❌ STRICT FAANG — ZERO TOLERANCE
    if personality == "Strict FAANG":
        if label != "correct":
            state.final_verdict = "NOT SELECTED"
            return "force_reject"
        return label

    # 🧠 PROFESSOR — needs theory & definition
    if personality == "Professor":
        if label == "partial":
            return label

        return label

    # 🚀 STARTUP — needs impact & metrics
    if personality == "Startup Hiring Manager":
        if label == "partial":
            return label

            
        return label

    # 🙂 FRIENDLY COACH — gentle follow-up
    if personality == "Friendly Coach":
        if label == "partial":
            return label            
        return label

    return label


# Finalize a round, compute score, and decide next transition
def finalize_round(state: InterviewState, round_name: str) -> InterviewState:
    recs = _round_records(state, round_name)
    score = score_round(recs, round_name, state.whiteboard_score)
    state.round_scores[round_name] = score

    pass_mark = PASS_MARK.get(round_name, 5.0)
    passed = score >= pass_mark

    # ---------- HR ----------
    if round_name == "hr":
        state.history.append(
            Turn(
                speaker="agent",
                text=f"✅ **HR Round completed** (Score: {score}/10). Moving to Technical Round."
            )
        )
        return state

    # ---------- TECHNICAL ----------
    if round_name == "technical":
        if not passed:
            state.history.append(
                Turn(
                    speaker="agent",
                    text=(
                        f"❌ **Technical Round NOT PASSED** (Score: {score}/10).\n\n"
                        "We won’t proceed to the Behavioral round."
                    )
                )
            )
            state.final_verdict = "NOT SELECTED"
            return state   # ✅ JUST RETURN STATE

        state.history.append(
            Turn(
                speaker="agent",
                text=f"✅ **Technical Round PASSED** (Score: {score}/10). Moving to Behavioral Round."
            )
        )

        # 🔥 CRITICAL FIX
        return go_to_round(state, get_llm(), "behavioral")


    # ---------- BEHAVIORAL ----------
    if round_name == "behavioral":
        state.history.append(
            Turn(
                speaker="agent",
                text=f"✅ **Behavioral Round completed** (Score: {score}/10)."
            )
        )
        return state

    return state

# Generate final interview evaluation paragraph using LLM
def generate_final_evaluation_paragraph(llm, state: InterviewState) -> str:
    """
    Generates a paragraph-style final evaluation summary
    to be appended to the existing interview report.
    """

    evidence = []
    for r in state.records:
        evidence.append(
            f"{r.round_name.upper()} | Topic: {r.topic} | Quality: {r.quality}"
        )

    prompt = f"""
You are an interviewer summarizing interview performance.

The interview has EXACTLY THREE rounds:
- HR
- Technical
- Behavioral

Evidence (DO NOT invent rounds or numbers):
{chr(10).join(evidence)}

Rules:
- Refer ONLY to HR, Technical, and Behavioral rounds
- NEVER invent round numbers
- Do NOT mention question numbers
- Do NOT mention scores
- Write 2 short paragraphs
- Neutral, professional tone

Return ONLY the evaluation text.
"""

    try:
        return llm.invoke_text(prompt).strip()
    except Exception:
        return (
            "The candidate demonstrated clear communication and a positive attitude "
            "during the interview. Strong engagement was shown in familiar topics.\n\n"
            "However, some responses lacked depth and concrete examples, especially in "
            "technical and behavioral scenarios. Improving structure and specificity "
            "would strengthen future performance."
        )

# Finalize interview verdict and move to closing phase
def finalize_interview(state: InterviewState) -> InterviewState:
    # 🔒 ABSOLUTE GUARD — prevent duplicate summaries
    if getattr(state, "interview_done", False):
        return state
    state.summary_emitted = True

    rs = state.round_scores or {}


    # ❌ Technical failed → auto reject
    if rs.get("technical", 0) < PASS_MARK["technical"]:
        verdict = "NOT SELECTED"
        overall = rs.get("technical", 0)

    # ✅ Technical passed, Behavioral done
    else:
        # ✅ Tech passed, but do NOT finalize until Behavioral score exists
        tech = rs.get("technical", 0)

        if "behavioral" not in rs:
            return state  # <-- CRITICAL: prevents premature summary/verdict

        beh = rs.get("behavioral", 0)
        overall = round((tech + beh) / 2, 1)
        verdict = "SELECTED" if overall >= 6.5 else "NOT SELECTED"

    state.final_verdict = verdict

    # ✅ Generate Final Evaluation paragraph
    llm = get_llm()
    state.final_evaluation = generate_final_evaluation_paragraph(llm, state)

    lines = ["🎯 **Interview Summary**"]
    for r in ROUNDS:
        if r in rs:
            lines.append(f"- {_round_title(r)}: **{rs[r]}/10**")

    lines.append(f"\n⭐ Overall: **{overall}/10**")
    lines.append(f"\n🏁 Final Result: **{verdict}**")

    state.history.append(Turn(speaker="agent", text="\n".join(lines)))

    



    # ❌ DO NOT finish interview yet
    state.phase = "closing"          # 🔥 IMPORTANT
    #state.interview_done = False     # 🔥 IMPORTANT

    # ✅ ENABLE typing
    state.awaiting_answer = True
    state.pending_user_input = False
    state.last_user_answer = None

    state.history.append(
        Turn(
            speaker="agent",
            text="That brings us to the end of the interview. Do you have any questions?"
        )
    )
    return state


# Transition state to a new interview round

def go_to_round(state: InterviewState, llm, round_name: str) -> InterviewState:
    state.current_round = round_name
    state.questions = _get_round_questions(llm, state, round_name)
    state.index = 0
    state.hints_used = 0
    state.depth_level = 0
    state.good_streak = 0
    state.weak_streak = 0
    state.awaiting_answer = False   # 🔥 IMPORTANT

    state.history.append(
        Turn(speaker="agent", text=f"🟢 **{_round_title(round_name)}**")
    )

    if not state.questions:
        return state

    return ask_question(state)  # 🔥 THIS WAS MISSING


    
# --------------------------------------------------
# Main Interview Logic
# Handles greeting, questioning, evaluation, hints, retries, and transitions
# --------------------------------------------------

def interview_step(state: InterviewState) -> InterviewState:
    llm = get_llm()


    # 🔒 GLOBAL HARD STOP — NEVER CONTINUE AFTER TECH FAIL
    stop = hard_stop_if_rejected(state)
    if stop:
        return stop


    # 🔒 HARD STOP — NEVER CONTINUE AFTER FAIL
    if getattr(state, "final_verdict", None) == "NOT SELECTED":
        return state   # ❌ DO NOT CALL finalize_interview AGAIN



    personality = getattr(state.candidate, "personality", "Friendly Coach")
    profile = PERSONALITIES.get(personality, PERSONALITIES["Friendly Coach"])
    interviewer = INTERVIEWER_PROFILES.get(personality, {"name": "Alex", "intro": "your interviewer"})

    if state.phase == "greeting":
        # 🔄 NEW interview = new entropy
        state.run_nonce = uuid.uuid4().hex[:8]
        state.asked_hr_questions = set()

        state.history.append(
            Turn(
                speaker="agent",
                text=(
                    f"{time_based_greeting()}, {state.candidate.name} 👋\n\n"
                    f"I’m **{interviewer['name']}**, {interviewer['intro']}.\n\n"
                    f"I’ll interview you in a **{personality}** style.\n\n"
                    "Let’s start."
                ),
            )
        )

        # ✅ Decide HR target ONCE (between min & max)
        strategy = PERSONALITY_STRATEGY[state.candidate.personality]
        state.hr_target = random.randint(
            strategy["hr_min"],
            strategy["hr_max"]
        )

        state.phase = "questioning"

        state.difficulty_level = PERSONALITY_START_LEVEL[personality]
        state.last_difficulty_change = None

        return go_to_round(state, llm, "hr")


    if getattr(state, "round_scores", None) is None:
        state.round_scores = {}

    if not getattr(state, "current_round", None):
        state.current_round = "hr"

    

    # ---------- WAIT ----------
    # If we asked a question and user hasn't responded yet, do nothing.
    if state.awaiting_answer and not state.pending_user_input:
        return state


    # ---------- INPUT ----------
    answer = (state.last_user_answer or "").strip()


    # q not available yet → detect intent without question context
    intent_data = llm_detect_intent(llm, "", answer)
    kind = intent_data["intent"]


    if not answer:
        state.awaiting_answer = True
        return state

    # ✅ SAFE TO ACCESS q NOW
    if not state.questions or state.index >= len(state.questions):
        state = finalize_round(state, state.current_round)
        return finalize_interview(state)


    q = state.questions[state.index]



    # ==================================================
    # 🔥 HARD BLOCK: REUSED ANSWER FOR A NEW QUESTION
    # ==================================================
    if repeated_answer_anywhere(llm, state, q.get("question", ""), answer):
        state.history.append(
            Turn(
                speaker="agent",
                text=(
                    "❌ This looks like the same answer you used for a previous question. "
                    "Please answer this question with a NEW example or NEW reasoning."
                )
            )
        )
        state.awaiting_answer = True
        return state

    state.pending_user_input = False
    state.last_user_answer = None


    if state.index >= len(state.questions):

        state = finalize_round(state, state.current_round)

        # 🔒 ABSOLUTE TECH GATE
        if state.current_round == "technical":
            if state.final_verdict == "NOT SELECTED":
                return state
            return go_to_round(state, llm, "behavioral")

        # HR → Technical
        if state.current_round == "hr":
            return go_to_round(state, llm, "technical")

        return finalize_interview(state)



    # ==================================================
    # 🔁 SAME CONCEPT — ONE-TIME APPLY QUESTION (ENTIRE INTERVIEW)
    # ==================================================
    if state.current_round == "technical" and not state.concept_reminder_used:
        prev_concept = find_similar_concept_record(
            state,
            q.get("question", "")
        )

        if prev_concept:
            current_label = judge_answer(
                llm,
                q.get("question", ""),
                answer,
                personality
            )

            if current_label in {"partial", "irrelevant"} or kind == "dont_know":

                state.concept_reminder_used = True  # 🔒 one-time only

                # 1️⃣ Explain the intent to the candidate
                state.history.append(
                    Turn(
                        speaker="agent",
                        text=(
                            "You explained a similar concept earlier quite well. "
                            "Let’s apply that idea to a concrete scenario."
                        )
                    )
                )

                # 2️⃣ Generate a NEW apply-style question
                followup = build_followup_question(
                    llm=llm,
                    role=state.candidate.role,
                    resume_text=state.candidate.resume_text,
                    prev_question=prev_concept.question,
                    prev_answer=prev_concept.answer,
                    run_nonce=state.run_nonce,
                    personality=personality,
                )

                # 3️⃣ Ask the new question immediately
                if followup:
                    state.questions.insert(state.index + 1, followup)
                    state.index += 1
                    state.awaiting_answer = False
                    return ask_question(state)

                # 4️⃣ Safety fallback (should rarely happen)
                state.awaiting_answer = True
                return state




    override = quick_intent_override(answer)
    if override:
        kind = override
    else:
        intent_data = llm_detect_intent(llm, q.get("question", ""), answer)
        kind = intent_data["intent"]

    

      




        # ==================================================
        # 🟦 CLARIFICATION HANDLING (CRITICAL FIX)
        # ==================================================
        if kind == "clarification":
            # Respond with clarification WITHOUT judging
            state.history.append(
                Turn(
                    speaker="agent",
                    text=(
                        "You can make reasonable assumptions. "
                        "The platform supports users, courses, enrollments, progress tracking, "
                        "and efficient querying for analytics."
                    )
                )
            )

            # Re-ask the SAME question
            state.awaiting_answer = True
            state.pending_user_input = False
            state.last_user_answer = None

            # ❗ DO NOT increment attempts
            # ❗ DO NOT judge
            # ❗ DO NOT move index
            return state


        # ---------- USER SKIPS QUESTION ----------
        if kind == "skip":
            state.history.append(
                Turn(speaker="agent", text="Okay, let’s move on.")
            )

            if state.current_round != "hr":
                _safe_add_record(
                    state,
                    q,
                    answer,
                    "weak",
                    state.hints_used,
                    state.current_round,
                )

            
            state.index += 1
            state.awaiting_answer = False

            return ask_question(state)


    
        else:
            intent_data = llm_detect_intent(llm, q.get("question", ""), answer)
            kind = intent_data["intent"]
            

    # 🚫 HARD BLOCK — SAME ANSWER REPEATED (SAME QUESTION)
    prev_answer = q.get("_last_answer")
    if prev_answer and _norm(prev_answer) == _norm(answer):
        state.history.append(
            Turn(
                speaker="agent",
                text=(
                    "⚠️ This is the same answer as before.\n"
                    "Please add a new explanation, example, metric, or reasoning."
                )
            )
        )
        state.awaiting_answer = True
        return state


    # ==================================================
    # HR ROUND HANDLING (MUST BE FIRST)
    # ==================================================
    if state.current_round == "hr":
        strategy = PERSONALITY_STRATEGY[state.candidate.personality]

        

        retries = int(q.get("_retry", 0))

        # basic HR quality
        label = judge_answer(llm, q.get("question", ""), answer, personality)
        quality = "good" if label in {"correct","alternative"} else "weak"
        # 🔁 HR streak tracking
        if quality == "weak":
            state.weak_streak += 1
            state.good_streak = 0
        else:
            state.good_streak += 1
            state.weak_streak = 0



        if quality == "good" or retries >= 1:
            _safe_add_record(state, q, answer, quality, 0, "hr")

        if quality == "weak" and retries < 1:
            q["_retry"] = retries + 1
            state.history.append(
                Turn(
                    speaker="agent",
                    text="⚠️ Could you expand a bit more on that?"
                )
            )
            state.awaiting_answer = True
            return state

        # HR FOLLOW-UP (only once, after retry)
        if quality == "weak" and retries >= 1:
            followup = build_hr_followup(
                llm,
                q.get("question", ""),
                answer,
                state.candidate.personality,
                state.run_nonce,
            )

            if followup:
                state.questions.insert(
                    state.index + 1,
                    {
                        "question": followup,
                        "topic": "HR",
                        "qtype": "hr-followup",
                    }
                )
                state.index += 1
                state.awaiting_answer = True
                return ask_question(state)




        # accept HR answer and MOVE ON
        reaction = llm_generate_reaction(
            llm=llm,
            personality=personality,
            round_name="hr",
            question=q.get("question", ""),
            answer=answer,
            evaluation=quality,
        )

        state.history.append(Turn(speaker="agent", text=reaction))
        state.awaiting_answer = False
        state.pending_user_input = False

          
        # ✅ Decide HR target ONCE (between min & max)
        if not hasattr(state, "hr_target"):
            strategy = PERSONALITY_STRATEGY[state.candidate.personality]
            state.hr_target = random.randint(
                strategy["hr_min"],
                strategy["hr_max"]
            )

        # ✅ Generate remaining HR questions ONCE
        if not getattr(state, "_hr_generated", False):
            hr_questions = build_hr_questions(
                llm,
                state,
                count=(state.hr_target or strategy["hr_max"]) - 1

            )
            state.questions.extend(hr_questions)
            state._hr_generated = True


        strategy = PERSONALITY_STRATEGY[state.candidate.personality]
        

        answered_hr = len([
            r for r in state.records
            if r.round_name == "hr"
        ])

        # 🟢 keep asking until target reached
        if answered_hr < state.hr_target:
            state.index += 1
            return ask_question(state)

        # 🟡 target reached → move on
        state = finalize_round(state, "hr")
        return go_to_round(state, llm, "technical")

        # otherwise continue
        state.index += 1
        return ask_question(state)

   
    # ---------- NEED TIME ----------
    # Only treat as needs_time if user did NOT ask for help/hint
    if kind == "needs_time" and not re.search(r"\b(help|hint|guide)\b", answer.lower()):
        state.history.append(
            Turn(
                speaker="agent",
                text=profile["msg_need_time"]
            )
        )
        state.awaiting_answer = True
        return state


    # ---------- USER REQUESTED HINT ----------
    if kind == "hint":
        limit = HINT_LIMITS.get(personality, 1)


        if state.hints_used >= limit:
            state.history.append(
                Turn(
                    speaker="agent",
                    text="Let’s move on to the next question."
                )
            )

            state.hints_used = 0
            state.awaiting_answer = False
            state.pending_user_input = False
            state.last_user_answer = None

            # 🔥 CRITICAL FIX — generate next question if needed
            if state.current_round == "technical":
                next_q = generate_next_technical_question(state, llm)

                if next_q:
                    state.questions.append(next_q)
                    state.index = len(state.questions) - 1
                    return ask_question(state)
                else:
                    state = finalize_round(state, "technical")
                    return finalize_interview(state)

            # Non-technical rounds
            state.index += 1
            return ask_question(state)


        hint = generate_dynamic_hint(
            llm=llm,
            question=q.get("question", ""),
            answer=answer,
            topic=q.get("topic", ""),
            personality=personality
        )
        state.user_hints_used += 1  
        state.hints_used += 1
        q["_hint_given"] = True

        state.history.append(
            Turn(speaker="agent", text=f"💡 Hint: {hint}")
        )

        state.awaiting_answer = True
        return state

    # ---------- DONT KNOW (QUESTION-AWARE, move-on) ----------
    if kind == "dont_know":

        # Track dont-know count per question
        dk_count = int(q.get("_dont_know", 0)) + 1
        q["_dont_know"] = dk_count

        

        # ---------- SECOND DONT KNOW → MOVE ON ----------
        state.history.append(
            Turn(
                speaker="agent",
                text="Okay, let’s move on."
            )
        )

        # Record weak answer
        if state.current_round != "hr":
            _safe_add_record(
                state,
                q,
                answer,
                "weak",
                state.hints_used,
                state.current_round,
            )

        state.weak_streak += 1
        

        # 🔒 Reset streaks to avoid difficulty bouncing
        state.good_streak = 0
        state.weak_streak = max(state.weak_streak, 1)


        # 🔽 DOWNGRADE DIFFICULTY AFTER STRUGGLE
        if state.current_round == "technical":
            adjust_difficulty(state, "down")

        # 🔒 FORCE DIFFICULTY FLOOR (prevents oscillation)
        personality = state.candidate.personality
        min_d, _ = PERSONALITY_DIFFICULTY_BOUNDS[personality]
        state.difficulty_level = max(state.difficulty_level, min_d)


        # 🔑 MOVE INDEX
        state.index += 1

        # 🔑 RESET STREAMLIT STATE (THIS WAS MISSING)
        state.awaiting_answer = False
        state.pending_user_input = False
        state.last_user_answer = None

        # ⛔ END ROUND IF WEAK LIMIT HIT
        if state.weak_streak >= profile["weak_limit"]:
            state = finalize_round(state, state.current_round)

            # 🔒 HARD TECHNICAL GATE (NO BEHAVIORAL AFTER FAIL)
            if state.current_round == "technical":
                return finalize_interview(state)

            current_idx = ROUNDS.index(state.current_round)
            if current_idx + 1 < len(ROUNDS):
                return go_to_round(state, llm, ROUNDS[current_idx + 1])

            return finalize_interview(state)


        # ==================================================
        # 🔥 CRITICAL FIX — TECH ROUND NEEDS DYNAMIC QUESTION
        # ==================================================
        if state.current_round == "technical":

            next_q = generate_next_technical_question(state, llm)

            # ⛔ stop tech round
            if not next_q:
                state = finalize_round(state, "technical")

                # 🔒 BLOCK behavioral if failed
                if state.final_verdict == "NOT SELECTED":
                    return finalize_interview(state)

                return go_to_round(state, llm, "behavioral")

            state.questions.append(next_q)
            return ask_question(state)

        # ---------- NON-TECH ROUNDS ----------
        return ask_question(state)


   # ---------- TECH / BEHAVIORAL ----------
    label = judge_answer(
        llm,
        q.get("question", ""),
        answer,
        personality
    )

    label = apply_personality_rules(state, q, answer, label, personality)

    # ==================================================
    # ✅ FINAL ANSWER DECISION — STRICT 2 ATTEMPT RULE
    # ==================================================

    # Track attempts per question
    q["_attempts"] = int(q.get("_attempts", 0)) + 1
    attempts = q["_attempts"]

    # Always show reaction
    reaction = llm_generate_reaction(
        llm=llm,
        personality=personality,
        round_name=state.current_round,
        question=q.get("question", ""),
        answer=answer,
        evaluation=label,
    )
    state.history.append(Turn(speaker="agent", text=reaction))

    # ---------- CASE 1: CORRECT → MOVE ----------
    if label in {"correct", "alternative","partial"}:
        _safe_add_record(
            state,
            q,
            answer,
            "good",
            state.hints_used,
            state.current_round,
        )

        # TECH ROUND MUST GENERATE NEXT QUESTION
        if state.current_round == "technical":
            next_q = generate_next_technical_question(state, llm)

            if next_q:
                state.questions.append(next_q)
                state.index = len(state.questions) - 1
                state.awaiting_answer = False
                return ask_question(state)

            # No more tech questions → end round
            state = finalize_round(state, "technical")
            return finalize_interview(state)

        # Non-technical rounds
        state.index += 1
        state.awaiting_answer = False
        return ask_question(state)


    # ---------- CASE 2: FIRST WRONG → HINT + RETRY ----------
    if attempts == 1:
        # SAVE last answer ONLY AFTER first failed attempt
        q["_last_answer"] = answer

        hint = generate_dynamic_hint(
            llm=llm,
            question=q.get("question", ""),
            answer=answer,
            topic=q.get("topic", ""),
            personality=personality,
        )

        state.history.append(
            Turn(
                speaker="agent",
                text=f"❌ Wrong answer.\n💡 Hint: {hint}\n\nTry once more."
            )
        )

        state.awaiting_answer = True
        return state

    # ---------- CASE 3: SECOND WRONG → MOVE ----------
    _safe_add_record(
        state,
        q,
        answer,
        "weak",
        state.hints_used,
        state.current_round,
    )

    state.history.append(
        Turn(
            speaker="agent",
            text="❌ Still not correct. Let’s move to the next question."
        )
    )

    state.index += 1
    state.awaiting_answer = False
    return ask_question(state)


    _safe_add_record(
        state,
        q,
        answer,
        quality,
        state.hints_used,
        state.current_round,
    )

    # If the final accepted result is weak, reduce difficulty for next question
    if state.current_round == "technical" and quality == "weak":
        adjust_difficulty(state, "down")


    # Whiteboard bonus (correct place)
    if state.pending_whiteboard_notes:
        state.whiteboard_score += 1
        state.pending_whiteboard_notes = None

    

    # streak logic
    if quality == "weak":
        state.weak_streak += 1
        state.good_streak = 0
    else:
        state.good_streak += 1
        state.weak_streak = 0

    
    # ==================================================
    # 🧠 BEHAVIORAL ROUND CONTROL (MIN / MAX ENFORCEMENT)
    # ==================================================
    if state.current_round == "behavioral":
        strategy = PERSONALITY_STRATEGY[state.candidate.personality]

        # ❌ Too many weak answers → stop behavioral
        if state.weak_streak >= PERSONALITIES[state.candidate.personality]["weak_limit"]:
            state = finalize_round(state, "behavioral")
            return finalize_interview(state)

        # 🟢 Enforce minimum behavioral questions
        if state.index + 1 < strategy["behavioral_min"]:
            state.index += 1
            return ask_question(state)

        # 🟡 Stop after max behavioral questions
        if state.index + 1 >= strategy["behavioral_max"]:
            state = finalize_round(state, "behavioral")
            return finalize_interview(state)
   


    return state


    

# --------------------------------------------------
# Closing Phase (Q&A works)
# Handles post-interview Q&A and polite termination
# --------------------------------------------------

def closing_step(state: InterviewState) -> InterviewState:
    
    # ⛔ ABSOLUTE STOP — interview fully finished
    if getattr(state, "interview_done", False) and state.phase == "done":
        return state

    llm = get_llm()


    # 🔒 GLOBAL HARD STOP — NEVER CONTINUE AFTER TECH FAIL
    stop = hard_stop_if_rejected(state)
    if stop:
        return stop


    # Wait until user types something
    if state.awaiting_answer and not state.pending_user_input:
        return state

    user_msg = _norm(state.last_user_answer)
    state.pending_user_input = False
    state.last_user_answer = None

    # Polite acknowledgements → close politely
    if user_msg in {"thank you", "thanks", "thankyou", "ok", "okay", "cool"}:
        state.interview_done = True
        state.phase = "done"
        state.awaiting_answer = False
        state.pending_user_input = False
        state.last_user_answer = None

        

        state.history.append(
            Turn(
                speaker="agent",
                text="You’re welcome 🙂 All the best for your future!"
            )
        )
        return state


    if user_msg in END_PHRASES:
        state.interview_done = True
        state.phase = "done"
        state.awaiting_answer = False
        state.pending_user_input = False

        state.history.append(
            Turn(
                speaker="agent",
                text="Thank you for your time. All the best! 🚀"
            )
        )
        return state


    # Answer user question and allow more questions
    reply = llm.invoke_text(f"Answer briefly and clearly as an interviewer:\n{user_msg}")
    state.history.append(Turn(speaker="agent", text=reply))
    state.history.append(Turn(speaker="agent", text="Any other questions?"))
    state.awaiting_answer = True
    return state

# --------------------------------------------------
# Router + Graph Builder
# Route execution between interview and closing phases
# --------------------------------------------------

def router(state: InterviewState) -> InterviewState:
    # 🔒 HARD STOP — NOTHING after interview is done
    if getattr(state, "interview_done", False) and state.phase == "done":
        return state

    if state.phase == "closing":
        return closing_step(state)

    return interview_step(state)

# Build and compile the LangGraph interview workflow
def build_graph():
    g = StateGraph(InterviewState)
    g.add_node("interview", router)
    g.set_entry_point("interview")
    g.add_edge("interview", END)
    return g.compile()


