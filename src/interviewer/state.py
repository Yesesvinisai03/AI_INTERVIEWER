from __future__ import annotations
from typing import List, Optional, Dict, Any, Set
from pydantic import BaseModel, Field


# --------------------------------------------------
# Candidate Info
# --------------------------------------------------

# Stores static candidate profile details
class CandidateInfo(BaseModel):
    name: str
    role: str
    experience_level: str
    resume_text: str = ""
    user_id: str = "default_user"
    personality: str = "Friendly Coach"


# --------------------------------------------------
# Conversation Turn
# Represents a single turn in the interview conversation
# --------------------------------------------------

class Turn(BaseModel):
    speaker: str   # "agent" | "candidate"
    text: str


# --------------------------------------------------
# Answer Record (Per Question)
# Stores evaluation details for a single interview question
# --------------------------------------------------

class AnswerRecord(BaseModel):
    question: str
    answer: str

    topic: str = "General"
    qtype: str = "theory"

    quality: str = "unknown"   # good | weak
    hints_used: int = 0

    # ✅ Round tracking
    round_name: str = "technical"

    # Whiteboard support
    whiteboard_used: bool = False
    whiteboard_notes: Optional[str] = None


# --------------------------------------------------
# Interview State (MAIN)
# Main interview state passed through the LangGraph workflow
# --------------------------------------------------

class InterviewState(BaseModel):
    # Candidate
    candidate: CandidateInfo


    # Conversation
    history: List[Turn] = Field(default_factory=list)

    # Interview questions
    questions: List[dict] = Field(default_factory=list)
    index: int = 0

    asked_hr_questions: Set[str] = Field(default_factory=set)

    hr_target: int | None = None


    # Flow control
    phase: str = "greeting"          # greeting | questioning | closing | done
    awaiting_answer: bool = False
    interview_done: bool = False
    pending_whiteboard_notes: Optional[str] = None
    # Input bridge (Streamlit)
    pending_user_input: bool = False
    last_user_answer: Optional[str] = None
    user_hints_used: int = 0   # ONLY when user types "hint"
    
    behavioral_done: bool = False
    awaiting_behavioral_answer: bool = False
    summary_emitted: bool = False
    concept_reminder_used: bool = False

    whiteboard_score: int = 0

    
    # Rounds
    current_round: str = "hr"        # hr → technical → behavioral
    round_scores: Dict[str, float] = Field(default_factory=dict)
    final_verdict: Optional[str] = None

    # Hint tracking
    hints_used: int = 0

    # Answer records
    records: List[AnswerRecord] = Field(default_factory=list)

    # Adaptive logic
    run_nonce: str = ""
    depth_level: int = 0
    good_streak: int = 0
    weak_streak: int = 0

    # Post interview
    feedback: Optional[Dict[str, Any]] = None
    learning_plan: Optional[Dict[str, Any]] = None

    final_evaluation: Optional[str] = None

    # Current difficulty level and last adjustment direction

    difficulty_level: int = 0
    last_difficulty_change: str | None = None

