# AI Interviewer – Adaptive LLM-Driven Interview System



---

## 1. Project Overview

This project implements an **AI-powered interviewer** that conducts a complete,
multi-round interview (HR, Technical, Behavioral) using Large Language Models (LLMs).

The system dynamically:
- Generates role-specific interview questions
- Evaluates candidate answers across multiple dimensions
- Adapts technical difficulty based on performance
- Enforces strict technical gating
- Produces structured feedback and personalized learning plans

The objective is to build a **reproducible, explainable, and scalable interview system**
that mimics real-world technical interviews.

---

## 2. Research Question

Can a state-driven, LLM-based interviewer reliably evaluate candidates across HR,
Technical, and Behavioral rounds while adapting difficulty and maintaining
reproducibility?

---

## 3. Motivation and Relevance

Traditional interviews often suffer from:
- Inconsistent evaluation criteria
- Interviewer bias
- Poor reproducibility
- Limited feedback for candidates

This system addresses these issues by:
- Using a deterministic state machine (LangGraph)
- Separating question generation, evaluation, and scoring
- Enforcing hard technical gates
- Providing transparent scoring and feedback

This approach is relevant for:
- Mock interview platforms
- Automated hiring pipelines
- Interview preparation tools
- Skill assessment systems

---

## 4. System Architecture

The system is implemented as a **state-machine-driven workflow**.

### Core Components
- **Streamlit UI** – Candidate interaction and visualization
- **LangGraph Controller** – Interview flow and round transitions
- **LLM Layer** – Question generation, evaluation, feedback
- **RAG Layer (ChromaDB)** – Resume-grounded questioning
- **Whiteboard Vision Module** – Diagram evaluation
- **Evaluation Engine** – Scoring, verdict, and learning plan generation

High-level flow:

Candidate  
→ Streamlit UI  
→ LangGraph State Machine  
→ LLM + RAG  
→ Evaluation Engine  
→ Feedback + Learning Plan



## 5. Interview Flow

The interview consists of **exactly three rounds**:

### HR Round
- Assesses communication and situational awareness
- Dynamic HR follow-up questions
- Enforces minimum and maximum question limits

### Technical Round (Hard Gate)
- Resume-grounded and system-level questions
- Adaptive difficulty adjustment
- Hint limits based on interviewer personality
- Failure immediately stops the interview

### Behavioral Round
- Role-specific behavioral questions
- Strict weak-answer thresholds
- Final verdict determination

---

## 6. Models Used


Text generation & evaluation - llama-3.1-8b-instant 
Vision (whiteboard analysis) - meta-llama/llama-4-scout-17b-16e-instruct 
Text fallback - mistralai/Mistral-7B-Instruct-v0.2 
Embeddings - all-MiniLM-L6-v2 

---

## 7. Prompting Strategy

- JSON-only structured outputs
- Personality-aware strictness rules
- No chain-of-thought exposure
- Deterministic retries with safe fallbacks
- Explicit constraints to prevent hallucination

---

## 8. Evaluation Protocol

Each answer is evaluated across **five dimensions**:
- Technical Depth
- Correctness
- Clarity
- Communication
- Confidence

Scoring rules:
- Each dimension normalized to a 0–10 scale
- Technical round acts as a **hard gate**
- Final verdict depends on Technical and Behavioral rounds only
- HR round influences feedback but not selection

---

## 9. Key Results

- Adaptive difficulty prevents question oscillation
- Hard technical gating improves evaluation signal
- Resume-grounded questions increase relevance
- Whiteboard vision improves explanation quality
- Learning plans provide actionable improvement steps


---

## 10. Limitations and Ethical Considerations

- LLM nondeterminism cannot be fully eliminated
- Language bias may affect interpretation
- Not suitable for high-stakes hiring without human oversight
- Usage of AI tools must be disclosed to candidates

---


