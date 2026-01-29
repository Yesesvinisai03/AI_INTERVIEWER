# Reproducibility Statement

This document describes the assumptions, constraints, and sources of variability for the **AI Interviewer Agent** project. The objective is to allow another evaluator or engineer to reproduce the system behavior, results, and limitations as closely as possible.

---

## 1. Hardware Assumptions

This project is designed to run on **commodity hardware** and does not require a GPU.

### Minimum Recommended Hardware
- **CPU:** 4-core modern processor (Intel i5 / AMD Ryzen 5 or equivalent)
- **RAM:** 8 GB minimum (16 GB recommended)
- **Storage:** ~1 GB free disk space
- **GPU:** Not required

### Tested Environments
- Local machines (CPU-only)
- Cloud-hosted environments (Hugging Face Spaces, Streamlit Cloud)
- Operating systems:
  - Windows 10 / 11
  - macOS
  - Linux (Ubuntu-based)

All vector embedding operations are CPU-based, and all LLM inference is handled through external APIs.

---

## 2. Runtime Estimates

Execution time primarily depends on **network latency and LLM API response time**, rather than local compute.

### Typical Runtime Breakdown

| Component                       | Approximate Time |
|-------------------------------- |------------------|
| Application startup             | 3–5 seconds      |
| Resume parsing and indexing     | 2–4 seconds      |
| HR round (per question)         | 2–5 seconds      |
| Technical round (per question)  | 3–7 seconds      |
| Behavioral round (per question) | 2–5 seconds      |
| Whiteboard vision analysis      | 10–30 seconds    |
| Final evaluation generation     | 2–4 seconds      |

A complete interview session typically finishes within **5–10 minutes**, depending on the number of questions and optional whiteboard usage.

---

## 3. Random Seed Handling

This system does **not enforce a fixed global random seed**.

### Controlled Randomness
- Python-level randomness is used for:
  - Question selection
  - Interview flow variation
- Difficulty progression is bounded and state-driven.
-The goal is to simulate **real interview dynamics**, which naturally vary. Enforcing strict determinism would reduce realism and adaptability. As a result, exact question phrasing may vary across runs.

---

## 4. Known Sources of Nondeterminism

### Large Language Model Outputs
- Question generation, evaluation, and summarization rely on LLMs.
- LLM inference is probabilistic, even with identical prompts.
- Temperature values are intentionally non-zero to prevent repetitive interviews.

### External API Dependencies
- Text inference uses Groq-hosted LLMs.
- Vision-based whiteboard analysis uses OpenRouter-hosted vision models.
- A Hugging Face text-only model is used as a fallback.

API latency, availability, or model updates may introduce variability.

### Resume-Based Retrieval (RAG)
- Resume content is chunked and embedded using SentenceTransformer models.
- Retrieval depends on semantic similarity scores.
- Minor formatting differences in resumes can affect retrieved context.

### Interviewer Personality Effects
- Selected interviewer personality alters:
  - Question difficulty
  - Hint availability
  - Evaluation strictness

This behavior is intentional and by design.

---

## 5. Cost Considerations

This project relies on **pay-per-use external APIs**. No local model training or fine-tuning is performed.

### API Usage
- **Text LLMs (Groq):**
  - Question generation
  - Answer evaluation
  - Feedback summarization
- **Vision LLMs (OpenRouter):**
  - Whiteboard diagram analysis (optional)
- **Embeddings:**
  - Generated locally using CPU-based SentenceTransformer models

### Estimated Cost per Interview
- Text LLM usage: low (free-tier friendly)
- Vision model usage: moderate and optional
- Embedding computation: free (local)

The system is designed to be cost-efficient and scalable.

---

