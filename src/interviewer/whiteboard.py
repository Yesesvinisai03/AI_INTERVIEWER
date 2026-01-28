import os
import io
import base64
import numpy as np
from PIL import Image
import requests

# ----------------------------
# OpenRouter config
# ----------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_VISION_MODEL = "meta-llama/llama-3.2-11b-vision-instruct"

# ----------------------------
# HF Mistral fallback (text-only)
# ----------------------------
from huggingface_hub import InferenceClient
HF_CLIENT = InferenceClient(token=os.environ.get("HF_API_TOKEN"))

# ----------------------------
# Convert Streamlit canvas → PNG bytes
# ----------------------------
def _canvas_to_png_bytes(canvas_image: np.ndarray) -> bytes:
    """
    streamlit_drawable_canvas returns RGBA array.
    Convert safely to resized PNG bytes.
    """
    arr = np.array(canvas_image).astype("uint8")
    img = Image.fromarray(arr)
    img = img.resize((768, 768))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ----------------------------
# OpenRouter Vision Analysis
# ----------------------------
def _openrouter_vision_analyze(question: str, candidate_text: str, png_bytes: bytes) -> dict:
    """
    Uses OpenRouter vision model to:
    - Give brief diagram feedback
    - Ask exactly ONE follow-up question
    """

    img_b64 = base64.b64encode(png_bytes).decode("utf-8")
    image_data_url = f"data:image/png;base64,{img_b64}"

    # ---------- FEEDBACK PROMPT ----------
    feedback_prompt = f"""
You are a technical interviewer.

You are shown a candidate's whiteboard diagram.

Context:
Question asked:
"{question}"

Candidate explanation:
"{candidate_text}"

Task:
Briefly evaluate the diagram.

Rules:
- Mention ONE correctness issue, risk, or improvement.
- Reference visible elements (boxes, arrows, missing steps).
- Be concise (1–2 sentences).
- Do NOT ask a question.
""".strip()

    # ---------- FOLLOW-UP QUESTION PROMPT ----------
    question_prompt = f"""
You are a technical interviewer.

You are shown a candidate's whiteboard diagram.

Context:
Question asked:
"{question}"

Candidate explanation:
"{candidate_text}"

Task:
Ask EXACTLY ONE follow-up interview question
based on the diagram.

Rules:
- Reference visible elements from the drawing.
- Do NOT explain the answer.
- Output ONLY the question.
""".strip()

    def call_openrouter(prompt: str) -> str:
        payload = {
            "model": OPENROUTER_VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url}
                        }
                    ]
                }
            ],
            "temperature": 0.4,
            "max_tokens": 120
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Whiteboard Vision"
        }

        r = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=90
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    feedback = call_openrouter(feedback_prompt)
    followup = call_openrouter(question_prompt).split("\n")[0].strip()

    return {
        "feedback": feedback,
        "question": followup
    }

# ----------------------------
# Text-only fallback (HF Mistral)
# ----------------------------
def _mistral_text_fallback(question: str, candidate_text: str) -> dict:
    """
    Text-only fallback that matches the vision output format.
    """
    system_prompt = (
        "You are a technical interviewer. Ask concise, interviewer-style follow-up questions. "
        "Never provide answers."
    )

    user_prompt = f"""
Interview Question:
{question}

Candidate explanation:
{candidate_text}

Ask ONE follow-up interview question targeting
an unclear, risky, or interesting part.

Rules:
- Ask ONLY ONE question.
- Output ONLY the question.
""".strip()

    resp = HF_CLIENT.chat_completion(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=80,
        temperature=0.6,
    )

    question = resp.choices[0].message["content"].strip().split("\n")[0].strip()

    return {
        "feedback": "⚠️ Whiteboard vision unavailable. Feedback generated from explanation only.",
        "question": question
    }

# ----------------------------
# MAIN ENTRY (used by app.py)
# ----------------------------
def analyze_whiteboard(question: str, candidate_text: str, canvas_image):
    """
    Strategy:
    - Primary: OpenRouter Vision (API, open-source model)
    - Fallback: HF Mistral (text-only)
    """

    if canvas_image is None:
        fallback = _mistral_text_fallback(question, candidate_text)
        return {
            "feedback": "⚠️ No whiteboard image provided. Feedback based on explanation only.",
            "question": fallback["question"],
        }

    try:
        png_bytes = _canvas_to_png_bytes(canvas_image)
        return _openrouter_vision_analyze(question, candidate_text, png_bytes)

    except Exception:
        fallback = _mistral_text_fallback(question, candidate_text)
        return {
            "feedback": fallback["feedback"],
            "question": fallback["question"],
        }
