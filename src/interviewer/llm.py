from __future__ import annotations
import os
from typing import Optional
from groq import Groq


class LLM:
    def __init__(self, text_model: Optional[str] = None, vision_model: Optional[str] = None):
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not found. Put it in .env and restart Streamlit.")

        self.client = Groq(api_key=api_key)

        # Text model
        self.text_model = text_model or os.getenv("GROQ_TEXT_MODEL", "llama-3.1-8b-instant")

        # ✅ Updated supported vision models (Llama 4)
        self.vision_model = vision_model or os.getenv(
            "GROQ_VISION_MODEL",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        )

    def invoke_text(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.text_model,
            messages=[
                {"role": "system", "content": "You are a helpful technical interviewer and mentor."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,  # ✅ more variety so questions differ each run
        )
        return resp.choices[0].message.content.strip()

    def explain_topic(self, topic: str, context: str = "") -> str:
        prompt = (
            "Explain the topic clearly in simple words.\n"
            "Include: 1) definition 2) 1 simple example 3) 1 common mistake.\n\n"
            f"Topic: {topic}\n"
            f"Context: {context}\n"
        )
        return self.invoke_text(prompt)

    def analyze_image(self, prompt: str, image_bytes: bytes) -> str:
        import base64
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        resp = self.client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {"role": "system", "content": "You are a technical interviewer analyzing a whiteboard."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                },
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
