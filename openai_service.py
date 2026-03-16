"""
OpenAI service — Whisper STT + GPT-4o-mini mood analysis.
One API key handles both speech-to-text and mood detection.
"""

import io
import json
import os
import re

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_KEY = os.getenv("OPENAI_API_KEY", "").strip()
_client = OpenAI(api_key=_KEY) if _KEY else None

if not _KEY:
    import warnings
    warnings.warn(
        "OPENAI_API_KEY is not set in .env — mood analysis will use fallback defaults. "
        "Add OPENAI_API_KEY=<your-key> to .env.",
        stacklevel=1,
    )

EMOTIONS = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

_SYSTEM_PROMPT = """You are an empathetic mood detection assistant.
Analyse the user's message and classify their emotional state as EXACTLY one of:
Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised

Return ONLY a raw JSON object — no markdown, no code block, no extra text:
{
  "emotion": "<one of the 7 emotions>",
  "tips": ["<practical tip 1>", "<practical tip 2>", "<practical tip 3>"],
  "response": "<warm, empathetic spoken reply under 400 characters that gently mentions the tips>"
}"""

_FALLBACK = {
    "emotion": "Neutral",
    "tips": [
        "Take a few deep breaths and pause for a moment.",
        "Step outside for some fresh air if you can.",
        "Talk to someone you trust about how you feel.",
    ],
    "response": "I hear you. Take it one moment at a time — you've got this.",
}


def analyze_mood(text: str) -> dict:
    if not _client:
        print("[openai_service] no API key — returning fallback")
        return _FALLBACK

    try:
        response = _client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        content = response.choices[0].message.content
        print(f"[openai_service] raw response: {content[:120]}")
        result = _parse_json(content)
        emotion = result.get("emotion", "Neutral")
        if emotion not in EMOTIONS:
            emotion = "Neutral"
        result["emotion"] = emotion
        return result
    except Exception as exc:
        print(f"[openai_service] analyze_mood error: {exc}")
        return _FALLBACK


def speech_to_text(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """Transcribe audio using OpenAI Whisper."""
    if not _client:
        raise RuntimeError("OPENAI_API_KEY not set")
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename          # OpenAI SDK uses the name to detect format
    transcript = _client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )
    print(f"[openai_service] whisper transcript: {repr(transcript.text)}")
    return transcript.text


def _parse_json(content: str) -> dict:
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", content)
    if match:
        content = match.group(1)
    return json.loads(content.strip())
