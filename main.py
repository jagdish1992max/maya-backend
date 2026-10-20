import os
import base64
import httpx
from anthropic import Anthropic
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

SARVAM_KEY = os.environ["SARVAM_API_KEY"]
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")  # haiku = fast, voice ke liye best
SPEAKER = os.getenv("TTS_SPEAKER", "anushka")

client = Anthropic()  # ANTHROPIC_API_KEY env se leta hai

SYSTEM_PROMPT = """Tum MAYA ho, ek friendly aur thodi masti-bhari AI dost.
- Hinglish mein baat karo (Hindi + English mix, Roman script mein).
- Jawab chhote rakho: 1-3 vakya. Ye voice call hai, essay nahi.
- Warm, casual aur natural tone rakho. Emoji, bullet points ya markdown mat use karo.
- Agar kuch pata nahi, to seedha bol do.
"""

app = FastAPI()

# Production mein "*" ki jagah apni website ka URL daalo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory history (server restart par reset ho jaati hai)
sessions: dict[str, list[dict]] = {}
MAX_TURNS = 20


async def speech_to_text(audio: bytes, filename: str, content_type: str) -> str:
    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": SARVAM_KEY},
            files={"file": (filename, audio, content_type)},
            data={"model": "saarika:v2.5", "language_code": "unknown"},
        )
    if r.status_code != 200:
        raise HTTPException(502, f"STT error: {r.text}")
    return r.json().get("transcript", "").strip()


async def text_to_speech(text: str) -> str:
    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={"api-subscription-key": SARVAM_KEY},
            json={
                "text": text,
                "target_language_code": "hi-IN",
                "speaker": SPEAKER,
                "model": "bulbul:v2",
            },
        )
    if r.status_code != 200:
        raise HTTPException(502, f"TTS error: {r.text}")
    return r.json()["audios"][0]  # base64 wav


@app.post("/chat")
async def chat(audio: UploadFile = File(...), session_id: str = Form("default")):
    data = await audio.read()
    if not data:
        raise HTTPException(400, "Empty audio")

    # 1. Sunna
    user_text = await speech_to_text(
        data, audio.filename or "audio.webm", audio.content_type or "audio/webm"
    )
    if not user_text:
        raise HTTPException(422, "Kuch sunai nahi diya, dobara bolo")

    # 2. Sochna
    history = sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": user_text})
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=history[-MAX_TURNS:],
    )
    reply = resp.content[0].text.strip()
    history.append({"role": "assistant", "content": reply})

    # 3. Bolna
    audio_b64 = await text_to_speech(reply)

    return {"user_text": user_text, "reply_text": reply, "audio_base64": audio_b64}


@app.get("/")
def health():
    return {"status": "ok"}
