import os
import time
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

SARVAM_KEY = os.environ["SARVAM_API_KEY"]
GEMINI_KEY = os.environ["GEMINI_API_KEY"]
# Model ka naam AI Studio mein check kar lena, ye naam badalte rehte hain
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
SPEAKER = os.getenv("TTS_SPEAKER", "ishita")
SYSTEM_PROMPT = """Tum MAYA ho, ek friendly aur thodi masti-bhari AI dost.
- Hinglish mein baat karo (Hindi + English mix, Roman script mein).
- Jawab chhote rakho: 1-3 vakya. Ye voice call hai, essay nahi.
- Warm, casual aur natural tone rakho. Emoji, bullet points ya markdown mat use karo.
- Agar kuch pata nahi, to seedha bol do.
"""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, list[dict]] = {}
MAX_TURNS = 20


async def speech_to_text(audio: bytes, filename: str, content_type: str) -> str:
    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": SARVAM_KEY},
            files={"file": (filename, audio, content_type.split(";")[0])},
            data={"model": "saarika:v2.5", "language_code": "unknown"},
        )
    if r.status_code != 200:
        raise HTTPException(502, f"STT error: {r.text}")
    return r.json().get("transcript", "").strip()


async def ask_gemini(history: list[dict]) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": history[-MAX_TURNS:],
        "generationConfig": {"maxOutputTokens": 600},
    }
    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.post(url, headers={"x-goog-api-key": GEMINI_KEY}, json=body)
    if r.status_code != 200:
        raise HTTPException(502, f"LLM error: {r.text}")
    try:
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        raise HTTPException(502, f"LLM empty reply: {r.text}")


async def text_to_speech(text: str) -> str:
    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={"api-subscription-key": SARVAM_KEY},
            json={
                "text": text,
                "target_language_code": "hi-IN",
                "speaker": SPEAKER,
                "model": "bulbul:v3",
            },
        )
    if r.status_code != 200:
        raise HTTPException(502, f"TTS error: {r.text}")
    return r.json()["audios"][0]


@app.post("/chat")
async def chat(audio: UploadFile = File(...), session_id: str = Form("default")):
    data = await audio.read()
    if not data:
        raise HTTPException(400, "Empty audio")

    user_text = await speech_to_text(
        data, audio.filename or "audio.webm", audio.content_type or "audio/webm"
    )
    if not user_text:
        raise HTTPException(422, "Kuch sunai nahi diya, dobara bolo")

    history = sessions.setdefault(session_id, [])
    history.append({"role": "user", "parts": [{"text": user_text}]})
    reply = await ask_gemini(history)
    history.append({"role": "model", "parts": [{"text": reply}]})

    audio_b64 = await text_to_speech(reply)
    return {"user_text": user_text, "reply_text": reply, "audio_base64": audio_b64}


@app.get("/")
def health():
    return {"status": "ok"}
