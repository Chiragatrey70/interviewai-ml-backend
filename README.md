---
title: InterviewAI ML Backend
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
# InterviewAI — ML Backend

> *v5.9 Production* · Python 3.11+ · FastAPI · Groq · faster-whisper · edge-tt~s

A state-free Python microservice that acts as the *Brain* and *Voice* of the InterviewAI (PrepMate) mock interview platform. Designed for zero-cost deployment on Hugging Face Spaces with full CPU compatibility.

---

## Live Service


https://your-huggingface-space.hf.space


---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | Python 3.11+, FastAPI, Uvicorn |
| LLM API | Groq (llama-3.1-8b-instant) |
| Speech-to-Text | faster-whisper (base model, CPU / int8) |
| Text-to-Speech | edge-tts (Microsoft Azure Neural Voices) |
| Audio Analysis | librosa, soundfile, numpy |
| PDF Parsing | PyMuPDF (fitz) |
| Cloud Hosting | Hugging Face Spaces (Docker) |
| CI/CD | GitHub Actions |

---

## Setup

### 1. Clone the repository

bash
git clone https://github.com/your-username/interviewai-ml-backend.git
cd interviewai-ml-backend


### 2. Create and activate a virtual environment

bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate


### 3. Install dependencies

bash
pip install -r requirements.txt


### 4. Create a .env file

bash
GROQ_API_KEY=your_groq_api_key_here


### 5. Start the server

bash
uvicorn main:app --reload


> *Optional:* Expose locally for frontend testing via ngrok http 8000

---

## API Reference

### GET /health

Returns the current status and version of the microservice.

*Response:*
json
{ "status": "ok", "version": "5.9" }


---

### POST /stt — Speech-to-Text

*Input:* multipart/form-data — audio file

*Logic:* Offline transcription via faster-whisper (CPU, int8).

*Response:*
json
{
  "transcript": "string",
  "language_detected": "en",
  "confidence": 0.95
}


---

### POST /generate-question — Interview Brain

*Input:*
json
{
  "domain": "Data Analyst",
  "language": "en",
  "history": [
    { "speaker": "ai", "text": "..." },
    { "speaker": "user", "text": "..." }
  ]
}


*Logic:*
- Analyzes conversation history and generates the next contextual question.
- Auto-wraps after *8 exchanges*.
- Uses explicit AI *Personas: *Sarah (English) and Priya (Hindi).
- Strict banned-phrase list prevents hallucinations (e.g., claiming the user is "repeating themselves").

*Response:*
json
{ "question": "Next interview question text..." }


---

### POST /tts — Text-to-Speech

*Input:*
json
{ "text": "Hello, let's begin.", "language": "en" }


> ⚠️ language is *strictly required*.

*Logic:* Converts text to high-quality MP3 audio using Microsoft Azure Neural voices. Applies -5% rate and -3Hz pitch adjustment for a warmer, less robotic cadence.

*Response:* FileResponse — .mp3 audio file

---

### POST /evaluate — Post-Interview Scoring

*Input:*
json
{
  "transcript": [...],
  "domain": "SDE",
  "language": "en"
}


*Response:*
json
{
  "scores": {
    "communication": 8.0,
    "technical_accuracy": 6.0,
    "confidence": 7.0,
    "clarity": 8.0,
    "overall": 7.2
  },
  "feedback": "Overall feedback paragraph...",
  "strengths": ["...", "..."],
  "improvements": ["...", "..."],
  "filler_words": { "um": 2, "like": 0, "uh": 1 }
}


> All scores are mathematically *clamped between 1.0 – 10.0* at the Python level. The backend will never return 0.0.
> If language is "hi", paragraph values are written in pure Hindi while JSON keys remain in English.

---

### POST /parse-resume

*Input:*
json
{ "resume_base64": "...", "filename": "resume.pdf" }


*Response:* Strict JSON with extracted candidate data and AI-suggested interview questions.

---

### POST /audio-confidence

*Input:* multipart/form-data — audio file

*Response:*
json
{
  "confidence_score": 8.5,
  "speaking_pace_wpm": 130,
  "silence_ratio": 0.12,
  "pitch_variation": "medium",
  "energy_level": "moderate"
}


---

## Architecture Notes (v5.9)

### 🧠 Persona Engineering & Hallucination Control
The /generate-question route uses an explicit *whitelist* of allowed acknowledgments and a strict *blacklist* of banned phrases (e.g., "repeating", "again"). This forces Llama 3 8B to behave as a supportive interviewer rather than a harsh critic.

### 🔢 Mathematical Score Clamping
The /evaluate route enforces a hard floor of 1.0 on all scores to prevent frontend crashes or UI bugs caused by LLM edge-case hallucinations.

### 🧹 Memory Management
Temporary audio files are scrubbed via FastAPI BackgroundTasks (for TTS) and finally: blocks (for STT/Scoring) to prevent memory leaks inside the cloud container.

### ☁️ Hugging Face Optimization
Whisper is configured with device="cpu" and compute_type="int8" to run flawlessly on standard, non-GPU cloud environments.

---

## Deployment (Hugging Face Spaces)

This project ships with a production-ready Dockerfile.

bash
# Build locally to verify
docker build -t interviewai-ml-backend .
docker run -p 8000:8000 --env-file .env interviewai-ml-backend


Push to GitHub — CI/CD via *GitHub Actions* auto-deploys to Hugging Face Spaces on every commit to main.

---

## Project Structure


interviewai-ml-backend/
├── main.py              # All FastAPI routes & ML logic
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container config for HF Spaces
├── .env                 # Local secrets (not committed)
├── .gitignore
└── README.md


---

## Language Support

| Language | STT | TTS Voice | Persona |
|---|---|---|---|
| English (en) | ✅ | Neerja (Neural) | Sarah |
| Hindi (hi) | ✅ | Swara (Neural) | Priya |

> MVP scope is English and Hindi exclusively, optimized for reliability over breadth.

---

## Related Repositories

- *Frontend:* [INTERVIEW-PLATFOARM-FRONTEND](https://github.com/Sanyam26362/INTERVIEW-PLATFOARM-FRONTEND-) — Tailwind + shadcn/ui
- *Backend:* [INTERVIEW-PLATFOARM-BACKEND](https://github.com/Sanyam26362/AI-Mock-Interview-Platform-BACKEND-) — Next.js 16 + Clerk Auth + TypeScript
- *Live App:* [interview-platfoarm-frontend.vercel.app](https://interview-platfoarm-frontend.vercel.app)