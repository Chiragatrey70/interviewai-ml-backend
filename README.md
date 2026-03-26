---
title: InterviewAI ML Backend
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
# InterviewAI ML Backend (v5.7 MVP)

## 📌 Project Overview
**InterviewAI (PrepMate)** is a state-free, GPU-accelerated Python microservice designed to handle heavy Machine Learning tasks for a React/Node.js mock interview platform. 

**MVP Scope:** We have aggressively cut scope to guarantee 100% reliability. This backend currently supports **English** and **Hindi** exclusively, utilizing strict bilingual prompt isolation to prevent LLM translation artifacts.

## 🛠️ Tech Stack & Environment
* **Framework:** Python 3.13+, FastAPI, Uvicorn
* **LLM API:** Groq (`llama-3.1-8b-instant`)
* **Speech-to-Text (STT):** `faster-whisper` (running locally on GPU via `float16`)
* **Text-to-Speech (TTS):** `edge-tts` (Microsoft Azure Neural voices: Neerja & Swara)
* **Audio Analysis:** `librosa`, `soundfile`, `numpy`
* **PDF Parsing:** PyMuPDF (`fitz`)
* **Hardware Profile:** Optimized for local deployment on an NVIDIA RTX 5060 Laptop GPU.

## 🚀 Setup & Installation
1. Create and activate a virtual environment: `python -m venv venv`
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file and add your API key: `GROQ_API_KEY=your_key_here`
4. Start the server: `uvicorn main:app --reload`
5. Expose the port: `ngrok http 8000`

---

## 📡 API Contract

### `GET /health`
Returns the current status and version of the microservice.

### `POST /stt` (Speech-to-Text)
* **Input:** `multipart/form-data` audio file.
* **Output:** `{"transcript": str, "language_detected": str, "confidence": float}`

### `POST /generate-question` (The Interview "Brain")
* **Input JSON:** ```json
  {
    "domain": "Data Analyst",
    "language": "hi",
    "history": [{"speaker": "ai", "text": "..."}, {"speaker": "user", "text": "..."}]
  }