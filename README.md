InterviewAI ML Backend (v5.4)

📌 Project Overview
InterviewAI (PrepMate) is a state-free, GPU-accelerated Python microservice designed to handle heavy Machine Learning tasks for a React/Node.js mock interview platform. It acts as the "Brain" and "Voice" of the AI interviewer, focusing on fast execution, strict JSON API contracts, and robust multi-user concurrency.

Target Audience: Engineering candidates in India, requiring strong multilingual support (English, Hindi, Tamil, Telugu, Bengali, Marathi) and regional Indian accents.

🛠️ Tech Stack & Environment
Framework: Python 3.13+, FastAPI, Uvicorn

LLM API: Groq (llama-3.1-8b-instant) via the groq SDK

Speech-to-Text (STT): faster-whisper (running locally on GPU via float16)

Text-to-Speech (TTS): edge-tts (Microsoft Azure Neural voices)

Audio Analysis: librosa, soundfile, numpy

PDF Parsing: PyMuPDF (fitz)

Hardware Profile: Optimized for local Windows deployment on an NVIDIA RTX 5060 Laptop GPU (CUDA architecture sm_120).

Networking: Exposed to the Node.js frontend via ngrok tunnels.

🚀 Setup & Installation
Create a virtual environment and activate it: python -m venv venv

Install dependencies: pip install -r requirements.txt

Create a .env file in the root directory and add your API key:

Code snippet
GROQ_API_KEY=your_api_key_here
Start the server: uvicorn main:app --reload

Expose the port (in a second terminal): ngrok http 8000

📡 API Contract (Fully Implemented)
GET /health
Returns the current status and version of the microservice.

POST /stt (Speech-to-Text)
Input: multipart/form-data audio file.

Logic: Uses faster-whisper for rapid GPU-based transcription.

Output: {"transcript": str, "language_detected": str, "confidence": float}

POST /generate-question (The Interview "Brain")
Input JSON: ```json
{
"domain": "Data Analyst",
"language": "hi",
"history": [{"speaker": "ai", "text": "..."}, {"speaker": "user", "text": "..."}]
}

Logic: Analyzes chat history and generates the next logical interview question. Includes cold-start handling for empty histories and auto-wraps the interview after 8 exchanges.

Output: {"question": "Next question text..."}

POST /tts (Text-to-Speech)
Input JSON: {"text": "Hello", "language": "en"}

Logic: Maps the ISO language code to premium Indian neural voices.

Output: FileResponse containing the generated .mp3 audio file.

POST /evaluate (Post-Interview Scoring)
Input JSON: {"transcript": [...], "domain": "SDE", "language": "en"}

Logic: Uses Llama 3.1 to grade the candidate out of 10 across multiple metrics.

Output JSON:

JSON
{
  "scores": {"communication": 8.0, "technical_accuracy": 7.5, ...},
  "feedback": "Overall feedback paragraph...",
  "strengths": ["...", "..."],
  "improvements": ["...", "..."],
  "filler_words": {"um": 2, "like": 0, "uh": 1}
}
POST /parse-resume
Input JSON: {"resume_base64": "...", "filename": "resume.pdf"}

Logic: Extracts text via PyMuPDF and structures it using Llama 3.1.

Output: Strict JSON with name, skills, experience_years, education, previous_roles, and suggested_questions.

POST /audio-confidence
Input: multipart/form-data audio file.

Logic: Uses librosa to calculate silence ratios, RMS energy, and pitch variation to generate a vocal confidence score (clamped between 1.0 and 10.0).

🔒 V5.4 Production Architecture Notes
To ensure stability during live integrations, the following safeguards are hardcoded into main.py:

UUID Concurrency: All temporary files (audio, PDFs) use uuid4() to prevent race conditions and file overwrites when multiple candidates use the platform simultaneously.

Background Cleanup: The /tts endpoint utilizes FastAPI's BackgroundTasks to delete the generated .mp3 after it has been successfully sent to the client, preventing memory leaks.

Fail-Safe Deletions: All temporary file deletions are wrapped in finally: blocks to ensure the hard drive stays clean even if an ML model crashes mid-processing.

Defense in Depth Language Routing: A normalize_lang_code helper intercepts sloppy frontend payloads (e.g., passing "Hindi" instead of "hi") to prevent the TTS engine from failing.

Strict LLM Persona: The get_full_lang_name helper forces Llama 3 to output in the requested language (e.g., "Hindi in Devanagari script") via a CRITICAL INSTRUCTION prompt, stopping the 8B model from accidentally defaulting back to English.