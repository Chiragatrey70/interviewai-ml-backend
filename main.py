from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
from groq import Groq
from faster_whisper import WhisperModel
import os
import json
import shutil
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="InterviewAI ML Backend", version="2.0")
client = Groq()

# --- PHASE 2: INITIALIZE WHISPER ON STARTUP ---
print("Waking up the RTX 5060 and loading Whisper into VRAM...")
# We load this globally so it only takes those 14 seconds once when the server boots!
whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
print("Whisper is locked and loaded!")

class Message(BaseModel):
    role: str 
    content: str

class InterviewRequest(BaseModel):
    messages: List[Message]
    language: str = "en" 

class EvaluationRequest(BaseModel):
    messages: List[Message]
    domain: str = "Software Engineering" 

@app.get("/")
def read_root():
    return {"status": "ML Backend is active and running Phase 2!"}

# --- EXISTING PHASE 1 ROUTES ---

@app.post("/api/chat")
def chat_with_ai(request: InterviewRequest):
    try:
        system_prompt = "You are an expert technical interviewer conducting a mock interview. Ask one question at a time. Keep your responses professional, conversational, and concise."
        groq_messages = [{"role": "system", "content": system_prompt}]
        for msg in request.messages:
            groq_messages.append({"role": msg.role, "content": msg.content})
            
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=groq_messages,
            temperature=0.7,
            max_tokens=500,
        )
        return {"response": completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate")
def evaluate_interview(request: EvaluationRequest):
    try:
        system_prompt = f"""You are an expert technical interviewer evaluating a candidate for a {request.domain} role.
        Review the following interview transcript.
        You MUST respond in strict JSON format with the following exact structure:
        {{
            "score": <int from 0 to 100>,
            "strengths": [<array of 2-3 short strings>],
            "weaknesses": [<array of 2-3 short strings>],
            "feedback": "<One paragraph summary of their performance>"
        }}
        """
        transcript = ""
        for msg in request.messages:
            transcript += f"{msg.role.upper()}: {msg.content}\n"
            
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the transcript to evaluate:\n{transcript}"}
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- NEW PHASE 2 ROUTE: AUDIO TRANSCRIPTION ---

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # 1. Save the incoming audio file temporarily to your laptop
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Run the audio through the GPU Whisper model
        # beam_size=5 makes the AI think slightly harder about context for better accuracy
        segments, info = whisper_model.transcribe(temp_file_path, beam_size=5)
        
        # 3. Whisper returns chunks (segments) of text. We join them together.
        transcript = ""
        for segment in segments:
            transcript += segment.text + " "
            
        # 4. Delete the temporary audio file so we don't clutter your hard drive
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
        # 5. Return the text and the detected language back to the frontend!
        return {
            "transcript": transcript.strip(),
            "detected_language": info.language,
            "confidence": info.language_probability
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))