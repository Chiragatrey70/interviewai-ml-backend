from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from groq import Groq
from faster_whisper import WhisperModel
import os
import json
import shutil
import base64
import fitz  # PyMuPDF
import librosa
import numpy as np
import edge_tts
import uuid 
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="InterviewAI ML Backend", version="5.4")
client = Groq()

print("Waking up the RTX 5060 and loading Whisper into VRAM...")
whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
print("Whisper is locked and loaded!")

# --- HELPER FUNCTIONS (Defense in Depth) ---
def normalize_lang_code(lang: str) -> str:
    """Ensures we always have a clean 2-letter ISO code for TTS routing."""
    lang = lang.lower().strip()
    mapping = {
        "english": "en", "en": "en",
        "hindi": "hi", "hi": "hi",
        "tamil": "ta", "ta": "ta",
        "telugu": "te", "te": "te",
        "bengali": "bn", "bn": "bn",
        "marathi": "mr", "mr": "mr"
    }
    return mapping.get(lang, "en") # Default to English if confused

def get_full_lang_name(lang_code: str) -> str:
    """Gives Llama 3 the explicit full name of the language to prevent English bleed-through."""
    mapping = {
        "en": "English",
        "hi": "Hindi (in Devanagari script)",
        "ta": "Tamil",
        "te": "Telugu",
        "bn": "Bengali",
        "mr": "Marathi"
    }
    return mapping.get(lang_code, "English")

# --- PYDANTIC MODELS (The API Contract) ---

class TranscriptTurn(BaseModel):
    speaker: str # "ai" or "user"
    text: str

class EvaluateInput(BaseModel):
    transcript: List[TranscriptTurn]
    domain: str
    language: str

class ParseResumeInput(BaseModel):
    resume_base64: str
    filename: str

class TTSRequest(BaseModel):
    text: str
    language: str = "en" 

class ChatMessage(BaseModel):
    speaker: str # "ai" or "user"
    text: str

class GenerateQuestionInput(BaseModel):
    domain: str
    language: str
    history: List[ChatMessage]


# ---------------------------------------------------------
# ROUTE 1: HEALTH CHECK (/health)
# ---------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ML Backend running with V5.4 Bulletproof Language Routing!"}


# ---------------------------------------------------------
# ROUTE 2: SPEECH TO TEXT (/stt)
# ---------------------------------------------------------
@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    temp_file = f"temp_stt_{uuid.uuid4().hex}_{file.filename}"
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        segments, info = whisper_model.transcribe(temp_file, beam_size=5)
        transcript_text = "".join([segment.text + " " for segment in segments]).strip()
            
        return {
            "transcript": transcript_text,
            "language_detected": info.language,
            "confidence": round(info.language_probability, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


# ---------------------------------------------------------
# ROUTE 3: GENERATE NEXT QUESTION (/generate-question)
# ---------------------------------------------------------
@app.post("/generate-question")
def generate_question(request: GenerateQuestionInput):
    try:
        history_length = len(request.history)
        lang_code = normalize_lang_code(request.language)
        full_lang_name = get_full_lang_name(lang_code)
        
        system_prompt = f"""You are an expert technical interviewer conducting a mock interview for a {request.domain} role.
        Your task is to ask the NEXT single interview question based on the conversation history.
        - Ask ONLY ONE question.
        - Do NOT evaluate the candidate's previous answer (save that for the end).
        - Keep it conversational, professional, and concise.
        - CRITICAL INSTRUCTION: You MUST speak, think, and respond EXCLUSIVELY in {full_lang_name}. Do NOT use English unless the requested language is English.
        - The interview has had {history_length} exchanges so far. If this is 8 or more exchanges, wrap up the interview naturally by thanking the candidate and telling them the interview is over.
        """

        messages = [{"role": "system", "content": system_prompt}]
        
        if not request.history:
            messages.append({"role": "user", "content": f"Start the interview. Ask the very first question for a {request.domain} role."})
        else:
            for msg in request.history:
                role = "assistant" if msg.speaker == "ai" else "user"
                messages.append({"role": role, "content": msg.text})

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7, 
            max_tokens=150,  
        )

        next_question = completion.choices[0].message.content.strip()

        return {"question": next_question}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# ROUTE 4: INTERVIEW EVALUATION (/evaluate)
# ---------------------------------------------------------
@app.post("/evaluate")
def evaluate_interview(request: EvaluateInput):
    try:
        lang_code = normalize_lang_code(request.language)
        full_lang_name = get_full_lang_name(lang_code)
        
        script = ""
        for turn in request.transcript:
            script += f"{turn.speaker.upper()}: {turn.text}\n"

        system_prompt = f"""You are an expert technical interviewer analyzing a candidate for a {request.domain} role.
        CRITICAL INSTRUCTION: All textual feedback ("feedback", "strengths", "improvements") MUST be written exclusively in {full_lang_name}.
        Review the transcript and return strict JSON exactly matching this structure:
        {{
          "scores": {{
            "communication": <float 1-10>,
            "technical_accuracy": <float 1-10>,
            "confidence": <float 1-10>,
            "clarity": <float 1-10>,
            "overall": <float 1-10>
          }},
          "feedback": "<2-3 sentences of overall feedback>",
          "strengths": ["<strength 1>", "<strength 2>"],
          "improvements": ["<improvement 1>", "<improvement 2>"],
          "filler_words": {{"um": <int>, "like": <int>, "uh": <int>}}
        }}"""

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the transcript:\n{script}"}
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# ROUTE 5: RESUME PARSER (/parse-resume)
# ---------------------------------------------------------
@app.post("/parse-resume")
def parse_resume(request: ParseResumeInput):
    temp_pdf = f"temp_resume_{uuid.uuid4().hex}.pdf"
    try:
        pdf_data = base64.b64decode(request.resume_base64)
        with open(temp_pdf, "wb") as f:
            f.write(pdf_data)
            
        doc = fitz.open(temp_pdf)
        resume_text = ""
        for page in doc:
            resume_text += page.get_text()
        doc.close()
        
        system_prompt = """You are an HR resume parser. Extract the data from the resume text provided.
        You MUST return strict JSON matching this structure perfectly:
        {
          "name": "<string>",
          "skills": ["<skill1>", "<skill2>"],
          "experience_years": <int>,
          "education": "<string>",
          "previous_roles": ["<role1>", "<role2>"],
          "suggested_questions": ["<tailored question 1>", "<tailored question 2>"]
        }"""
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract data from this resume:\n{resume_text}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_pdf):
            os.remove(temp_pdf)


# ---------------------------------------------------------
# ROUTE 6: AUDIO CONFIDENCE SCORING (/audio-confidence)
# ---------------------------------------------------------
@app.post("/audio-confidence")
async def audio_confidence(file: UploadFile = File(...)):
    temp_file = f"temp_audio_{uuid.uuid4().hex}_{file.filename}"
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        y, sr = librosa.load(temp_file, sr=None)
        
        non_mute_intervals = librosa.effects.split(y, top_db=20)
        speaking_samples = sum([end - start for start, end in non_mute_intervals])
        total_samples = len(y)
        silence_ratio = 1.0 - (speaking_samples / total_samples)
        
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pitch_std = np.nanstd(f0) if f0 is not None else 0
        pitch_variation = "low"
        if pitch_std > 20: pitch_variation = "medium"
        if pitch_std > 40: pitch_variation = "high"
        
        rms = librosa.feature.rms(y=y)
        mean_rms = np.mean(rms)
        energy_level = "low"
        if mean_rms > 0.05: energy_level = "moderate"
        if mean_rms > 0.15: energy_level = "high"
            
        raw_score = 7.0 + (mean_rms * 10) - (silence_ratio * 2)
        clamped_score = max(1.0, min(10.0, round(raw_score, 1)))
            
        return {
            "confidence_score": clamped_score, 
            "speaking_pace_wpm": 130, 
            "silence_ratio": round(silence_ratio, 2),
            "pitch_variation": pitch_variation,
            "energy_level": energy_level
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


# ---------------------------------------------------------
# ROUTE 7: TEXT-TO-SPEECH (/tts)
# ---------------------------------------------------------
@app.post("/tts")
async def text_to_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    try:
        lang_code = normalize_lang_code(request.language)
        
        voice_map = {
            "en": "en-IN-NeerjaNeural",   
            "hi": "hi-IN-SwaraNeural",    
            "ta": "ta-IN-PallaviNeural",  
            "te": "te-IN-ShrutiNeural",   
            "bn": "bn-IN-TanishaaNeural", 
            "mr": "mr-IN-AarohiNeural"    
        }
        
        voice = voice_map.get(lang_code, "en-IN-NeerjaNeural")
        output_file = f"ai_response_{lang_code}_{uuid.uuid4().hex}.mp3"
        
        communicate = edge_tts.Communicate(request.text, voice)
        await communicate.save(output_file)
        
        background_tasks.add_task(os.remove, output_file)
        
        return FileResponse(
            path=output_file, 
            media_type="audio/mpeg", 
            filename=output_file
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))