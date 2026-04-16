from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
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

app = FastAPI(title="InterviewAI ML Backend", version="5.10")
client = Groq()

# CORS — allows Node.js backend to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Waking up and loading Whisper into CPU RAM...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("Whisper is locked and loaded on CPU!")

# --- HELPER FUNCTIONS ---
def normalize_lang_code(lang: str) -> str:
    """Ensures we always have a clean 2-letter ISO code for TTS routing."""
    lang = lang.lower().strip()
    mapping = {
        "english": "en", "en": "en",
        "hindi": "hi", "hi": "hi"
    }
    return mapping.get(lang, "en")

def get_full_lang_name(lang_code: str) -> str:
    """Gives Llama 3 the explicit full name of the language to prevent English bleed-through."""
    mapping = {
        "en": "English",
        "hi": "Hindi (in Devanagari script)"
    }
    return mapping.get(lang_code, "English")


# --- PYDANTIC MODELS ---

class TranscriptTurn(BaseModel):
    speaker: str  # "ai" or "user"
    text: str

class EvaluateInput(BaseModel):
    transcript: List[TranscriptTurn]
    domain: str
    language: str
    # NEW: Optional audio metrics passed from Node.js at the end of the interview
    audio_metrics: Optional[Dict[str, str]] = None 

class ParseResumeInput(BaseModel):
    resume_base64: str
    filename: str

class TTSRequest(BaseModel):
    text: str
    language: str  # No default — forces frontend compliance

class ChatMessage(BaseModel):
    speaker: str  # "ai" or "user"
    text: str

class GenerateQuestionInput(BaseModel):
    domain: str
    language: str
    history: List[ChatMessage]


# --------------------------------------------------------
# ROUTE 1: HEALTH CHECK (/health)
# --------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ML Backend V5.10 — Audio Feedback & Identity Fix"}


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

        if lang_code == "hi":
            system_prompt = f"""आप Priya हैं, एक senior technical interviewer जो एक top Indian tech company में {request.domain} role के लिए एक उम्मीदवार का mock interview ले रही हैं।

आपकी personality: professional लेकिन friendly, जैसे एक असली Indian interviewer होती है।

नियम:
- एक बार में केवल एक सवाल पूछें।
- उम्मीदवार के पिछले जवाब पर एक छोटी सी स्वाभाविक प्रतिक्रिया दें।
- केवल positive या neutral acknowledgment दें जैसे: "अच्छा।", "ठीक है।", "समझ आया।", "बढ़िया।"
- कभी मत कहें कि उम्मीदवार repeat कर रहा है।
- "repeating", "फिर से वही", "दोबारा" जैसे शब्द कभी मत इस्तेमाल करें।
- अगर उम्मीदवार को जवाब नहीं पता, तो कहें "कोई बात नहीं, आगे बढ़ते हैं।" और अगला सवाल पूछें।
- पूरा जवाब 3 sentences से कम रखें।
- पहले message के बाद अपना नाम कभी मत बताएं।
- ABSOLUTE RULE: केवल और केवल हिंदी में बोलें। एक भी अंग्रेजी शब्द नहीं।
- इंटरव्यू में अब तक {history_length} exchanges हुए हैं। 8 या उससे ज़्यादा होने पर गर्मजोशी से धन्यवाद देकर interview समाप्त करें।"""

        else:
            system_prompt = f"""You are Sarah, a senior technical interviewer at a top Indian tech company, conducting a mock interview for a {request.domain} role.

Your personality: warm but professional, encouraging but direct. You sound like a real person, not a chatbot.

RULES:
- Ask ONE focused question at a time.
- Before your question, acknowledge the previous answer in ONE short sentence.
- You may ONLY use positive or neutral acknowledgments such as: "Good point.", "That makes sense.", "Interesting approach.", "Got it.", "Fair enough.", "I like that framework."
- NEVER accuse the candidate of repeating themselves.
- NEVER use the words "repeating", "again", "you said that already", or any variation.
- If the candidate says they don't know, say "No worries, let's move on." and ask the next question.
- Keep your TOTAL response under 3 sentences.
- NEVER introduce yourself by name again after the first message.
- Do NOT evaluate, score, or give feedback during the interview.
- The interview has had {history_length} exchanges. If 8 or more, wrap up warmly — thank the candidate and let them know you'll be reviewing their responses."""

        messages = [{"role": "system", "content": system_prompt}]

        if not request.history:
            if lang_code == "hi":
                messages.append({
                    "role": "user",
                    "content": f"{request.domain} के लिए interview शुरू करें। आपका नाम Priya है। अपना brief introduction दें और पहला सवाल हिंदी में पूछें।"
                })
            else:
                messages.append({
                    "role": "user",
                    "content": f"Start the interview. Your name is Sarah. Briefly introduce yourself as the interviewer and ask the first question for a {request.domain} role."
                })
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

        return {"question": completion.choices[0].message.content.strip()}

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

        system_prompt = f"""You are a strict but fair technical interviewer evaluating a job candidate for a {request.domain} role.

CRITICAL IDENTITY RULE:
- "Sarah" (English) and "Priya" (Hindi) are the names of the AI INTERVIEWERS.
- You are evaluating the CANDIDATE. 
- NEVER call the candidate "Sarah" or "Priya". Refer to them strictly as "the candidate".

The transcript below contains an interview conversation. Lines starting with "AI", "INTERVIEWER", "SARAH", or "PRIYA" are the interviewer's questions. ALL OTHER lines are the CANDIDATE's answers — these are what you must evaluate.

SCORING RULES (all scores must be between 1.0 and 10.0, NEVER 0):
- "technical_accuracy": Score based ONLY on the candidate's answers.
    * Candidate says "I don't know" or gives no technical content at all → score 1-2
    * Partial or basic answers → score 3-5
    * Mostly correct answers → score 6-7
    * Strong and detailed answers → score 8-10
- "communication": How clearly and professionally did the candidate express themselves?
- "confidence": Did they sound confident or hesitant?
- "clarity": Were their answers structured and easy to follow?
- "overall": A fair weighted average of all four scores above.
"""
        # Inject Audio Metrics into the prompt if the frontend provided them
        if request.audio_metrics:
            system_prompt += f"""
VOCAL DELIVERY DATA:
The candidate's audio was analyzed during the interview. 
Pitch Variation: {request.audio_metrics.get('pitch_variation', 'average')}
Energy/Volume Level: {request.audio_metrics.get('energy_level', 'average')}

CRITICAL AUDIO INSTRUCTION: You MUST include 1 sentence in the "feedback" paragraph commenting on their pitch and volume based on the data above (e.g., "Your vocal energy was a bit low," or "You had great pitch variation which kept the answers engaging").
"""

        system_prompt += f"""
CRITICAL: All text in "feedback", "strengths", and "improvements" MUST be written exclusively in {full_lang_name}.

Return ONLY this strict JSON:
{{
  "scores": {{
    "communication": <float 1-10>,
    "technical_accuracy": <float 1-10>,
    "confidence": <float 1-10>,
    "clarity": <float 1-10>,
    "overall": <float 1-10>
  }},
  "feedback": "<2-3 sentences of overall feedback in {full_lang_name}>",
  "strengths": ["<strength 1>", "<strength 2>"],
  "improvements": ["<improvement 1>", "<improvement 2>"],
  "filler_words": {{"um": <int>, "like": <int>, "uh": <int>}}
}}"""

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the interview transcript:\n\n{script}"}
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        result = json.loads(completion.choices[0].message.content)

        # Safety net: clamp all scores to 1-10 regardless of what model returns
        if "scores" in result:
            for key in result["scores"]:
                result["scores"][key] = max(1.0, min(10.0, float(result["scores"][key])))

        return result

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

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
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
            "en": "en-IN-NeerjaNeural",   # Female — matches persona "Sarah"
            "hi": "hi-IN-SwaraNeural"     # Female — matches persona "Priya"
        }

        voice = voice_map.get(lang_code, "en-IN-NeerjaNeural")
        output_file = f"ai_response_{lang_code}_{uuid.uuid4().hex}.mp3"

        # Rate and pitch tuning to reduce robotic feel
        communicate = edge_tts.Communicate(
            request.text,
            voice,
            rate="-5%",    # Slightly slower = more natural pacing
            pitch="-3Hz"   # Slightly lower = warmer, less robotic
        )
        await communicate.save(output_file)

        background_tasks.add_task(os.remove, output_file)

        return FileResponse(
            path=output_file,
            media_type="audio/mpeg",
            filename=output_file
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))