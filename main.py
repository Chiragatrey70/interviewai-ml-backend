from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="InterviewAI ML Backend", version="1.2")
client = Groq()

class Message(BaseModel):
    role: str 
    content: str

class InterviewRequest(BaseModel):
    messages: List[Message]
    language: str = "en" 

# --- NEW: Request model for the evaluation endpoint ---
class EvaluationRequest(BaseModel):
    messages: List[Message]
    domain: str = "Software Engineering" # e.g., "React", "Python", etc.

@app.get("/")
def read_root():
    return {"status": "ML Backend is active and running!"}

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
        
        ai_response = completion.choices[0].message.content
        return {"response": ai_response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- NEW: The Evaluation Endpoint ---
@app.post("/api/evaluate")
def evaluate_interview(request: EvaluationRequest):
    try:
        # 1. Strict prompt engineering to force a specific JSON structure
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
        
        # 2. Convert the message array into a readable chat transcript script
        transcript = ""
        for msg in request.messages:
            transcript += f"{msg.role.upper()}: {msg.content}\n"
            
        # 3. Call Llama 3.1 and force JSON output
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the transcript to evaluate:\n{transcript}"}
            ],
            temperature=0.2, # Lower temperature makes the AI more analytical and less creative
            response_format={"type": "json_object"}, # This forces Groq to return valid JSON
        )
        
        # 4. Parse the AI's text response into an actual JSON dictionary
        raw_evaluation = completion.choices[0].message.content
        evaluation_json = json.loads(raw_evaluation)
        
        return evaluation_json
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))