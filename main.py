from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="InterviewAI ML Backend", version="1.1")
client = Groq()

# --- NEW: Define a single message structure ---
class Message(BaseModel):
    role: str # Will be either "user" or "assistant"
    content: str

# --- UPDATED: Expect a list of messages ---
class InterviewRequest(BaseModel):
    messages: List[Message]
    language: str = "en" 

@app.get("/")
def read_root():
    return {"status": "ML Backend is active and running!"}

@app.post("/api/chat")
def chat_with_ai(request: InterviewRequest):
    try:
        # 1. Define the interviewer persona
        system_prompt = "You are an expert technical interviewer conducting a mock interview. Ask one question at a time. Keep your responses professional, conversational, and concise."
        
        # 2. Start building the message array with the system prompt
        groq_messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # 3. Append the entire conversation history sent from the frontend/Node
        for msg in request.messages:
            groq_messages.append({"role": msg.role, "content": msg.content})
            
        # 4. Make the call to Llama 3.1
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