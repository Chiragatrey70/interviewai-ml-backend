from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv

# 1. Load the secret variables from your .env file
load_dotenv()

# 2. Initialize the FastAPI application
app = FastAPI(title="InterviewAI ML Backend", version="1.0")

# 3. Initialize the Groq client 
# (It automatically looks for the GROQ_API_KEY in your environment)
client = Groq()

# 4. Define the exact structure of data we expect to receive
class InterviewRequest(BaseModel):
    user_message: str
    # We will use this later when you start phase 2 multilingual support
    language: str = "en" 

# 5. A simple health-check route to make sure the server is alive
@app.get("/")
def read_root():
    return {"status": "ML Backend is active and running!"}

# 6. The main route that generates the AI response
@app.post("/api/chat")
def chat_with_ai(request: InterviewRequest):
    try:
        # The 'persona' we are giving the AI
        system_prompt = "You are an expert technical interviewer. Respond to the candidate's message professionally and concisely."
        
        # Make the call to Llama 3
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant", # The fast, free tier model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.user_message}
            ],
            temperature=0.7, # Adds a little natural variation to the responses
            max_tokens=500,
        )
        
        # Extract and return just the text from the AI's response
        ai_response = completion.choices[0].message.content
        return {"response": ai_response}
        
    except Exception as e:
        # If Groq is down or your key is wrong, this will tell us why
        raise HTTPException(status_code=500, detail=str(e))