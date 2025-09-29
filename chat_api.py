# api/chat_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils.preprocess import load_models
from utils.response_generator import generate_response
import traceback

app = FastAPI(title="Dynamic Chatbot API")

@app.on_event("startup")
def startup_event():
    try:
        load_models()
        print("[startup] models loaded")
    except Exception as e:
        print("[startup] failed to load models:", e)

@app.get("/")
def root():
    return {"message": " Dynamic Chatbot API is running!"}

@app.get("/health")
def health():
    return {"status":"ok"}

class ChatRequest(BaseModel):
    user: str = "default"
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    try:
        out = generate_response(req.user, req.message)   # expects (user, text)
        return out
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
