from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.ragService import get_best_response

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"message": "Evey Chatbot API is running!"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        reply = get_best_response(request.message)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
