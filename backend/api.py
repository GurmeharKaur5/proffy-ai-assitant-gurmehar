from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.chatbot import ask_question

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "Proffy AI API is running"}

@app.post("/chat")
def chat(query: Query):
    answer = ask_question(query.question)
    return {"answer": answer}