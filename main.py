import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from pymongo import MongoClient
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
mongo_uri = os.getenv("MONGO_URI")

client = MongoClient(mongo_uri)
db = client["Chatbot"]
collections = db["users"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Chatrequest(BaseModel):
    user_id: str
    question: str

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("placeholder", "{history}"),
    ("human", "{question}")
])

llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")
chain = prompt | llm


def get_history(user_id):
    chats = collections.find({"user_id": user_id}).sort("timestamp", 1)
    history = []

    for chat in chats:
        if chat["role"] == "human":
            history.append(HumanMessage(content=chat["message"]))
        elif chat["role"] == "assistant":
            history.append(AIMessage(content=chat["message"]))

    return history

@app.get("/")
def home():
    return {"message":"Welcome to AI Study Assistant."}


@app.post("/chat")
def chat(request: Chatrequest):
    history = get_history(request.user_id)

    response = chain.invoke({
        "history": history,
        "question": request.question
    })

    collections.insert_one({
        "user_id": request.user_id,
        "role": "human",
        "message": request.question,
        "timestamp": datetime.utcnow()
    })

    collections.insert_one({
        "user_id": request.user_id,
        "role": "assistant",
        "message": response.content,
        "timestamp": datetime.utcnow()
    })

    return {"response": response.content}