from fastapi import FastAPI, Request
from chatbot.rag_chatbot import RAGChatbot
from ingestion.vector_store import VectorStore
from chatbot.retrieval import Retriever

app = FastAPI()

vector_store = VectorStore()
retriever = Retriever(vector_store)
rag_chatbot = RAGChatbot()

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question", "")
    if not question:
        return {"answer": "Please provide a question."}
    answer = await rag_chatbot.generate_answer(question)
    return {"answer": answer}
