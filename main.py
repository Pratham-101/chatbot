import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot.enhanced_chatbot import EnhancedMutualFundChatbot
from ingestion.vector_store import VectorStore

# --- App Initialization ---
app = FastAPI(
    title="Mutual Fund Chatbot API",
    description="An API to get insights about mutual funds using RAG and live web search.",
    version="1.0.0",
)

chatbot = EnhancedMutualFundChatbot(model_name="llama3-8b-8192")

class QueryRequest(BaseModel):
    text: str

class QueryResponse(BaseModel):
    answer: str
    response_time: float

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    """
    On startup, connect the chatbot to the vector store.
    The data ingestion should be handled as a separate, one-time process
    before starting the API.
    """
    print("API server starting up...")
    vector_store = VectorStore()
    chatbot.set_vector_store(vector_store)
    print("Chatbot is connected to the vector store and ready.")

@app.get("/health", summary="Health Check")
def health_check():
    """
    Simple health check endpoint for Kubernetes liveness and readiness probes.
    """
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse, summary="Ask a question to the chatbot")
async def ask_question(request: QueryRequest):
    """
    Receives a query, processes it with the chatbot, and returns the answer.
    """
    try:
        start_time = asyncio.get_event_loop().time()
        answer = await chatbot.process_query(request.text)
        end_time = asyncio.get_event_loop().time()
        
        if not answer:
            raise HTTPException(status_code=500, detail="Failed to generate a response.")
            
        return QueryResponse(
            answer=answer,
            response_time=round(end_time - start_time, 2)
        )
    except Exception as e:
        print(f"An error occurred during query processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
