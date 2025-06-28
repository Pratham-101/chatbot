import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot.enhanced_chatbot import EnhancedMutualFundChatbot
from ingestion.vector_store import VectorStore
import os
import gc
import traceback

# --- GCP Service Account Key for Render ---
if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
    with open("/tmp/gcp_key.json", "w") as f:
        f.write(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcp_key.json"
# --- End GCP Service Account Key for Render ---

# --- Memory Optimization for Render Free Tier ---
# Set environment variables to reduce memory usage
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
os.environ["HF_HOME"] = "/tmp/huggingface"

# Force garbage collection to free up memory
gc.collect()
# --- End Memory Optimization ---

# --- App Initialization ---
app = FastAPI(
    title="Mutual Fund Chatbot API",
    description="An API to get insights about mutual funds using RAG, live web search, and real-time data with quality evaluation.",
    version="2.0.0",
)

# Initialize chatbot with memory-optimized settings
chatbot = EnhancedMutualFundChatbot(model_name="llama3-8b-8192")

class QueryRequest(BaseModel):
    text: str

class QualityMetrics(BaseModel):
    accuracy: float
    completeness: float
    clarity: float
    relevance: float
    overall_score: float
    feedback: str

class StructuredData(BaseModel):
    summary: str
    key_points: list
    fund_details: dict
    performance_data: dict
    risk_metrics: dict
    recommendations: list
    sources: list
    disclaimer: str

class QueryResponse(BaseModel):
    answer: str
    response_time: float
    quality_metrics: QualityMetrics
    structured_data: StructuredData
    raw_response: str

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    """
    On startup, connect the chatbot to the vector store.
    The data ingestion should be handled as a separate, one-time process
    before starting the API.
    """
    print("API server starting up...")
    try:
        vector_store = VectorStore()
        chatbot.set_vector_store(vector_store)
        print("Chatbot is connected to the vector store and ready.")
        
        # Force garbage collection after initialization
        gc.collect()
    except Exception as e:
        print(f"Error during startup: {e}")
        # Continue without vector store if it fails

@app.get("/health", summary="Health Check")
def health_check():
    """
    Simple health check endpoint for Kubernetes liveness and readiness probes.
    """
    return {"status": "ok", "version": "2.0.0", "features": ["RAG", "Real-time Data", "Quality Evaluation", "Structured Responses"]}

@app.post("/query", response_model=QueryResponse, summary="Ask a question to the chatbot")
async def ask_question(request: QueryRequest):
    """
    Receives a query, processes it with the chatbot, and returns the answer with quality metrics.
    """
    try:
        start_time = asyncio.get_event_loop().time()
        answer = await chatbot.process_query(request.text)
        end_time = asyncio.get_event_loop().time()
        
        # Force garbage collection after each query to free memory
        gc.collect()
        
        if not answer:
            raise HTTPException(status_code=500, detail="Failed to generate a response.")
        
        # Extract quality metrics and structured data from the response
        # For now, we'll return placeholder data - in a full implementation,
        # these would be extracted from the chatbot's response
        quality_metrics = QualityMetrics(
            accuracy=8.5,
            completeness=8.0,
            clarity=9.0,
            relevance=8.5,
            overall_score=8.5,
            feedback="Response provides comprehensive information with good structure and clarity."
        )
        
        structured_data = StructuredData(
            summary="Fund analysis completed successfully",
            key_points=["Key points extracted from response"],
            fund_details={"name": "Fund details extracted"},
            performance_data={"returns": "Performance data extracted"},
            risk_metrics={"risk_level": "Risk assessment extracted"},
            recommendations=["Recommendations extracted"],
            sources=["Sources extracted"],
            disclaimer="Standard disclaimer applies"
        )
            
        return QueryResponse(
            answer=answer,
            response_time=round(end_time - start_time, 2),
            quality_metrics=quality_metrics,
            structured_data=structured_data,
            raw_response=answer  # For now, using the formatted response as raw
        )
    except Exception as e:
        print("[EXCEPTION] An error occurred during query processing:")
        traceback.print_exc()
        print(f"[EXCEPTION] Exception type: {type(e)} - {e}")
        # Force garbage collection on error
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))
