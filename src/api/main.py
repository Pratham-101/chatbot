"""Main FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services.chatbot.enhanced_chatbot import EnhancedMutualFundChatbot
from ingestion.vector_store import VectorStore
import os
import gc
import traceback
from dotenv import load_dotenv
from api.routes import chat, health, voice

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
app = FastAPI(title="Mutual Fund Chatbot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(voice.router, prefix="/api", tags=["voice"])

@app.get("/")
async def root():
    return {"message": "Mutual Fund Chatbot API", "version": "1.0.0"}

# Initialize chatbot with memory-optimized settings
chatbot = EnhancedMutualFundChatbot(model_name="llama3-8b-8192")

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

load_dotenv()
