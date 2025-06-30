import asyncio
import re
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot.enhanced_chatbot import EnhancedMutualFundChatbot
from ingestion.vector_store import VectorStore
import os
import gc

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

def extract_structured_data_from_response(response: str) -> StructuredData:
    """
    Extract structured data from the chatbot's formatted response
    """
    try:
        # Extract summary
        summary_match = re.search(r'## 📋 Summary\s*\n(.*?)(?=\n##|\n🔑|\n📈|\n📊|\n⚠️|\n💡|\n📚|\n🎯|\n---|$)', response, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else "Summary not available"
        
        # Extract key points
        key_points_match = re.search(r'## 🔑 Key Points\s*\n(.*?)(?=\n##|\n📈|\n📊|\n⚠️|\n💡|\n📚|\n🎯|\n---|$)', response, re.DOTALL)
        key_points = []
        if key_points_match:
            points_text = key_points_match.group(1).strip()
            key_points = [point.strip().lstrip('•').strip() for point in points_text.split('\n') if point.strip() and '•' in point]
        
        # Extract fund details
        fund_details_match = re.search(r'## 📈 Fund Details\s*\n(.*?)(?=\n##|\n📊|\n⚠️|\n💡|\n📚|\n🎯|\n---|$)', response, re.DOTALL)
        fund_details = {}
        if fund_details_match:
            details_text = fund_details_match.group(1).strip()
            # Extract name from the text
            name_match = re.search(r'\*\*Name:\*\*\s*(.*?)(?=\n|$)', details_text)
            if name_match:
                fund_details["name"] = name_match.group(1).strip()
            else:
                fund_details["name"] = "Fund details not available"
        
        # Extract performance data
        performance_match = re.search(r'## 📊 Performance Data\s*\n(.*?)(?=\n##|\n⚠️|\n💡|\n📚|\n🎯|\n---|$)', response, re.DOTALL)
        performance_data = {}
        if performance_match:
            perf_text = performance_match.group(1).strip()
            returns_match = re.search(r'\*\*Returns:\*\*\s*(.*?)(?=\n|$)', perf_text)
            if returns_match:
                performance_data["returns"] = returns_match.group(1).strip()
            else:
                performance_data["returns"] = "Performance data not available"
        
        # Extract risk metrics
        risk_match = re.search(r'## ⚠️ Risk Metrics\s*\n(.*?)(?=\n##|\n💡|\n📚|\n🎯|\n---|$)', response, re.DOTALL)
        risk_metrics = {}
        if risk_match:
            risk_text = risk_match.group(1).strip()
            risk_level_match = re.search(r'\*\*Risk Level:\*\*\s*(.*?)(?=\n|$)', risk_text)
            if risk_level_match:
                risk_metrics["risk_level"] = risk_level_match.group(1).strip()
            else:
                risk_metrics["risk_level"] = "Risk assessment not available"
        
        # Extract recommendations
        recommendations_match = re.search(r'## 💡 Recommendations\s*\n(.*?)(?=\n##|\n📚|\n🎯|\n---|$)', response, re.DOTALL)
        recommendations = []
        if recommendations_match:
            rec_text = recommendations_match.group(1).strip()
            recommendations = [rec.strip().lstrip('•').strip() for rec in rec_text.split('\n') if rec.strip() and '•' in rec]
        
        # Extract sources
        sources_match = re.search(r'## 📚 Sources\s*\n(.*?)(?=\n##|\n🎯|\n---|$)', response, re.DOTALL)
        sources = []
        if sources_match:
            sources_text = sources_match.group(1).strip()
            sources = [src.strip().lstrip('•').strip() for src in sources_text.split('\n') if src.strip() and '•' in src]
        
        # Extract disclaimer
        disclaimer_match = re.search(r'Past performance does not guarantee future results.*?(?=\n##|\n🎯|\n---|$)', response, re.DOTALL)
        disclaimer = disclaimer_match.group(0).strip() if disclaimer_match else "Standard disclaimer applies"
        
        return StructuredData(
            summary=summary,
            key_points=key_points,
            fund_details=fund_details,
            performance_data=performance_data,
            risk_metrics=risk_metrics,
            recommendations=recommendations,
            sources=sources,
            disclaimer=disclaimer
        )
    except Exception as e:
        print(f"Error extracting structured data: {e}")
        return StructuredData(
            summary="Error extracting summary",
            key_points=["Error extracting key points"],
            fund_details={"name": "Error extracting fund details"},
            performance_data={"returns": "Error extracting performance data"},
            risk_metrics={"risk_level": "Error extracting risk metrics"},
            recommendations=["Error extracting recommendations"],
            sources=["Error extracting sources"],
            disclaimer="Standard disclaimer applies"
        )

def extract_quality_metrics_from_response(response: str) -> QualityMetrics:
    """
    Extract quality metrics from the chatbot's response
    """
    try:
        # Look for quality metrics section
        quality_match = re.search(r'## 🎯 Response Quality Assessment\s*\n(.*?)(?=\n---|$)', response, re.DOTALL)
        if quality_match:
            quality_text = quality_match.group(1).strip()
            
            # Extract scores
            overall_match = re.search(r'\*\*Overall Score:\*\*\s*(\d+\.?\d*)/10', quality_text)
            accuracy_match = re.search(r'\*\*Accuracy:\*\*\s*(\d+\.?\d*)/10', quality_text)
            completeness_match = re.search(r'\*\*Completeness:\*\*\s*(\d+\.?\d*)/10', quality_text)
            clarity_match = re.search(r'\*\*Clarity:\*\*\s*(\d+\.?\d*)/10', quality_text)
            relevance_match = re.search(r'\*\*Relevance:\*\*\s*(\d+\.?\d*)/10', quality_text)
            
            # Extract feedback
            feedback_match = re.search(r'\*\*Feedback:\*\*\s*(.*?)(?=\n|$)', quality_text)
            
            return QualityMetrics(
                accuracy=float(accuracy_match.group(1)) if accuracy_match else 7.0,
                completeness=float(completeness_match.group(1)) if completeness_match else 7.0,
                clarity=float(clarity_match.group(1)) if clarity_match else 7.0,
                relevance=float(relevance_match.group(1)) if relevance_match else 7.0,
                overall_score=float(overall_match.group(1)) if overall_match else 7.0,
                feedback=feedback_match.group(1).strip() if feedback_match else "Feedback not available"
            )
    except Exception as e:
        print(f"Error extracting quality metrics: {e}")
    
    # Fallback to default values
    return QualityMetrics(
        accuracy=7.0,
        completeness=7.0,
        clarity=7.0,
        relevance=7.0,
        overall_score=7.0,
        feedback="Quality metrics extraction failed"
    )

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
        
        # Extract quality metrics and structured data from the chatbot's response
        quality_metrics = extract_quality_metrics_from_response(answer)
        structured_data = extract_structured_data_from_response(answer)
            
        return QueryResponse(
            answer=answer,
            response_time=round(end_time - start_time, 2),
            quality_metrics=quality_metrics,
            structured_data=structured_data,
            raw_response=answer
        )
    except Exception as e:
        print(f"An error occurred during query processing: {e}")
        # Force garbage collection on error
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))
