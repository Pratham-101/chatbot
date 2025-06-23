from fastapi import FastAPI, Request, HTTPException
from chatbot.rag_chatbot import RAGChatbot
from ingestion.vector_store import VectorStore
from chatbot.retrieval import Retriever
import asyncio
import logging
from fastapi.responses import JSONResponse

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vector_store = VectorStore()
retriever = Retriever(vector_store)
rag_chatbot = RAGChatbot()

# Simple queue for batching queries
query_queue = []
queue_lock = asyncio.Lock()

async def process_queue():
    while True:
        await asyncio.sleep(0.1)  # Batch interval
        async with queue_lock:
            if not query_queue:
                continue
            batch = query_queue.copy()
            query_queue.clear()
        # Process batch concurrently
        tasks = [rag_chatbot.generate_answer(item["question"]) for item in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for item, result in zip(batch, results):
            if isinstance(result, Exception):
                logger.error(f"Error processing question: {item['question']}", exc_info=result)
                item["future"].set_exception(result)
            else:
                item["future"].set_result(result)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_queue())

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/ask")
async def ask_question(request: Request):
    try:
        # First check if we can read the request body
        body = await request.body()
        if not body:
            logger.error("Empty request body received")
            return JSONResponse(
                content={"answer": "Please provide a question in the request body."},
                status_code=400
            )

        # Then try to parse as JSON
        try:
            data = await request.json()
        except Exception as e:
            logger.error(f"Invalid JSON received: {body.decode()}", exc_info=e)
            return JSONResponse(
                content={"answer": "Invalid request format. Please send valid JSON."},
                status_code=400
            )

        question = data.get("question", "")
        logger.info(f"Received question: {question}")
        
        if not question:
            return JSONResponse(
                content={"answer": "Please provide a question in the request body."},
                status_code=400
            )

        future = asyncio.get_event_loop().create_future()
        async with queue_lock:
            logger.info(f"Adding question to queue: {question}")
            query_queue.append({"question": question, "future": future})

        try:
            answer = await asyncio.wait_for(future, timeout=120.0)  # Increased timeout to 2 minutes
            logger.info(f"Returning answer for question: {question}")
            return {"answer": answer}
        except asyncio.TimeoutError:
            logger.error(f"Timeout processing question: {question} after 120 seconds")
            return JSONResponse(
                content={"answer": "The request took too long to process. Please try a different question."},
                status_code=504
            )
        except Exception as e:
            logger.error(f"Error processing question: {question}", exc_info=e)
            return JSONResponse(
                content={"answer": "An error occurred while processing your question."},
                status_code=500
            )

    except Exception as e:
        logger.error("Unexpected error in ask_question endpoint", exc_info=e)
        return JSONResponse(
            content={"answer": "An unexpected error occurred."},
            status_code=500
        )
