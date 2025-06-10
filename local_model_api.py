from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    # Simple echo response for testing
    prompt = request.prompt
    response_text = f"Simulated response for prompt: {prompt[:100]}..."
    return {"response": response_text}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=11434)
