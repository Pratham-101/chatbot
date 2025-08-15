import os
from langchain_groq import ChatGroq

llm = ChatGroq(
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model_name=os.environ.get("GROQ_MODEL", "llama3-70b-8192"),
)
print(llm.invoke("Say hello in one sentence."))