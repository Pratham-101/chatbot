from chatbot.retrieval import Retriever
from ingestion.vector_store import VectorStore

def test_get_relevant_context():
    vector_store = VectorStore()
    retriever = Retriever(vector_store)
    query = "performance of HDFC Banking and PSU Debt Fund"
    context_chunks = retriever.get_relevant_context(query, k=5)
    print(f"Retrieved {len(context_chunks)} context chunks for query: {query}")
    for idx, chunk in enumerate(context_chunks):
        print(f"Chunk {idx+1}: {chunk[:500]}\\n")

if __name__ == "__main__":
    test_get_relevant_context()
