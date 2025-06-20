from chatbot.retrieval import Retriever
from ingestion.vector_store import VectorStore

def test_get_fund_manager(query: str):
    vector_store = VectorStore()
    retriever = Retriever(vector_store)
    result = retriever.get_fund_manager(query)
    print(f"Query: {query}")
    print(f"Fund Manager Extraction Result: {result}")

if __name__ == "__main__":
    test_query = "who is the manager of Nippon India Consumption Fund?"
    test_get_fund_manager(test_query)
