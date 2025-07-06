import json
from ingestion.vector_store import VectorStore
from sentence_transformers import SentenceTransformer

VECTOR_STORE_DIR = "vector_store"
TOP_FUNDS_JSON = "data/top_funds_latest.json"


def main():
    # Load top funds data
    with open(TOP_FUNDS_JSON, "r") as f:
        funds = json.load(f)
    print(f"Loaded {len(funds)} funds from {TOP_FUNDS_JSON}")

    # Initialize vector store and embedding model
    vector_store = VectorStore(persist_directory=VECTOR_STORE_DIR)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Prepare documents for ingestion
    documents = []
    for fund in funds:
        text = f"Fund Name: {fund['name']}\nCategory: {fund.get('category', '')}\nAUM: {fund.get('aum', '')}\n1Y Return: {fund.get('1y_return', '')}\n3Y Return: {fund.get('3y_return', '')}\n5Y Return: {fund.get('5y_return', '')}\nSource: {fund.get('source', '')}"
        embedding = embedding_model.encode(text).tolist()
        doc = {
            "id": f"topfund_{fund['name'].replace(' ', '_')}_{fund.get('scraped_at', '')}",
            "text": text,
            "embedding": embedding,
            "source": fund.get('source', ''),
            "fund_name": fund['name'],
            "chunk_type": "top_fund"
        }
        documents.append(doc)

    # Add to vector store
    vector_store.add_documents(documents)
    print(f"Ingested {len(documents)} top funds into vector store.")

if __name__ == "__main__":
    main() 