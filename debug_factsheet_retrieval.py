import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.vector_store import VectorStore
from sentence_transformers import SentenceTransformer

async def debug_factsheet_retrieval():
    print("Debugging Factsheet Retrieval...")
    print("=" * 50)
    
    # Initialize vector store
    vector_store = VectorStore()
    print(f"Vector store contains {vector_store.count_documents()} documents.")
    
    # Test query
    query = "HDFC Large and Mid Cap Fund"
    print(f"\nTest query: {query}")
    
    # Create embedding
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embedding_model.encode(query).tolist()
    
    # Query vector store
    print("Querying vector store...")
    results = vector_store.query(query_embedding, k=5, score_threshold=0.1)
    
    print(f"Found {len(results)} results")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Text: {result['text'][:300]}...")
        print(f"  Metadata: {result['metadata']}")
    
    # Test with different query
    query2 = "ICICI Prudential"
    print(f"\n\nTest query 2: {query2}")
    
    query_embedding2 = embedding_model.encode(query2).tolist()
    results2 = vector_store.query(query_embedding2, k=3, score_threshold=0.1)
    
    print(f"Found {len(results2)} results")
    for i, result in enumerate(results2):
        print(f"\nResult {i+1}:")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Text: {result['text'][:300]}...")
        print(f"  Metadata: {result['metadata']}")

if __name__ == "__main__":
    asyncio.run(debug_factsheet_retrieval()) 