import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.vector_store import VectorStore
from sentence_transformers import SentenceTransformer

def test_vector_store():
    print("Testing Vector Store...")
    print("=" * 50)
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Test query
    test_query = "HDFC Large and Mid Cap Fund"
    print(f"Test query: {test_query}")
    
    # Create embedding
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embedding_model.encode(test_query).tolist()
    
    # Query vector store
    print("Querying vector store...")
    results = vector_store.query(query_embedding, k=5, score_threshold=0.1)
    
    print(f"Found {len(results)} results")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Text: {result['text'][:200]}...")
        print(f"  Metadata: {result['metadata']}")
        print()

if __name__ == "__main__":
    test_vector_store() 