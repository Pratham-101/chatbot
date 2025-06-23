import sys
import os
import json
import hashlib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.vector_store import VectorStore
from sentence_transformers import SentenceTransformer

def populate_new_vector_store():
    print("Populating Vector Store with New Data...")
    print("=" * 50)
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Load the new processed chunks
    chunks_file = "processed_data_new/processed_chunks.json"
    
    if not os.path.exists(chunks_file):
        print(f"Chunks file not found: {chunks_file}")
        return
    
    print(f"Loading chunks from {chunks_file}...")
    
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"Loaded {len(chunks)} chunks")
        
        # Create embeddings for each chunk and ensure unique IDs
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Creating embeddings and unique IDs...")
        for i, chunk in enumerate(chunks):
            if i % 1000 == 0:
                print(f"Processing chunk {i}/{len(chunks)}")
            
            # Create unique ID based on content and index
            text = chunk.get('text', '')
            fund_name = chunk.get('fund_name', '')
            unique_id = hashlib.md5(f"{text[:100]}_{fund_name}_{i}".encode()).hexdigest()
            chunk['id'] = unique_id
            
            # Create embedding
            if text:
                embedding = embedding_model.encode(text).tolist()
                chunk['embedding'] = embedding
        
        print("Adding chunks to vector store...")
        vector_store.add_documents(chunks)
        
        print("Vector store population complete!")
        print(f"Total chunks added: {len(chunks)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    populate_new_vector_store() 