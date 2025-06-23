import sys
import os
import json
import shutil
import hashlib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.vector_store import VectorStore
from sentence_transformers import SentenceTransformer

def reset_and_populate():
    print("Resetting and Populating Vector Store...")
    print("=" * 50)
    
    # Clear vector store directory
    vector_store_dir = "vector_store"
    if os.path.exists(vector_store_dir):
        print("Clearing existing vector store...")
        shutil.rmtree(vector_store_dir)
    
    os.makedirs(vector_store_dir, exist_ok=True)
    
    # Initialize new vector store
    vector_store = VectorStore()
    
    # Load chunks
    chunks_file = "processed_data_new/processed_chunks.json"
    
    if not os.path.exists(chunks_file):
        print(f"Chunks file not found: {chunks_file}")
        return
    
    print(f"Loading chunks from {chunks_file}...")
    
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"Loaded {len(chunks)} chunks")
        
        # Create embeddings and unique IDs
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Processing chunks...")
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i % 1000 == 0:
                print(f"Processing chunk {i}/{len(chunks)}")
            
            text = chunk.get('text', '')
            if not text:
                continue
            
            # Create unique ID
            unique_id = f"chunk_{i}_{hashlib.md5(text[:50].encode()).hexdigest()[:8]}"
            
            # Create embedding
            embedding = embedding_model.encode(text).tolist()
            
            processed_chunk = {
                'id': unique_id,
                'text': text,
                'embedding': embedding,
                'metadata': {
                    'fund_name': chunk.get('fund_name', ''),
                    'source': chunk.get('source', 'factsheet'),
                    'chunk_type': chunk.get('chunk_type', 'fund_info')
                }
            }
            
            processed_chunks.append(processed_chunk)
        
        print(f"Adding {len(processed_chunks)} chunks to vector store...")
        vector_store.add_documents(processed_chunks)
        
        print("Vector store population complete!")
        print(f"Total chunks added: {len(processed_chunks)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reset_and_populate() 