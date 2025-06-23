import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.vector_store import VectorStore

def populate_vector_store():
    print("Populating Vector Store...")
    print("=" * 50)
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Get all processed files
    processed_dir = "processed_data"
    files = [f for f in os.listdir(processed_dir) if f.endswith('.json')]
    
    print(f"Found {len(files)} processed files")
    
    for filename in files:
        print(f"Processing {filename}...")
        file_path = os.path.join(processed_dir, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Add documents to vector store
            vector_store.add_documents(data)
            print(f"Added {len(data)} chunks from {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print("Vector store population complete!")

if __name__ == "__main__":
    populate_vector_store() 