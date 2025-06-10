import os
import json
import sys

# Add parent directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.vector_store import VectorStore

def load_processed_data_to_vector_store(processed_data_dir: str = "processed_data"):
    vector_store = VectorStore()
    all_chunks = []

    for filename in os.listdir(processed_data_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(processed_data_dir, filename)
            print(f"Loading {filename}...")
            with open(file_path, "r") as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)

    print(f"Adding {len(all_chunks)} chunks to vector store...")
    vector_store.add_documents(all_chunks)
    print("Vector store updated successfully.")

if __name__ == "__main__":
    load_processed_data_to_vector_store()
