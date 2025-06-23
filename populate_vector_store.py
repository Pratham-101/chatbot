import sys
import os
import json
import shutil
from ingestion.pdf_processor import PDFProcessor # Import the class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.vector_store import VectorStore

print("--- Running Full Ingestion and Population Pipeline ---")

# --- Step 1: Process Raw PDFs into JSON ---
PDF_SOURCE_DIR = "data"
PROCESSED_DATA_DIR = "processed_data"

print(f"Preparing to process PDFs from '{PDF_SOURCE_DIR}'...")

# Clean up any previous processed data to ensure a fresh start
if os.path.exists(PROCESSED_DATA_DIR):
    shutil.rmtree(PROCESSED_DATA_DIR)
os.makedirs(PROCESSED_DATA_DIR)

print(f"Calling PDF processor to populate '{PROCESSED_DATA_DIR}'...")
try:
    # Create an instance of the processor and call the method
    processor = PDFProcessor()
    
    # --- New: Select only a subset of files for faster builds ---
    files_to_process = [
        "factsheet1.pdf",
        "factsheet2.pdf"
    ]
    # ---
    
    processor.process_directory(PDF_SOURCE_DIR, PROCESSED_DATA_DIR, file_list=files_to_process)
    print("PDF processing was successful.")
except Exception as e:
    print(f"PDF processing failed: {e}")
    exit(1) # Stop the build if this critical step fails

print("\n--- PDF Processing Complete. Now Populating Vector Store. ---")

# The original code from your `populate_vector_store.py` will now run,
# but with the guarantee that the 'processed_data' directory exists and is populated.

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