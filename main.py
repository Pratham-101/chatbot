import os
import json
import traceback
from ingestion.pdf_processor import PDFProcessor
from ingestion.vector_store import VectorStore

def ingest_data():
    print("Starting data ingestion...")
    
    try:
        # Initialize components
        pdf_processor = PDFProcessor()
        vector_store = VectorStore()
        
        # Create processed_data directory if it doesn't exist
        os.makedirs("processed_data", exist_ok=True)
        
        # Process all PDFs in the data directory
        pdf_processor.process_directory("data", "processed_data")
        
        # Add processed documents to vector store
        for filename in os.listdir("processed_data"):
            if filename.endswith('.json'):
                filepath = os.path.join("processed_data", filename)
                with open(filepath, 'r') as f:
                    documents = json.load(f)
                    print(f"Loaded {len(documents)} documents from {filename}")
                    
                    for i, doc in enumerate(documents):
                        # Defensive check: Replace None values in all fields with empty string
                        for key, value in doc.items():
                            if value is None:
                                print(f"Warning: Document {i} field '{key}' is None in file {filename}, replacing with empty string.")
                                doc[key] = ""
                        doc["source"] = filename
                    
                    # Add documents after cleaning
                    vector_store.add_documents(documents)
        
        print("Data ingestion complete!")
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # First run: ingest data
    ingest_data()
    
    # Then start the API (in production you'd run this separately)
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
