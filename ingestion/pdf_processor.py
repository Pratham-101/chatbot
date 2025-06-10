import os
import json
from typing import List, Dict
import pdfplumber
from sentence_transformers import SentenceTransformer

class PDFProcessor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Split text into chunks with overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append({
                "text": chunk,
                "start_word": i,
                "end_word": min(i + chunk_size, len(words))
            })
            
        return chunks
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Process a PDF file into chunks with embeddings"""
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(text)
        
        # Add embeddings
        for chunk in chunks:
            chunk["embedding"] = self.embedding_model.encode(chunk["text"]).tolist()
            
        return chunks
    
    def save_processed_data(self, chunks: List[Dict], output_path: str):
        """Save processed chunks to JSON"""
        with open(output_path, 'w') as f:
            json.dump(chunks, f)
            
    def process_directory(self, input_dir: str, output_dir: str):
        """Process all PDFs in a directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in os.listdir(input_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
                
                print(f"Processing {filename}...")
                chunks = self.process_pdf(pdf_path)
                self.save_processed_data(chunks, output_path)
