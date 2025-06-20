import os
import json
from typing import List, Dict
import pandas as pd
from sentence_transformers import SentenceTransformer

class ExcelProcessor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        
    def extract_text_from_excel(self, excel_path: str) -> str:
        """Extract text from an Excel file by concatenating all cell values"""
        df = pd.read_excel(excel_path, sheet_name=None)  # Read all sheets
        text = ""
        for sheet_name, sheet_df in df.items():
            # Convert all cells to string and join with spaces
            sheet_text = sheet_df.astype(str).apply(lambda x: ' '.join(x), axis=1).str.cat(sep=' ')
            text += sheet_text + "\n"
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[Dict]:
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
    
    def process_excel(self, excel_path: str) -> List[Dict]:
        """Process an Excel file into chunks with embeddings"""
        import pandas as pd
        df = pd.read_excel(excel_path, sheet_name=None)
        chunks = []
        for sheet_name, sheet_df in df.items():
            # Extract fund name column if exists
            fund_names = []
            if 'Fund Name' in sheet_df.columns:
                fund_names = sheet_df['Fund Name'].astype(str).tolist()
            elif 'Fund' in sheet_df.columns:
                fund_names = sheet_df['Fund'].astype(str).tolist()
            else:
                fund_names = [os.path.basename(excel_path)] * len(sheet_df)
            
            # Convert each row to text and create chunk with fund name metadata
            for idx, row in sheet_df.iterrows():
                row_text = ' '.join(str(val) for val in row.values)
                embedding = self.embedding_model.encode(row_text).tolist()
                chunk = {
                    "text": row_text,
                    "embedding": embedding,
                    "source": os.path.basename(excel_path),
                    "fund_name": fund_names[idx] if idx < len(fund_names) else os.path.basename(excel_path)
                }
                chunks.append(chunk)
        return chunks
    
    def save_processed_data(self, chunks: List[Dict], output_path: str):
        """Save processed chunks to JSON"""
        with open(output_path, 'w') as f:
            json.dump(chunks, f)
            
    def process_file(self, excel_path: str, output_dir: str):
        """Process a single Excel file and save the processed data"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(excel_path))[0]}.json")
        print(f"Processing {excel_path}...")
        chunks = self.process_excel(excel_path)
        self.save_processed_data(chunks, output_path)
        print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python excel_processor.py <excel_file_path> <output_dir>")
        sys.exit(1)
    excel_file_path = sys.argv[1]
    output_dir = sys.argv[2]
    processor = ExcelProcessor()
    processor.process_file(excel_file_path, output_dir)
