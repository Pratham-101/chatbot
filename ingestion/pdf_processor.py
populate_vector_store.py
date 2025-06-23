from typing import List
import os

class PDFProcessor:
    def process_directory(self, input_dir: str, output_dir: str, file_list: List[str] = None):
        """
        Processes PDFs in a directory and saves chunks to a JSON file.
        If file_list is provided, only processes files in that list.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        all_chunks = []
        
        # Determine which files to process
        if file_list:
            files_to_run = file_list
            print(f"Processing a specific list of {len(files_to_run)} PDF(s).")
        else:
            files_to_run = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
            print(f"Found {len(files_to_run)} PDF(s) to process in '{input_dir}'.")

        for filename in files_to_run:
            if not os.path.exists(os.path.join(input_dir, filename)):
                print(f"--- Warning: File '{filename}' not found in '{input_dir}'. Skipping. ---")
                continue
                
            print(f"--- Processing file: {filename} ---")
            pdf_path = os.path.join(input_dir, filename)
            try:
                # ... existing code ...
                return all_chunks
            except Exception as e:
                # ... existing code ...

if __name__ == "__main__":
    processor = PDFProcessor()
    
    # --- New: Select only a subset of files for faster builds ---
    files_to_process = [
        "factsheet1.pdf",
        "factsheet2.pdf",
        # "HDFC MF Factsheet - April 2025.pdf" # Commented out for now
    ]
    
    # ---
    
    processor.process_directory(PDF_SOURCE_DIR, PROCESSED_DATA_DIR, file_list=files_to_process)
    print("PDF processing was successful.")
except Exception as e:
    # ... existing code ...
    # ... existing code ... 