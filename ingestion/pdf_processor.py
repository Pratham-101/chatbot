import json
import re
import os
from typing import List, Dict, Any
import pdfplumber
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

class PDFProcessor:
    def __init__(self):
        # This regex is designed to find fund names that are likely to be headers of sections.
        # It looks for names of major fund houses followed by a standard fund name structure.
        self.fund_header_pattern = re.compile(
            r'^\s*(HDFC|ICICI\s+Prudential|Kotak|SBI|Nippon\s+India)\s+([A-Z][a-z]+\s*){1,5}(Fund|Scheme|Plan|Opportunities)\b',
            re.MULTILINE | re.IGNORECASE
        )
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from a PDF."""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(x_tolerance=2)
                    if page_text:
                        text += page_text + "\n\n"  # Add space between pages
        except Exception as e:
            print(f"pdfplumber failed for {pdf_path}, falling back. Error: {e}")
        
        if not text.strip():
            try:
                reader = PdfReader(pdf_path)
                for page in reader.pages:
                    text += page.extract_text() + "\n\n"
            except Exception as e:
                print(f"pypdf failed for {pdf_path}: {e}")
        return text

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Processes a single PDF into structured, content-rich chunks."""
        print(f"Processing {pdf_path}...")
        full_text = self.extract_text_from_pdf(pdf_path)
        if not full_text:
            return []

        sections = self._split_text_into_sections(full_text)
        chunks = self._chunk_sections(sections)

        print(f"Created {len(chunks)} chunks from {pdf_path}")
        return chunks

    def _split_text_into_sections(self, text: str) -> List[Dict[str, str]]:
        """Splits the full text into sections, one for each fund."""
        sections = []
        # Find all potential fund headers in the document
        matches = list(self.fund_header_pattern.finditer(text))

        for i, match in enumerate(matches):
            start_pos = match.start()
            # The section for the current fund ends where the next fund's section begins
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            section_text = text[start_pos:end_pos].strip()
            # Clean up the extracted fund name
            fund_name = ' '.join(match.group(0).strip().split())

            # A simple heuristic to filter out table of contents entries: real sections should have substantial content.
            if len(section_text.split()) > 30:
                 sections.append({'fund_name': fund_name, 'text': section_text})

        return sections

    def _chunk_sections(self, sections: List[Dict[str, str]], chunk_size: int = 400, overlap_sentences: int = 1) -> List[Dict[str, Any]]:
        """Breaks down large fund sections into smaller, overlapping chunks of text."""
        all_chunks = []
        for section in sections:
            # Split section into sentences. This is more robust than splitting by character count alone.
            sentences = re.split(r'(?<=[.!?])\s+', section['text'].replace('\n', ' '))
            
            current_chunk_sentences = []
            current_length = 0
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sentence_len = len(sentence)
                if current_length + sentence_len > chunk_size and current_chunk_sentences:
                    # When a chunk is full, finalize it
                    chunk_text = " ".join(current_chunk_sentences)
                    # Generate embedding for the chunk
                    embedding = self.embedding_model.encode(chunk_text).tolist()
                    all_chunks.append({
                        "fund_name": section['fund_name'],
                        "text": chunk_text,
                        "embedding": embedding,
                        "source": "factsheet",
                        "chunk_type": "fund_details"
                    })
                    # Start the next chunk with an overlap to maintain context
                    current_chunk_sentences = current_chunk_sentences[-overlap_sentences:]
                    current_length = len(" ".join(current_chunk_sentences))

                current_chunk_sentences.append(sentence)
                current_length += sentence_len + 1

            # Add the final chunk for the section
            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                # Generate embedding for the final chunk
                embedding = self.embedding_model.encode(chunk_text).tolist()
                all_chunks.append({
                    "fund_name": section['fund_name'],
                    "text": chunk_text,
                    "embedding": embedding,
                    "source": "factsheet",
                    "chunk_type": "fund_details"
                })
        return all_chunks

    def process_directory(self, input_dir: str, output_dir: str):
        """Processes all PDFs in a directory and saves the chunks to a single JSON file."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        all_chunks = []
        for filename in os.listdir(input_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(input_dir, filename)
                chunks = self.process_pdf(pdf_path)
                # Add the source filename to each chunk for traceability
                for chunk in chunks:
                    chunk['source_file'] = filename
                all_chunks.extend(chunks)
                
        output_file = os.path.join(output_dir, "processed_chunks.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
            
        print(f"Saved {len(all_chunks)} total chunks to {output_file}")
        return all_chunks
