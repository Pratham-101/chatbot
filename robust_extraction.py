#!/usr/bin/env python3
"""
Robust PDF extraction script with timeout protection and error handling
"""
import os
import sys
import json
import time
import signal
import logging
from pathlib import Path
from typing import Dict, Any
import warnings

# Suppress PDF font warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pdfminer")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add timeout protection
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def extract_with_timeout(pdf_path: str, timeout_seconds: int = 300) -> Dict[str, Any]:
    """Extract data from PDF with timeout protection"""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        from ingestion.pdf_table_extractor import extract_fund_factsheet_data
        result = extract_fund_factsheet_data(pdf_path)
        signal.alarm(0)  # Cancel the alarm
        return result
    except TimeoutError:
        logger.error(f"Timeout processing {pdf_path}")
        return {"error": "timeout", "file": pdf_path}
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        return {"error": str(e), "file": pdf_path}
    finally:
        signal.alarm(0)  # Ensure alarm is cancelled

def process_pdfs_robustly(input_dir: str, output_dir: str):
    """Process all PDFs with robust error handling"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    pdf_files = list(input_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    for i, pdf_file in enumerate(pdf_files, 1):
        logger.info(f"Processing {i}/{len(pdf_files)}: {pdf_file.name}")
        
        try:
            # Extract with timeout
            data = extract_with_timeout(str(pdf_file), timeout_seconds=300)
            
            # Save result
            output_file = output_path / f"{pdf_file.stem}.json"
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"✓ Completed: {pdf_file.name}")
            
            # Small delay to prevent overwhelming the system
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"✗ Failed to process {pdf_file.name}: {e}")
            # Save error info
            error_data = {"error": str(e), "file": pdf_file.name}
            output_file = output_path / f"{pdf_file.stem}_error.json"
            with open(output_file, 'w') as f:
                json.dump(error_data, f, indent=2)
    
    logger.info("Extraction completed!")

if __name__ == "__main__":
    input_dir = "data"
    output_dir = "extracted_json"
    
    logger.info("Starting robust PDF extraction...")
    process_pdfs_robustly(input_dir, output_dir) 