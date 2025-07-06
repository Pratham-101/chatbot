import pdfplumber
import re
import os
from typing import Dict, List, Any
import difflib
import json
from collections import defaultdict
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
try:
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    import torch
    DONUT_AVAILABLE = True
except ImportError:
    DONUT_AVAILABLE = False
try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False

# Canonical field mapping for normalization
FIELD_ALIASES = {
    'fund_manager': ['fund manager', 'manager', 'fund mgr', 'fund managers'],
    'aum': ['aum', 'assets under management'],
    'nav': ['nav', 'net asset value'],
    'inception_date': ['inception date', 'launched on', 'launch date'],
    'risk': ['risk', 'riskometer'],
    'portfolio': ['portfolio'],
    'scheme_code': ['scheme code', 'code'],
    'ratings': ['rating', 'ratings'],
    'benchmark': ['benchmark'],
    'exit_load': ['exit load'],
    'min_investment': ['minimum investment', 'min investment', 'minimum amount'],
    'contact': ['contact', 'contact details', 'contact info'],
    # Add more as needed
}

# Reverse mapping for quick lookup
FIELD_LOOKUP = {alias: canonical for canonical, aliases in FIELD_ALIASES.items() for alias in aliases}

# Helper to normalize field names
def normalize_field(field):
    field = field.lower().strip(':').strip()
    for alias, canonical in FIELD_LOOKUP.items():
        if alias in field:
            return canonical
    return field.replace(' ', '_')

# Advanced key-value extraction from text
def extract_key_values(text):
    key_values = {}
    # Look for patterns like "Key: Value" or "Key - Value"
    for line in text.split('\n'):
        match = re.match(r'([\w\s\-/().]+)[:\-]\s*(.+)', line)
        if match:
            key = normalize_field(match.group(1))
            value = match.group(2).strip()
            key_values[key] = value
    return key_values

# Improved table parsing: try to convert to list of dicts if header row detected
def parse_table(table):
    if not table or len(table) < 2:
        return {'raw': table}
    header = [normalize_field(h) for h in table[0]]
    rows = []
    for row in table[1:]:
        if len(row) == len(header):
            rows.append(dict(zip(header, row)))
        else:
            rows.append({'raw': row})
    return rows

def extract_fund_factsheet_data(pdf_path: str) -> Dict[str, Any]:
    """
    Extracts structured data from a mutual fund factsheet PDF.
    Returns a dictionary with keys: fund_manager (list of dicts), inception_date, aum, nav, portfolio, etc.
    """
    data = {
        "fund_manager": [],
        "inception_date": None,
        "aum": None,
        "nav": {},
        "portfolio": [],
        "other": {}
    }
    # Try to infer the main fund name from the PDF filename
    fund_name_from_file = os.path.splitext(os.path.basename(pdf_path))[0].replace("-", " ").replace("_", " ").lower()
    fund_managers_by_fund = {}
    current_fund = None
    last_fund_candidate = None
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables):
                if not table or not table[0]:
                    continue
                # Detect section header: single cell, uppercase, contains 'FUND'
                if len(table[0]) == 1 and table[0][0] and table[0][0].isupper() and 'FUND' in table[0][0]:
                    current_fund = table[0][0].strip()
                    print(f"[DEBUG][SECTION] Page {page_num+1} Table {t_idx+1}: Section header detected: {current_fund}")
                # Fallback: multi-cell table with uppercase fund name
                elif any(cell and cell.isupper() and 'FUND' in cell for cell in table[0]):
                    last_fund_candidate = [cell for cell in table[0] if cell and cell.isupper() and 'FUND' in cell][0]
                    print(f"[DEBUG][FALLBACK] Page {page_num+1} Table {t_idx+1}: Fund candidate detected: {last_fund_candidate}")
                header = [str(cell).strip().lower() if cell else "" for cell in table[0]]
                if any("fund manager" in h for h in header):
                    name_idx = next((i for i, h in enumerate(header) if "fund manager" in h), None)
                    since_idx = next((i for i, h in enumerate(header) if "since" in h), None)
                    exp_idx = next((i for i, h in enumerate(header) if "exp" in h), None)
                    fund_idx = next((i for i, h in enumerate(header) if "fund name" in h or "scheme" in h), None)
                    for row in table[1:]:
                        row = [str(cell).strip() if cell else "" for cell in row]
                        fund_cell = row[fund_idx].lower() if fund_idx is not None else ""
                        # Use fuzzy matching to compare fund name
                        match_score = difflib.SequenceMatcher(None, fund_name_from_file, fund_cell).ratio() if fund_cell else 0
                        if match_score > 0.6:
                            if match_score > 0.6:
                                manager_name = row[name_idx] if name_idx is not None else None
                                if manager_name:
                                    manager_name = re.sub(r"\b(hdfc|fund|scheme|co-managed|co manager|co\-manager|co\s*manager|\(.*?\))\b", "", manager_name, flags=re.I).strip()
                                since = row[since_idx] if since_idx is not None else None
                                exp = row[exp_idx] if exp_idx is not None else None
                                if manager_name:
                                    fund_managers_by_fund.setdefault(current_fund, []).append({
                                        "name": manager_name,
                                        "since": since,
                                        "experience": exp
                                    })
                elif any("co-managed by" in h for h in header):
                    mgr_idx = next((i for i, h in enumerate(header) if "co-managed by" in h), None)
                    for row in table[1:]:
                        row = [str(cell).strip() if cell else "" for cell in row]
                        manager_names = row[mgr_idx] if mgr_idx is not None else ""
                        if manager_names:
                            names = re.split(r"\s*&\s*|\s*,\s*|\s+and\s+", manager_names)
                            for name in names:
                                name = name.strip()
                                if name:
                                    # Use current_fund if available, else fallback to last_fund_candidate
                                    fund_key = current_fund or last_fund_candidate or "UNKNOWN_FUND"
                                    print(f"[DEBUG][MANAGER] Page {page_num+1} Table {t_idx+1}: Assigning manager '{name}' to fund '{fund_key}'")
                                    fund_managers_by_fund.setdefault(fund_key, []).append({
                                        "name": name,
                                        "since": None,
                                        "experience": None
                                    })
            # --- NEW: Raw text regex extraction ---
            raw_text = page.extract_text() or ""
            # Pattern: <FUND NAME> (Co-managed scheme) Since <date>
            co_managed_pattern = re.compile(r"([A-Z][A-Z\s]+FUND).*?Co-managed scheme.*?Since ([A-Za-z0-9, ]+)", re.DOTALL)
            for match in co_managed_pattern.finditer(raw_text):
                fund = match.group(1).strip()
                since = match.group(2).strip()
                # Try to find manager names in the preceding lines
                lines = raw_text.split("\n")
                idx = [i for i, l in enumerate(lines) if fund in l]
                managers = []
                if idx:
                    for i in range(max(0, idx[0]-3), idx[0]+1):
                        m = re.findall(r"Mr\.\s*([A-Za-z ]+)", lines[i])
                        if m:
                            managers.extend([x.strip() for x in m])
                        # Also try: 'CO-MANAGED BY <names>'
                        m2 = re.findall(r"CO-MANAGED BY ([A-Za-z &]+)", lines[i], re.IGNORECASE)
                        if m2:
                            managers.extend([x.strip() for x in m2])
                if fund and managers:
                    fund_managers_by_fund.setdefault(fund, []).append({"managers": managers, "since": since})
            # Pattern: Managed by <names> Since <date>
            managed_pattern = re.compile(r"Managed by ([A-Za-z &]+) Since ([A-Za-z0-9, ]+)", re.IGNORECASE)
            for match in managed_pattern.finditer(raw_text):
                managers = [x.strip() for x in match.group(1).split('&')]
                since = match.group(2).strip()
                # Try to infer fund name from nearby lines
                lines = raw_text.split("\n")
                idx = [i for i, l in enumerate(lines) if match.group(0) in l]
                fund = None
                if idx and idx[0] > 0:
                    for j in range(idx[0]-1, max(0, idx[0]-5), -1):
                        if 'FUND' in lines[j] and lines[j].isupper():
                            fund = lines[j].strip()
                            break
                if fund and managers:
                    fund_managers_by_fund.setdefault(fund, []).append({"managers": managers, "since": since})
            # --- OCR fallback ---
            if OCR_AVAILABLE and not fund_managers_by_fund:
                try:
                    page_image = page.to_image(resolution=300)
                    pil_img = page_image.original
                    ocr_text = pytesseract.image_to_string(pil_img)
                    # Use same regex as before on ocr_text
                    co_managed_pattern = re.compile(r"([A-Z][A-Z\s]+FUND).*?Co-managed scheme.*?Since ([A-Za-z0-9, ]+)", re.DOTALL)
                    for match in co_managed_pattern.finditer(ocr_text):
                        fund = match.group(1).strip()
                        since = match.group(2).strip()
                        lines = ocr_text.split("\n")
                        idx = [i for i, l in enumerate(lines) if fund in l]
                        managers = []
                        if idx:
                            for i in range(max(0, idx[0]-3), idx[0]+1):
                                m = re.findall(r"Mr\.\s*([A-Za-z ]+)", lines[i])
                                if m:
                                    managers.extend([x.strip() for x in m])
                                m2 = re.findall(r"CO-MANAGED BY ([A-Za-z &]+)", lines[i], re.IGNORECASE)
                                if m2:
                                    managers.extend([x.strip() for x in m2])
                        if fund and managers:
                            fund_managers_by_fund.setdefault(fund, []).append({"managers": managers, "since": since})
                    managed_pattern = re.compile(r"Managed by ([A-Za-z &]+) Since ([A-Za-z0-9, ]+)", re.IGNORECASE)
                    for match in managed_pattern.finditer(ocr_text):
                        managers = [x.strip() for x in match.group(1).split('&')]
                        since = match.group(2).strip()
                        lines = ocr_text.split("\n")
                        idx = [i for i, l in enumerate(lines) if match.group(0) in l]
                        fund = None
                        if idx and idx[0] > 0:
                            for j in range(idx[0]-1, max(0, idx[0]-5), -1):
                                if 'FUND' in lines[j] and lines[j].isupper():
                                    fund = lines[j].strip()
                                    break
                        if fund and managers:
                            fund_managers_by_fund.setdefault(fund, []).append({"managers": managers, "since": since})
                except Exception as e:
                    print(f"[OCR ERROR] Page {page_num+1}: {e}")
            # --- Donut fallback ---
            if DONUT_AVAILABLE and not fund_managers_by_fund:
                try:
                    page_image = page.to_image(resolution=300)
                    pil_img = page_image.original
                    donut_managers = donut_extract_fund_manager_from_image(pil_img)
                    if donut_managers:
                        # Try to infer fund name from page text or previous logic
                        text = page.extract_text() or ""
                        fund_candidates = re.findall(r"([A-Z][A-Z\s]+FUND)", text)
                        fund = fund_candidates[0].strip() if fund_candidates else os.path.splitext(os.path.basename(pdf_path))[0]
                        fund_managers_by_fund.setdefault(fund, []).append({"managers": donut_managers, "since": None})
                except Exception as e:
                    print(f"[DONUT ERROR] Page {page_num+1}: {e}")
    print(f"[DEBUG][MAPPING] Final fund manager mapping for {os.path.basename(pdf_path)}: {fund_managers_by_fund}")
    # Try to match the PDF's main fund name to the extracted mapping
    best_match = None
    best_score = 0
    for fund, managers in fund_managers_by_fund.items():
        score = difflib.SequenceMatcher(None, fund_name_from_file, fund.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = managers
    if best_match:
        data["fund_manager"] = best_match
    # Extract text for key-value pairs
    text = page.extract_text() or ""
    # Inception Date
    match = re.search(r"DATE OF ALLOTMENT/INCEPTION DATE\s*([A-Za-z]+ \d{2}, \d{4})", text)
    if match:
        data["inception_date"] = match.group(1)
    # AUM
    match = re.search(r"ASSETS UNDER MANAGEMENT.*?([₹\d,\.]+Cr)", text)
    if match:
        data["aum"] = match.group(1)
    doctr_details = doctr_extract_factsheet_details(pdf_path) if DOCTR_AVAILABLE else {}
    # Merge doctr_details into data
    for k, v in doctr_details.items():
        if k in data and isinstance(data[k], list):
            data[k].extend(v)
        else:
            data[k] = v
    return data

def donut_extract_fund_manager_from_image(pil_img):
    if not DONUT_AVAILABLE:
        return []
    try:
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
        task_prompt = "<s_docvqa><s_question>Who is the fund manager?<s_answer>"
        pixel_values = processor(pil_img, return_tensors="pt").pixel_values
        outputs = model.generate(pixel_values, decoder_input_ids=processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids)
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        # Try to extract manager names from result
        managers = re.findall(r"fund manager:?\s*([A-Za-z ,&]+)", result, re.IGNORECASE)
        return [m.strip() for m in managers if m.strip()]
    except Exception as e:
        print(f"[DONUT ERROR] {e}")
        return []

def doctr_extract_factsheet_details(pdf_path):
    if not DOCTR_AVAILABLE:
        return {}
    try:
        doc = DocumentFile.from_pdf(pdf_path)
        model = ocr_predictor(pretrained=True)
        result = model(doc)
        details = defaultdict(list)
        for page in result.pages:
            # Extract all text blocks and key-value pairs
            for block in page.blocks:
                text = block.text.strip()
                # Key-value extraction
                kv = extract_key_values(text)
                for k, v in kv.items():
                    details[k].append(v)
                # Also, try to match canonical fields directly
                for canonical, aliases in FIELD_ALIASES.items():
                    for alias in aliases:
                        if alias in text.lower():
                            details[canonical].append(text)
            # Extract and parse tables
            for table in page.tables:
                parsed = parse_table(table)
                details['tables'].append(parsed)
        # Convert defaultdict to dict
        return dict(details)
    except Exception as e:
        print(f"[DOCTR ERROR] {e}")
        return {}

# Batch process all PDFs in a directory and save as JSON
import glob
import os

def batch_extract_and_save(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = glob.glob(os.path.join(input_dir, '*.pdf'))
    for pdf_path in pdf_files:
        data = extract_fund_factsheet_data(pdf_path)
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        out_path = os.path.join(output_dir, base + '.json')
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Extracted and saved: {out_path}")

# Example usage:
# factsheet_data = extract_fund_factsheet_data("data/HDFC MF Factsheet - April 2025.pdf")
# print(factsheet_data) 