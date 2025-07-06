import re
from typing import Optional, List, Tuple

# Add fuzzywuzzy and spacy imports
from fuzzywuzzy import process
import spacy

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def extract_fund_name_candidates(query: str) -> List[str]:
    """Extract candidate fund names from a query using regex and NER."""
    nlp = get_nlp()
    doc = nlp(query)
    candidates = set()
    # Use NER to find ORG or PRODUCT entities as fund names
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"]:
            candidates.add(ent.text.strip())
    # Use noun chunks ending with fund-related suffixes
    for chunk in doc.noun_chunks:
        text = chunk.text.strip()
        if any(text.lower().endswith(suffix) for suffix in ["fund", "scheme", "tax saver", "elss"]):
            candidates.add(text)
    # Regex fallback for common patterns
    patterns = [
        r'(HDFC.*?Fund)',
        r'(ICICI.*?Fund)',
        r'(SBI.*?Fund)',
        r'(Kotak.*?Fund)',
        r'(Nippon.*?Fund)'
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            candidates.add(match.group(1))
    # If no candidates, fallback to the whole query
    if not candidates:
        candidates.add(query.strip())
    return list(candidates)

def match_fund_name(query: str, fund_names: List[str], min_score: int = 60) -> Tuple[Optional[str], List[Tuple[str, int]]]:
    """
    Match the query to the best fund name from a list using NER and fuzzy matching.
    Returns (best_match, close_matches) where close_matches is a list of (name, score).
    """
    candidates = extract_fund_name_candidates(query)
    # Fuzzy match each candidate against the fund_names
    best_match = None
    best_score = 0
    close_matches = []
    for candidate in candidates:
        match, score = process.extractOne(candidate, fund_names)
        if score > best_score:
            best_score = score
            best_match = match
        if score >= min_score:
            close_matches.append((match, score))
    # If no good match, try fuzzy matching the whole query
    if not close_matches:
        match, score = process.extractOne(query, fund_names)
        if score >= min_score:
            close_matches.append((match, score))
            if score > best_score:
                best_score = score
                best_match = match
    return (best_match, close_matches)
