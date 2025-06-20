from typing import List, Optional
from sentence_transformers import SentenceTransformer
import re

class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def get_relevant_context(self, query: str, k: int = 15) -> List[str]:
        """Hybrid search combining vector and keyword matches"""
        # Vector similarity search
        query_embedding = self.embedding_model.encode(query).tolist()
        vector_results = self.vector_store.query(query_embedding, k=k)
        
        # Keyword boost - look for exact or fuzzy fund name matches
        fund_name = self.extract_fund_name(query)
        if fund_name:
            keyword_results = self.vector_store.query(
                self.embedding_model.encode(fund_name).tolist(),
                k=k*2  # increase k for keyword search to get more results
            )
            # Combine results
            combined = vector_results + keyword_results
            # Deduplicate and sort by score
            combined = sorted(combined, key=lambda x: x['distance'])
            # Deduplicate by text content
            seen_texts = set()
            deduped = []
            for item in combined:
                if item['text'] not in seen_texts:
                    deduped.append(item)
                    seen_texts.add(item['text'])
            # Add weighting: prioritize chunks containing fund name or close match
            weighted = []
            for chunk in deduped:
                weight = 1.0
                # Use fuzzy matching to check if fund name is close to chunk text
                from fuzzywuzzy import fuzz
                if fuzz.partial_ratio(fund_name.lower(), chunk['text'].lower()) > 70:
                    weight -= 0.5  # higher priority (lower distance)
                weighted.append((weight, chunk))
            weighted = sorted(weighted, key=lambda x: x[0])
            weighted_chunks = [w[1] for w in weighted]
            print(f"[Retriever] Retrieved {len(weighted_chunks[:k])} weighted context chunks for query: {query}")
            for idx, chunk in enumerate(weighted_chunks[:k]):
                print(f"Chunk {idx+1}: {chunk['text'][:200]}...")
            return [result["text"] for result in weighted_chunks[:k]]
            
        print(f"[Retriever] Retrieved {len(vector_results)} context chunks for query: {query}")
        for idx, chunk in enumerate(vector_results):
            print(f"Chunk {idx+1}: {chunk['text'][:200]}...")
        return [result["text"] for result in vector_results]
    
    @staticmethod
    def extract_fund_name(query: str) -> Optional[str]:
        import spacy
        import json
        from fuzzywuzzy import process
        import os
        import re

        # Cache nlp model to avoid reloading every call
        if not hasattr(Retriever, "_nlp"):
            Retriever._nlp = spacy.load("en_core_web_sm")

        # Load known fund names from all JSON files in processed_data (cache in class variable)
        if not hasattr(Retriever, "_known_fund_names"):
            try:
                fund_names = set()
                processed_data_dir = "processed_data"
                for filename in os.listdir(processed_data_dir):
                    if filename.endswith(".json"):
                        file_path = os.path.join(processed_data_dir, filename)
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        text_data = " ".join(item.get("text", "") for item in data)
                        # Extract fund names by regex: words ending with Fund, Scheme, etc.
                        pattern = r"([A-Za-z0-9& ,.-]+?(?:Fund|Scheme|Tax Saver|ELSS))"
                        matches = re.findall(pattern, text_data, re.IGNORECASE)
                        for match in matches:
                            fund_names.add(match.strip())
                Retriever._known_fund_names = list(fund_names)
            except Exception as e:
                print(f"Error loading known fund names: {e}")
                Retriever._known_fund_names = []
        else:
            fund_names = Retriever._known_fund_names

        # Extract candidate fund names from query using cached spaCy NER and noun chunks
        candidates = set()

        nlp = Retriever._nlp
        doc = nlp(query)

        # Use NER to find ORG or PRODUCT entities as fund names
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT"]:
                candidates.add(ent.text.strip())

        # Use noun chunks ending with fund-related suffixes
        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            if any(text.endswith(suffix) for suffix in ["Fund", "Scheme", "Tax Saver", "ELSS"]):
                candidates.add(text)

        # If no candidates found, fallback to fuzzy matching on entire query
        if not candidates:
            best_match, score = process.extractOne(query, fund_names)
            if score and score > 70:
                return best_match
            else:
                return None

        # Use fuzzy matching to find best match among candidates against known fund names
        best_candidate = None
        best_score = 0
        for candidate in candidates:
            match, score = process.extractOne(candidate, fund_names)
            if score > best_score:
                best_score = score
                best_candidate = match

        if best_score > 70:
            return best_candidate

        return None

    def get_fund_manager(self, query: str) -> Optional[str]:
        fund_name = self.extract_fund_name(query)
        if not fund_name:
            return "Could not extract fund name from the query."

        context_chunks = self.get_relevant_context(query)
        # Regex pattern to find fund manager info in context chunks
        manager_pattern = re.compile(r"Fund Manager\s*[:\-]?\s*([A-Z][a-zA-Z\s\.\-]+)", re.IGNORECASE)
        fallback_pattern = re.compile(r"(?:fund manager|is managed by|managed by|manager is|has|managed by Mr\.?|managed by Ms\.?|managed by Mrs\.?)\s*([A-Z][a-zA-Z\s\.\-]+)", re.IGNORECASE)

        for chunk in context_chunks:
            print(f"[Debug] fund_name: {fund_name.lower()}")
            print(f"[Debug] chunk.lower(): {chunk.lower()}")
            condition_result = fund_name.lower() in chunk.lower()
            print(f"[Debug] Condition fund_name in chunk: {condition_result}")
            # Check if chunk contains the fund name
            if condition_result:
                print(f"[Debug] Checking chunk: {chunk}")
                match = manager_pattern.search(chunk)
                if not match:
                    match = fallback_pattern.search(chunk)
                if match:
                    print(f"[Debug] Regex match found: {match.group(0)}")
                    manager_name = match.group(1).strip()
                    return f"The fund manager for {fund_name} is {manager_name}."
        # If no match found, try fuzzy matching on chunks
        from fuzzywuzzy import process
        best_match = None
        best_score = 0
        for chunk in context_chunks:
            match = process.extractOne(fund_name, [chunk])
            if match and match[1] > 70:
                best_match = chunk
                best_score = match[1]
                # Try regex on best match chunk
                match_regex = manager_pattern.search(best_match)
                if not match_regex:
                    match_regex = fallback_pattern.search(best_match)
                if match_regex:
                    manager_name = match_regex.group(1).strip()
                    return f"The fund manager for {fund_name} is {manager_name}."
        print(f"[Debug] Fund manager info not found in any chunk for {fund_name}")
        return f"Fund manager information for {fund_name} not found in the factsheet."
