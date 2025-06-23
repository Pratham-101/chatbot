from typing import List, Optional
from sentence_transformers import SentenceTransformer
import re

class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def get_relevant_context(self, query: str, k: int = 3) -> List[str]:
        """Hybrid search combining vector and keyword matches"""
        # Vector similarity search with more lenient threshold
        query_embedding = self.embedding_model.encode(query).tolist()
        vector_results = self.vector_store.query(
            query_embedding, 
            k=k*3,  # Get more results initially for filtering
            score_threshold=0.1  # Much lower threshold to catch more relevant chunks
        )
        
        # Keyword boost - look for exact or fuzzy fund name matches
        fund_name = self.extract_fund_name(query)
        if fund_name:
            keyword_results = self.vector_store.query(
                self.embedding_model.encode(fund_name).tolist(),
                k=k*3  # increase k for keyword search to get more results
            )
            # Combine and filter results
            combined = vector_results + keyword_results
            # Filter by relevance score and deduplicate
            seen_texts = set()
            filtered = []
            for item in combined:
                if item['text'] not in seen_texts and item['score'] >= 0.1:  # Much lower threshold
                    seen_texts.add(item['text'])
                    filtered.append(item)
            # Sort by score (highest first)
            filtered.sort(key=lambda x: x['score'], reverse=True)
            print(f"[Retriever] Retrieved {len(filtered[:k])} relevant context chunks for query: '{query}'")
            for idx, chunk in enumerate(filtered[:k]):
                print(f"Chunk {idx+1} (score: {chunk['score']:.2f}): {chunk['text'][:200]}...")
            return [result["text"] for result in filtered[:k]]
            
        # More lenient thresholds for all queries
        filtered_results = [r for r in vector_results if r['score'] >= 0.1]  # Much lower threshold
        if not filtered_results and vector_results:
            # If nothing matches, return top results regardless of score
            filtered_results = sorted(vector_results, key=lambda x: x['score'], reverse=True)[:k]
        
        print(f"[Retriever] Retrieved {len(filtered_results[:k])} relevant context chunks for query: '{query}'")
        if filtered_results:
            print(f"Top result score: {filtered_results[0]['score']:.2f}")
            print(f"Top result text: {filtered_results[0]['text'][:200]}...")
        for idx, chunk in enumerate(filtered_results[:k]):
            print(f"Chunk {idx+1} (score: {chunk['score']:.2f}): {chunk['text'][:200]}...")
        return [result["text"] for result in filtered_results[:k]]
    
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
                        pattern = r"((HDFC\s+(?:Large\s+and\s+Mid\s+Cap|Small\s+Cap|Multi\s+Cap|Flexi\s+Cap|Focused\s+Equity|Hybrid\s+Equity|Balanced\s+Advantage|Arbitrage|Liquid|Overnight|Short\s+Term|Medium\s+Term|Long\s+Term|Corporate\s+Bond|Banking\s+and\s+PSU|Gilt|Dynamic\s+Bond|Credit\s+Risk|Retirement|Tax\s+Saver|ELSS|Children's|Index|Top\s+100|Midcap\s+Opportunities|Growth|Value|Infrastructure|Banking|Large\s+Cap|Mid\s+Cap|Equity|Debt|Hybrid|Balanced)\s+Fund)|(HDFC\s+[A-Za-z0-9& ,.-]+?(?:Fund|Scheme|Plan|Opportunities|Mutual Fund|Investment))|(mutual fund|investment scheme))"
                        matches = re.findall(pattern, text_data, re.IGNORECASE)
                        for match_group in matches:
                            for match in match_group:
                                if match:  # Skip empty matches
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
            if score and score > 40:  # Even lower fuzzy matching threshold
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

        if best_score > 30:  # Very low fuzzy matching threshold
            print(f"[Fund Extraction] Found match for fund with score {best_score}: {best_candidate}")
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
            print(f"[Debug] Checking for fund manager of: {fund_name}")
            print(f"[Debug] Current chunk (first 200 chars): {chunk[:200]}")
            condition_result = fund_name.lower() in chunk.lower()
            print(f"[Debug] Fund name match in chunk: {condition_result}")
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
            if match and match[1] > 60:
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
