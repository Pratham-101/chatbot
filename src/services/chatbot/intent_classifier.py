"""
Intent Classification Module for Mutual Fund Chatbot
Detects user intent and extracts entities for better query processing.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class QueryIntent(Enum):
    FUND_ANALYSIS = "fund_analysis"
    COMPARE_FUNDS = "compare_funds"
    FUND_SUITABILITY = "fund_suitability"
    SECTOR_PERFORMANCE = "sector_performance"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    NAV_HISTORY = "nav_history"
    GENERAL_QUERY = "general_query"

@dataclass
class ExtractedEntities:
    fund_names: List[str]
    sectors: List[str]
    durations: List[str]
    amounts: List[float]
    comparison_keywords: List[str]

@dataclass
class QueryAnalysis:
    intent: QueryIntent
    entities: ExtractedEntities
    confidence: float
    raw_query: str

class IntentClassifier:
    """Classifies user queries and extracts relevant entities."""
    
    def __init__(self):
        # Intent patterns
        self.intent_patterns = {
            QueryIntent.FUND_ANALYSIS: [
                r"tell me about",
                r"what is",
                r"show me",
                r"analyze",
                r"fund details",
                r"fund information"
            ],
            QueryIntent.COMPARE_FUNDS: [
                r"compare",
                r"vs",
                r"versus",
                r"difference between",
                r"which is better",
                r"side by side"
            ],
            QueryIntent.FUND_SUITABILITY: [
                r"suitable for",
                r"who should invest",
                r"risk profile",
                r"investment horizon",
                r"should i invest"
            ],
            QueryIntent.SECTOR_PERFORMANCE: [
                r"sector",
                r"theme",
                r"banking funds",
                r"technology funds",
                r"manufacturing funds"
            ],
            QueryIntent.PORTFOLIO_ANALYSIS: [
                r"portfolio",
                r"my investments",
                r"diversification",
                r"allocation",
                r"holdings"
            ],
            QueryIntent.NAV_HISTORY: [
                r"nav history",
                r"performance over time",
                r"historical",
                r"chart",
                r"trend"
            ]
        }
        
        # Fund name patterns
        self.fund_patterns = [
            r'\b(HDFC|ICICI|SBI|Kotak|Nippon|Axis|Mirae|Aditya Birla|Tata|Franklin|DSP|IDFC|L&T|UTI)\s+[\w\s-]*Fund\b',
            r'\b[\w\s-]*Fund\b'
        ]
        
        # Sector patterns
        self.sector_patterns = [
            r'\b(banking|technology|pharma|healthcare|manufacturing|infrastructure|energy|consumer|auto|real estate)\b',
            r'\b(large cap|mid cap|small cap|multi cap|flexi cap)\b'
        ]
        
        # Duration patterns
        self.duration_patterns = [
            r'\b(1 year|3 year|5 year|1y|3y|5y|long term|short term)\b'
        ]
        
        # Amount patterns
        self.amount_patterns = [
            r'\b(\d+)\s*(lakh|lac|cr|crore|thousand|k)\b',
            r'\b₹?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\b'
        ]

    def classify_intent(self, query: str) -> QueryAnalysis:
        """Classify the intent of a user query."""
        query_lower = query.lower()
        
        # Calculate confidence scores for each intent
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            intent_scores[intent] = score
        
        # Find the intent with highest score
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            confidence = min(best_intent[1] / 2, 1.0)  # Normalize confidence
        else:
            best_intent = (QueryIntent.GENERAL_QUERY, 0)
            confidence = 0.5
        
        # Extract entities
        entities = self._extract_entities(query)
        
        return QueryAnalysis(
            intent=best_intent[0],
            entities=entities,
            confidence=confidence,
            raw_query=query
        )
    
    def _extract_entities(self, query: str) -> ExtractedEntities:
        """Extract entities from the query."""
        fund_names = self._extract_fund_names(query)
        sectors = self._extract_sectors(query)
        durations = self._extract_durations(query)
        amounts = self._extract_amounts(query)
        comparison_keywords = self._extract_comparison_keywords(query)
        
        return ExtractedEntities(
            fund_names=fund_names,
            sectors=sectors,
            durations=durations,
            amounts=amounts,
            comparison_keywords=comparison_keywords
        )
    
    def _extract_fund_names(self, query: str) -> List[str]:
        """Extract fund names from query."""
        fund_names = []
        for pattern in self.fund_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            fund_names.extend(matches)
        return list(set(fund_names))
    
    def _extract_sectors(self, query: str) -> List[str]:
        """Extract sector/theme mentions."""
        sectors = []
        for pattern in self.sector_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            sectors.extend(matches)
        return list(set(sectors))
    
    def _extract_durations(self, query: str) -> List[str]:
        """Extract time duration mentions."""
        durations = []
        for pattern in self.duration_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            durations.extend(matches)
        return list(set(durations))
    
    def _extract_amounts(self, query: str) -> List[float]:
        """Extract monetary amounts."""
        amounts = []
        for pattern in self.amount_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        amount_str = match[0]
                    else:
                        amount_str = match
                    
                    # Convert to float
                    amount_str = amount_str.replace(',', '')
                    amount = float(amount_str)
                    
                    # Handle units
                    if 'lakh' in query.lower() or 'lac' in query.lower():
                        amount *= 100000
                    elif 'cr' in query.lower() or 'crore' in query.lower():
                        amount *= 10000000
                    elif 'k' in query.lower() or 'thousand' in query.lower():
                        amount *= 1000
                    
                    amounts.append(amount)
                except ValueError:
                    continue
        return amounts
    
    def _extract_comparison_keywords(self, query: str) -> List[str]:
        """Extract comparison-related keywords."""
        comparison_keywords = ['compare', 'vs', 'versus', 'difference', 'better', 'worse']
        found_keywords = []
        for keyword in comparison_keywords:
            if keyword in query.lower():
                found_keywords.append(keyword)
        return found_keywords

# Global instance
intent_classifier = IntentClassifier() 