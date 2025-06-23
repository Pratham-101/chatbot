import asyncio
import json
import time
from typing import List, Dict, Optional
import httpx
from .retrieval import Retriever
from .web_search import WebSearch
from .generation import ResponseGenerator

class EnhancedRAGChatbot:
    def __init__(self, model_name: str = "llama3"):
        self.retriever = Retriever(None)  # Will be set after vector store initialization
        self.web_search = WebSearch()
        self.generator = ResponseGenerator(model_name)
        self.conversation_history = []
        self.current_fund_context = ""
        
    def set_vector_store(self, vector_store):
        """Set the vector store for the retriever"""
        self.retriever.vector_store = vector_store
    
    async def generate_answer(self, query: str) -> str:
        """Generate a comprehensive answer using both factsheet and web data"""
        start_time = time.time()
        print(f"[Enhanced RAG] Processing query: {query}")
        
        try:
            # Step 1: Extract fund names and intent from query
            fund_names = self._extract_fund_names(query)
            intent = self._classify_intent(query)
            
            # Step 2: Get factsheet context
            factsheet_context = await self._get_factsheet_context(query, fund_names)
            
            # Step 3: Get web data based on intent
            web_data = await self._get_web_data(query, fund_names, intent)
            
            # Step 4: Generate comprehensive response
            response = await self._generate_comprehensive_response(
                query, factsheet_context, web_data, fund_names, intent
            )
            
            # Step 5: Update conversation history
            self._update_conversation_history(query, response)
            
            elapsed = time.time() - start_time
            return f"{response}\n\n[Response time: {elapsed:.2f} seconds]"
            
        except Exception as e:
            print(f"Error in enhanced RAG: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    def _extract_fund_names(self, query: str) -> List[str]:
        """Extract fund names from query"""
        fund_names = []
        
        # Use the retriever's fund extraction
        extracted_fund = self.retriever.extract_fund_name(query)
        if extracted_fund:
            fund_names.append(extracted_fund)
        
        # Also look for common fund patterns
        fund_patterns = [
            r'HDFC\s+[A-Za-z\s&]+?(?:Fund|Scheme)',
            r'ICICI\s+Prudential\s+[A-Za-z\s&]+?(?:Fund|Scheme)',
            r'Kotak\s+[A-Za-z\s&]+?(?:Fund|Scheme)',
            r'SBI\s+[A-Za-z\s&]+?(?:Fund|Scheme)',
            r'Nippon\s+India\s+[A-Za-z\s&]+?(?:Fund|Scheme)'
        ]
        
        for pattern in fund_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            fund_names.extend(matches)
        
        return list(set(fund_names))  # Remove duplicates
    
    def _classify_intent(self, query: str) -> str:
        """Classify the user's intent"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference']):
            return 'comparison'
        elif any(word in query_lower for word in ['performance', 'returns', 'nav', 'cagr']):
            return 'performance'
        elif any(word in query_lower for word in ['manager', 'who manages']):
            return 'fund_manager'
        elif any(word in query_lower for word in ['expense', 'ter', 'cost']):
            return 'expense_ratio'
        elif any(word in query_lower for word in ['portfolio', 'allocation', 'holdings']):
            return 'portfolio'
        elif any(word in query_lower for word in ['market', 'overview', 'trends']):
            return 'market_overview'
        else:
            return 'general_info'
    
    async def _get_factsheet_context(self, query: str, fund_names: List[str]) -> List[str]:
        """Get relevant context from factsheets"""
        try:
            if self.retriever.vector_store:
                context = self.retriever.get_relevant_context(query, k=5)
                return context
            else:
                print("Vector store not initialized")
                return []
        except Exception as e:
            print(f"Error getting factsheet context: {e}")
            return []
    
    async def _get_web_data(self, query: str, fund_names: List[str], intent: str) -> Dict:
        """Get relevant web data based on intent"""
        try:
            web_data = {}
            
            if intent == 'comparison' and len(fund_names) >= 2:
                comparison_data = await self.web_search.get_fund_comparison(
                    fund_names[0], fund_names[1]
                )
                if comparison_data:
                    web_data['comparison'] = comparison_data
            
            elif intent == 'performance' and fund_names:
                for fund_name in fund_names:
                    performance_data = await self.web_search.get_fund_performance(fund_name)
                    if performance_data:
                        web_data[f'performance_{fund_name}'] = performance_data
            
            elif intent == 'market_overview':
                market_data = await self.web_search.get_market_overview()
                if market_data:
                    web_data['market_overview'] = market_data
            
            else:
                # General search for the query
                general_results = await self.web_search.search_mutual_funds(query, 3)
                if general_results:
                    web_data['general'] = general_results
            
            return web_data
            
        except Exception as e:
            print(f"Error getting web data: {e}")
            return {}
    
    async def _generate_comprehensive_response(
        self, 
        query: str, 
        factsheet_context: List[str], 
        web_data: Dict, 
        fund_names: List[str], 
        intent: str
    ) -> str:
        """Generate a comprehensive response combining factsheet and web data"""
        
        # Build the prompt
        prompt_parts = [
            "You are an expert mutual fund advisor with access to both factsheet data and current web information.",
            f"User Query: {query}",
            "\nFactsheet Information:"
        ]
        
        if factsheet_context:
            for i, context in enumerate(factsheet_context[:3], 1):
                prompt_parts.append(f"{i}. {context[:300]}...")
        else:
            prompt_parts.append("No specific factsheet data available.")
        
        prompt_parts.append("\nCurrent Web Information:")
        
        if web_data:
            for key, data in web_data.items():
                if isinstance(data, dict) and 'data' in data:
                    for item in data['data'][:2]:
                        prompt_parts.append(f"- {item.get('title', '')}: {item.get('snippet', '')[:200]}...")
        else:
            prompt_parts.append("No additional web data available.")
        
        prompt_parts.extend([
            "\nInstructions:",
            "1. Provide a comprehensive answer combining factsheet and web data",
            "2. If factsheet data is available, prioritize it for accuracy",
            "3. Use web data to supplement with current information",
            "4. Be specific about fund names, performance metrics, and dates",
            "5. If comparing funds, provide clear comparisons",
            "6. Include relevant warnings or disclaimers when appropriate",
            "7. Structure your response clearly with headings and bullet points",
            "8. If information is not available, clearly state what's missing",
            "\nProvide your comprehensive response:"
        ])
        
        prompt = "\n".join(prompt_parts)
        
        try:
            # Generate response using the LLM
            response = await self.generator._call_ollama_async(prompt, timeout=180)
            
            # Post-process the response
            response = self._post_process_response(response, fund_names, intent)
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return self._generate_fallback_response(query, factsheet_context, web_data)
    
    def _post_process_response(self, response: str, fund_names: List[str], intent: str) -> str:
        """Post-process the generated response"""
        # Add source attribution
        sources = []
        if fund_names:
            sources.append("Factsheet data")
        if intent in ['performance', 'comparison', 'market_overview']:
            sources.append("Current web data")
        
        if sources:
            response += f"\n\nSources: {', '.join(sources)}"
        
        return response
    
    def _generate_fallback_response(self, query: str, factsheet_context: List[str], web_data: Dict) -> str:
        """Generate a fallback response when LLM fails"""
        response_parts = [f"Based on your query about '{query}':"]
        
        if factsheet_context:
            response_parts.append("\nFrom our factsheet database:")
            for context in factsheet_context[:2]:
                response_parts.append(f"- {context[:200]}...")
        
        if web_data:
            response_parts.append("\nFrom current market data:")
            for key, data in web_data.items():
                if isinstance(data, dict) and 'data' in data:
                    for item in data['data'][:1]:
                        response_parts.append(f"- {item.get('title', '')}: {item.get('snippet', '')[:200]}...")
        
        if not factsheet_context and not web_data:
            response_parts.append("\nI couldn't find specific information about this in our database or current sources.")
            response_parts.append("Please try rephrasing your question or ask about a specific mutual fund.")
        
        return "\n".join(response_parts)
    
    def _update_conversation_history(self, query: str, response: str):
        """Update conversation history"""
        self.conversation_history.append({
            'query': query,
            'response': response,
            'timestamp': time.time()
        })
        
        # Keep only last 10 interactions
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation"""
        if not self.conversation_history:
            return "No conversation history available."
        
        summary = "Recent conversation:\n"
        for i, interaction in enumerate(self.conversation_history[-5:], 1):
            summary += f"{i}. Q: {interaction['query'][:100]}...\n"
        
        return summary 