#!/usr/bin/env python3
"""
Database-Powered Mutual Fund Chatbot
Integrates PostgreSQL database for real-time fund data and analytics
Falls back to RAG/vector store if DB has no answer.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import re
from datetime import datetime
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import database access
from src.services.data.db_access import get_db_instance, MutualFundDB
# Import comprehensive finance chatbot
from src.services.chatbot.comprehensive_finance_chatbot import get_comprehensive_chatbot

logger = logging.getLogger(__name__)

class DatabasePoweredChatbot:
    """Enhanced chatbot that uses PostgreSQL database as primary data source, with RAG fallback"""
    
    def __init__(self):
        self.db = get_db_instance()
        # Initialize comprehensive chatbot for fallback
        self.comprehensive_chatbot = get_comprehensive_chatbot()
        # Intent detection keywords
        self.fund_fact_keywords = [
            'nav', 'aum', 'manager', 'fund manager', 'expense ratio', 'launch date',
            'benchmark', 'category', 'sub category', 'fund type', 'min investment',
            'rating', 'holdings', 'portfolio', 'returns', 'performance'
        ]
        self.analytics_keywords = [
            'compare', 'comparison', 'risk', 'volatility', 'sharpe ratio', 'beta',
            'alpha', 'drawdown', 'rolling returns', 'tracking error', 'information ratio',
            'sortino ratio', 'calmar ratio', 'best', 'top', 'worst', 'ranking'
        ]
        self.general_keywords = [
            'what is', 'define', 'explain', 'how does', 'who is', 'elss', 'equity',
            'debt', 'hybrid', 'liquid', 'ultra short', 'overnight', 'sebi', 'amfi'
        ]
    
    def detect_intent(self, query: str) -> str:
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in self.fund_fact_keywords):
            return "fund_fact"
        if any(keyword in query_lower for keyword in self.analytics_keywords):
            return "analytics"
        if any(keyword in query_lower for keyword in self.general_keywords):
            return "general"
        return "general"
    
    def extract_fund_names(self, query: str) -> List[str]:
        fund_names = []
        fund_houses = [
            'HDFC', 'Axis', 'Mirae', 'SBI', 'ICICI', 'Kotak', 'Aditya Birla',
            'Tata', 'Nippon', 'Franklin', 'DSP', 'Edelweiss', 'Invesco',
            'PGIM', 'Mahindra', 'Canara', 'Union', 'L&T', 'IDFC', 'Motilal'
        ]
        for house in fund_houses:
            if house.lower() in query.lower():
                pattern = rf'{house}[^.!?]*?(?:Fund|Scheme|Plan)'
                matches = re.findall(pattern, query, re.IGNORECASE)
                fund_names.extend(matches)
        fund_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Fund|Scheme|Plan))',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Growth|Direct|Regular))',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Large Cap|Mid Cap|Small Cap|Flexi Cap))'
        ]
        for pattern in fund_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            fund_names.extend(matches)
        fund_names = list(set(fund_names))
        fund_names = [name.strip() for name in fund_names if len(name.strip()) > 3]
        return fund_names
    
    def _comprehensive_fallback(self, query: str) -> str:
        """Comprehensive fallback using multiple data sources"""
        try:
            # Use the comprehensive chatbot which combines all data sources
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.comprehensive_chatbot.answer_query(query))
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Comprehensive fallback error: {e}")
            return f"I'm sorry, I couldn't find specific information about that. Please try rephrasing your question or ask about a different mutual fund topic."

    def answer_fund_fact_query(self, query: str, fund_names: List[str]) -> str:
        if not fund_names:
            return self._comprehensive_fallback(query)  # fallback if no fund detected
        fund_name = fund_names[0]
        try:
            fund_data = self.db.get_fund_details(fund_name=fund_name)
            if not fund_data or not fund_data.get('fund_info'):
                print(f"Fund '{fund_name}' not found in database, falling back to RAG...")
                return self._comprehensive_fallback(query)  # fallback if not found in DB
            fund_info = fund_data['fund_info']
            latest_nav = fund_data.get('latest_nav')
            ratings = fund_data.get('ratings', [])
            returns = fund_data.get('returns')
            analytics = fund_data.get('analytics')
            answer_parts = []
            answer_parts.append(f"**{fund_info['fund_name']}**")
            answer_parts.append(f"• **AMC:** {fund_info['amc_name']}")
            answer_parts.append(f"• **Category:** {fund_info['category']} - {fund_info['sub_category']}")
            answer_parts.append(f"• **Fund Type:** {fund_info['fund_type']}")
            if fund_info.get('fund_manager'):
                answer_parts.append(f"• **Fund Manager:** {fund_info['fund_manager']}")
            if fund_info.get('benchmark'):
                answer_parts.append(f"• **Benchmark:** {fund_info['benchmark']}")
            if fund_info.get('aum'):
                answer_parts.append(f"• **AUM:** ₹{fund_info['aum']:,.2f} Cr")
            if fund_info.get('expense_ratio'):
                answer_parts.append(f"• **Expense Ratio:** {fund_info['expense_ratio']:.2f}%")
            if fund_info.get('min_investment'):
                answer_parts.append(f"• **Min Investment:** ₹{fund_info['min_investment']:,.0f}")
            if latest_nav:
                answer_parts.append(f"• **Latest NAV:** ₹{latest_nav['nav_value']:.4f} (as of {latest_nav['nav_date']})")
            if returns:
                answer_parts.append("\n**Returns:**")
                if returns.get('returns_1y'):
                    answer_parts.append(f"• 1 Year: {returns['returns_1y']:.2f}%")
                if returns.get('returns_3y'):
                    answer_parts.append(f"• 3 Years: {returns['returns_3y']:.2f}%")
                if returns.get('returns_5y'):
                    answer_parts.append(f"• 5 Years: {returns['returns_5y']:.2f}%")
            if ratings:
                answer_parts.append("\n**Ratings:**")
                for rating in ratings[:3]:
                    answer_parts.append(f"• {rating['rating_agency']}: {rating['rating']}")
            if analytics:
                answer_parts.append("\n**Risk Metrics:**")
                if analytics.get('sharpe_ratio'):
                    answer_parts.append(f"• Sharpe Ratio: {analytics['sharpe_ratio']:.2f}")
                if analytics.get('volatility'):
                    answer_parts.append(f"• Volatility: {analytics['volatility']:.2f}%")
                if analytics.get('beta'):
                    answer_parts.append(f"• Beta: {analytics['beta']:.2f}")
                if analytics.get('max_drawdown'):
                    answer_parts.append(f"• Max Drawdown: {analytics['max_drawdown']:.2f}%")
            answer_parts.append(f"\n*Source: PostgreSQL Database - Updated daily*")
            return "\n".join(answer_parts)
        except Exception as e:
            logger.error(f"Error answering fund fact query: {e}")
            return self._comprehensive_fallback(query)
    
    def answer_analytics_query(self, query: str, fund_names: List[str]) -> str:
        if not fund_names:
            print("No fund names detected, falling back to comprehensive search...")
            return self._comprehensive_fallback(query)
        try:
            if len(fund_names) == 1:
                return self.answer_single_fund_analytics(fund_names[0], query)
            else:
                return self.answer_fund_comparison(fund_names, query)
        except Exception as e:
            logger.error(f"Error answering analytics query: {e}")
            return self._comprehensive_fallback(query)
    
    def answer_single_fund_analytics(self, fund_name: str, query: str) -> str:
        fund_data = self.db.get_fund_details(fund_name=fund_name)
        if not fund_data or not fund_data.get('analytics'):
            print(f"Analytics not available for '{fund_name}', falling back to comprehensive search...")
            return self._comprehensive_fallback(query)
        analytics = fund_data.get('analytics')
        returns = fund_data.get('returns')
        fund_info = fund_data['fund_info']
        answer_parts = [f"**Analytics for {fund_info['fund_name']}**"]
        if analytics:
            answer_parts.append("\n**Risk & Performance Metrics:**")
            if analytics.get('sharpe_ratio'):
                answer_parts.append(f"• Sharpe Ratio: {analytics['sharpe_ratio']:.2f}")
            if analytics.get('volatility'):
                answer_parts.append(f"• Volatility: {analytics['volatility']:.2f}%")
            if analytics.get('beta'):
                answer_parts.append(f"• Beta: {analytics['beta']:.2f}")
            if analytics.get('alpha'):
                answer_parts.append(f"• Alpha: {analytics['alpha']:.2f}%")
            if analytics.get('max_drawdown'):
                answer_parts.append(f"• Maximum Drawdown: {analytics['max_drawdown']:.2f}%")
        if returns:
            answer_parts.append("\n**Returns:**")
            if returns.get('returns_1y'):
                answer_parts.append(f"• 1 Year: {returns['returns_1y']:.2f}%")
            if returns.get('returns_3y'):
                answer_parts.append(f"• 3 Years: {returns['returns_3y']:.2f}%")
            if returns.get('returns_5y'):
                answer_parts.append(f"• 5 Years: {returns['returns_5y']:.2f}%")
        answer_parts.append(f"\n*Source: PostgreSQL Database - Updated daily*")
        return "\n".join(answer_parts)
    
    def answer_fund_comparison(self, fund_names: List[str], query: str) -> str:
        fund_codes = []
        for fund_name in fund_names:
            funds = self.db.search_fund_by_name(fund_name, 1)
            if funds:
                fund_codes.append(funds[0]['fund_code'])
        if len(fund_codes) < 2:
            return self._comprehensive_fallback(query)
        comparison_data = self.db.get_fund_comparison(fund_codes)
        if not comparison_data or not comparison_data.get('analytics'):
            return self._comprehensive_fallback(query)
        answer_parts = ["**Fund Comparison**"]
        funds_info = comparison_data.get('funds_info', [])
        navs = comparison_data.get('navs', [])
        returns = comparison_data.get('returns', [])
        analytics = comparison_data.get('analytics', [])
        for fund_info in funds_info:
            answer_parts.append(f"\n**{fund_info['fund_name']}**")
            answer_parts.append(f"• AMC: {fund_info['amc_name']}")
            answer_parts.append(f"• Category: {fund_info['category']} - {fund_info['sub_category']}")
            nav = next((n for n in navs if n['fund_code'] == fund_info['fund_code']), None)
            if nav:
                answer_parts.append(f"• Latest NAV: ₹{nav['nav_value']:.4f}")
            fund_returns = next((r for r in returns if r['fund_code'] == fund_info['fund_code']), None)
            if fund_returns:
                if fund_returns.get('returns_1y'):
                    answer_parts.append(f"• 1Y Return: {fund_returns['returns_1y']:.2f}%")
                if fund_returns.get('returns_3y'):
                    answer_parts.append(f"• 3Y Return: {fund_returns['returns_3y']:.2f}%")
            fund_analytics = next((a for a in analytics if a['fund_code'] == fund_info['fund_code']), None)
            if fund_analytics:
                if fund_analytics.get('sharpe_ratio'):
                    answer_parts.append(f"• Sharpe Ratio: {fund_analytics['sharpe_ratio']:.2f}")
                if fund_analytics.get('volatility'):
                    answer_parts.append(f"• Volatility: {fund_analytics['volatility']:.2f}%")
        answer_parts.append(f"\n*Source: PostgreSQL Database - Updated daily*")
        return "\n".join(answer_parts)
    
    def answer_general_query(self, query: str) -> str:
        return self._comprehensive_fallback(query)
    
    def answer_query(self, query: str) -> str:
        try:
            intent = self.detect_intent(query)
            fund_names = []
            if intent in ["fund_fact", "analytics"]:
                fund_names = self.extract_fund_names(query)
            if intent == "fund_fact":
                return self.answer_fund_fact_query(query, fund_names)
            elif intent == "analytics":
                return self.answer_analytics_query(query, fund_names)
            else:
                return self.answer_general_query(query)
        except Exception as e:
            logger.error(f"Error in answer_query: {e}")
            return self._comprehensive_fallback(query)

db_chatbot = None
def get_db_chatbot():
    global db_chatbot
    if db_chatbot is None:
        db_chatbot = DatabasePoweredChatbot()
    return db_chatbot

if __name__ == "__main__":
    chatbot = get_db_chatbot()
    print("\nWelcome to the Mutual Fund Chatbot (DB + RAG fallback)")
    print("Type your question and press Enter. Type 'exit' to quit.\n")
    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        answer = chatbot.answer_query(user_query)
        print(f"Bot: {answer}\n") 