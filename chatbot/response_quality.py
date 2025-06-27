import asyncio
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from groq import Groq, APIError
import os

@dataclass
class ResponseQuality:
    """Response quality metrics"""
    accuracy: float  # 0-10
    completeness: float  # 0-10
    clarity: float  # 0-10
    relevance: float  # 0-10
    overall_score: float  # 0-10
    feedback: str

@dataclass
class StructuredResponse:
    """Structured response format"""
    summary: str
    key_points: List[str]
    fund_details: Dict
    performance_data: Dict
    risk_metrics: Dict
    recommendations: List[str]
    sources: List[str]
    disclaimer: str

class ResponseEvaluator:
    """
    Evaluates response quality using LLM-as-Judge approach
    """
    
    def __init__(self, model_name="llama3-8b-8192"):
        try:
            self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
            self.model = model_name
        except KeyError:
            print("ERROR: GROQ_API_KEY environment variable not set.")
            self.client = None

    async def evaluate_response(self, query: str, context: str, response: str) -> ResponseQuality:
        """
        Evaluate response quality using LLM-as-Judge
        """
        if not self.client:
            return ResponseQuality(
                accuracy=7.0, completeness=7.0, clarity=7.0, relevance=7.0,
                overall_score=7.0, feedback="Evaluation unavailable - API key missing"
            )

        prompt = f"""
        You are an expert evaluator of financial chatbot responses. Rate the following response on a scale of 1-10 for each criterion:

        USER QUERY: "{query}"

        CONTEXT PROVIDED: "{context[:1000]}..."

        RESPONSE: "{response}"

        EVALUATION CRITERIA:
        1. ACCURACY (1-10): Are the facts correct and consistent with the context?
        2. COMPLETENESS (1-10): Does it fully answer the user's question?
        3. CLARITY (1-10): Is the response easy to understand and well-structured?
        4. RELEVANCE (1-10): Does it stay focused on the user's query?

        Provide your evaluation in this exact JSON format:
        {{
            "accuracy": <score>,
            "completeness": <score>,
            "clarity": <score>,
            "relevance": <score>,
            "overall_score": <average>,
            "feedback": "<detailed feedback>"
        }}
        """

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.3,
                max_tokens=500,
                stream=False,
            )
            
            result_text = chat_completion.choices[0].message.content or ""
            
            # Parse JSON response
            try:
                evaluation = json.loads(result_text)
                return ResponseQuality(
                    accuracy=float(evaluation.get("accuracy", 7.0)),
                    completeness=float(evaluation.get("completeness", 7.0)),
                    clarity=float(evaluation.get("clarity", 7.0)),
                    relevance=float(evaluation.get("relevance", 7.0)),
                    overall_score=float(evaluation.get("overall_score", 7.0)),
                    feedback=evaluation.get("feedback", "No feedback provided")
                )
            except (json.JSONDecodeError, ValueError):
                return ResponseQuality(
                    accuracy=7.0, completeness=7.0, clarity=7.0, relevance=7.0,
                    overall_score=7.0, feedback="Failed to parse evaluation"
                )
                
        except APIError as e:
            print(f"Groq API error during evaluation: {e}")
            return ResponseQuality(
                accuracy=7.0, completeness=7.0, clarity=7.0, relevance=7.0,
                overall_score=7.0, feedback=f"Evaluation failed: {e}"
            )

class StructuredResponseGenerator:
    """
    Generates structured, well-formatted responses
    """
    
    def __init__(self, model_name="llama3-8b-8192"):
        try:
            self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
            self.model = model_name
        except KeyError:
            print("ERROR: GROQ_API_KEY environment variable not set.")
            self.client = None

    async def generate_structured_response(self, query: str, raw_response: str, 
                                        real_time_data: Dict = None) -> StructuredResponse:
        """
        Convert raw response into structured format
        """
        if not self.client:
            return self._fallback_structured_response(raw_response)

        # Prepare context for structured generation
        context = f"Raw Response: {raw_response}"
        if real_time_data:
            context += f"\nReal-time Data: {json.dumps(real_time_data, indent=2)}"

        prompt = f"""
        Convert the following mutual fund response into a structured format:

        USER QUERY: "{query}"
        RAW RESPONSE: "{raw_response}"

        Create a structured response with these sections:
        1. SUMMARY: 2-3 sentence overview
        2. KEY_POINTS: 3-5 bullet points of main information
        3. FUND_DETAILS: Extract fund name, type, objective, etc.
        4. PERFORMANCE_DATA: Returns, AUM, expense ratio, etc.
        5. RISK_METRICS: Risk level, volatility, etc.
        6. RECOMMENDATIONS: 2-3 actionable insights
        7. SOURCES: List of data sources used
        8. DISCLAIMER: Standard mutual fund disclaimer

        Respond in this exact JSON format:
        {{
            "summary": "<brief summary>",
            "key_points": ["<point1>", "<point2>", "<point3>"],
            "fund_details": {{
                "name": "<fund_name>",
                "type": "<fund_type>",
                "objective": "<investment_objective>",
                "category": "<fund_category>"
            }},
            "performance_data": {{
                "1_year_return": "<return>",
                "3_year_return": "<return>",
                "5_year_return": "<return>",
                "aum": "<aum>",
                "expense_ratio": "<expense_ratio>"
            }},
            "risk_metrics": {{
                "risk_level": "<risk_level>",
                "volatility": "<volatility>",
                "beta": "<beta>"
            }},
            "recommendations": ["<rec1>", "<rec2>", "<rec3>"],
            "sources": ["<source1>", "<source2>"],
            "disclaimer": "<standard_disclaimer>"
        }}
        """

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.3,
                max_tokens=1000,
                stream=False,
            )
            
            result_text = chat_completion.choices[0].message.content or ""
            
            try:
                structured_data = json.loads(result_text)
                return StructuredResponse(
                    summary=structured_data.get("summary", "Summary unavailable"),
                    key_points=structured_data.get("key_points", []),
                    fund_details=structured_data.get("fund_details", {}),
                    performance_data=structured_data.get("performance_data", {}),
                    risk_metrics=structured_data.get("risk_metrics", {}),
                    recommendations=structured_data.get("recommendations", []),
                    sources=structured_data.get("sources", []),
                    disclaimer=structured_data.get("disclaimer", "Standard disclaimer applies.")
                )
            except (json.JSONDecodeError, ValueError):
                return self._fallback_structured_response(raw_response)
                
        except APIError as e:
            print(f"Groq API error during structured generation: {e}")
            return self._fallback_structured_response(raw_response)

    def _fallback_structured_response(self, raw_response: str) -> StructuredResponse:
        """
        Fallback structured response when API is unavailable
        """
        return StructuredResponse(
            summary=raw_response[:200] + "..." if len(raw_response) > 200 else raw_response,
            key_points=["Information extracted from response"],
            fund_details={"name": "Fund details unavailable"},
            performance_data={"returns": "Performance data unavailable"},
            risk_metrics={"risk_level": "Risk assessment unavailable"},
            recommendations=["Consult a financial advisor for personalized advice"],
            sources=["Response generated from available data"],
            disclaimer="Past performance does not guarantee future results. Please consult a financial advisor."
        )

    def format_structured_response(self, structured_response: StructuredResponse) -> str:
        """
        Format structured response into readable text
        """
        formatted = f"""
# 📊 Mutual Fund Analysis

## 📋 Summary
{structured_response.summary}

## 🔑 Key Points
"""
        for point in structured_response.key_points:
            formatted += f"• {point}\n"

        formatted += f"""
## 📈 Fund Details
"""
        for key, value in structured_response.fund_details.items():
            formatted += f"**{key.replace('_', ' ').title()}:** {value}\n"

        formatted += f"""
## 📊 Performance Data
"""
        for key, value in structured_response.performance_data.items():
            formatted += f"**{key.replace('_', ' ').title()}:** {value}\n"

        formatted += f"""
## ⚠️ Risk Metrics
"""
        for key, value in structured_response.risk_metrics.items():
            formatted += f"**{key.replace('_', ' ').title()}:** {value}\n"

        formatted += f"""
## 💡 Recommendations
"""
        for rec in structured_response.recommendations:
            formatted += f"• {rec}\n"

        formatted += f"""
## 📚 Sources
"""
        for source in structured_response.sources:
            formatted += f"• {source}\n"

        formatted += f"""
---
*{structured_response.disclaimer}*
"""
        return formatted

# Global instances
response_evaluator = ResponseEvaluator()
structured_generator = StructuredResponseGenerator() 