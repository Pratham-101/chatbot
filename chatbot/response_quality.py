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
                if not isinstance(evaluation, dict):
                    print(f"[DEBUG] Evaluation is not a dict: {type(evaluation)} - {evaluation}")
                return ResponseQuality(
                    accuracy=float(evaluation.get("accuracy", 7.0)) if isinstance(evaluation, dict) else 7.0,
                    completeness=float(evaluation.get("completeness", 7.0)) if isinstance(evaluation, dict) else 7.0,
                    clarity=float(evaluation.get("clarity", 7.0)) if isinstance(evaluation, dict) else 7.0,
                    relevance=float(evaluation.get("relevance", 7.0)) if isinstance(evaluation, dict) else 7.0,
                    overall_score=float(evaluation.get("overall_score", 7.0)) if isinstance(evaluation, dict) else 7.0,
                    feedback=evaluation.get("feedback", "No feedback provided") if isinstance(evaluation, dict) else str(evaluation)
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
        You are a financial assistant. Given the following chatbot answer, extract and return a JSON object with these fields:
        - summary
        - key_points
        - fund_details
        - performance_data
        - risk_metrics
        - recommendations
        - sources
        - disclaimer

        RAW RESPONSE:
        {raw_response}
        """

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.3,
                max_tokens=1200,
                stream=False,
            )
            result_text = chat_completion.choices[0].message.content or ""
            try:
                structured_data = json.loads(result_text)
                if not isinstance(structured_data, dict):
                    print(f"[DEBUG] Structured data is not a dict: {type(structured_data)} - {structured_data}")
                return StructuredResponse(
                    summary=structured_data.get("summary", "Summary unavailable") if isinstance(structured_data, dict) else "Summary unavailable",
                    key_points=structured_data.get("key_points", []) if isinstance(structured_data, dict) else [],
                    fund_details=structured_data.get("fund_details", {}) if isinstance(structured_data, dict) else {},
                    performance_data=structured_data.get("performance_data", {}) if isinstance(structured_data, dict) else {},
                    risk_metrics=structured_data.get("risk_metrics", {}) if isinstance(structured_data, dict) else {},
                    recommendations=structured_data.get("recommendations", []) if isinstance(structured_data, dict) else [],
                    sources=structured_data.get("sources", []) if isinstance(structured_data, dict) else [],
                    disclaimer=structured_data.get("disclaimer", "Standard disclaimer applies.") if isinstance(structured_data, dict) else "Standard disclaimer applies."
                )
            except (json.JSONDecodeError, ValueError):
                # Fallback: try to extract sections using regex/heuristics
                import re
                def extract_section(pattern, text):
                    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                    return match.group(1).strip() if match else ""
                summary = extract_section(r"Summary:?\n(.+?)(\n\w|$)", raw_response)
                key_points = re.findall(r"- (.+)", extract_section(r"Key Points:?\n(.+?)(\n\w|$)", raw_response))
                fund_details = {}
                fd_section = extract_section(r"Fund Details:?\n(.+?)(\n\w|$)", raw_response)
                for line in fd_section.split("\n"):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        fund_details[k.strip()] = v.strip()
                performance_data = {}
                pd_section = extract_section(r"Performance Data:?\n(.+?)(\n\w|$)", raw_response)
                for line in pd_section.split("\n"):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        performance_data[k.strip()] = v.strip()
                risk_metrics = {}
                rm_section = extract_section(r"Risk Metrics:?\n(.+?)(\n\w|$)", raw_response)
                for line in rm_section.split("\n"):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        risk_metrics[k.strip()] = v.strip()
                recommendations = re.findall(r"- (.+)", extract_section(r"Recommendations:?\n(.+?)(\n\w|$)", raw_response))
                sources = re.findall(r"- (.+)", extract_section(r"Sources:?\n(.+?)(\n\w|$)", raw_response))
                disclaimer = extract_section(r"Disclaimer:?\n(.+?)(\n\w|$)", raw_response)
                return StructuredResponse(
                    summary=summary or "Summary unavailable",
                    key_points=key_points,
                    fund_details=fund_details,
                    performance_data=performance_data,
                    risk_metrics=risk_metrics,
                    recommendations=recommendations,
                    sources=sources,
                    disclaimer=disclaimer or "Standard disclaimer applies."
                )
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
        Format structured response into readable plain text (no markdown, no emojis)
        """
        formatted = f"{structured_response.summary}\n\n"
        if structured_response.key_points:
            formatted += "Key Points:\n"
            for point in structured_response.key_points:
                formatted += f"  - {point}\n"
            formatted += "\n"
        if structured_response.fund_details:
            formatted += "Fund Details:\n"
            for key, value in structured_response.fund_details.items():
                formatted += f"  {key.replace('_', ' ').title()}: {value}\n"
            formatted += "\n"
        if structured_response.performance_data:
            formatted += "Performance Data:\n"
            for key, value in structured_response.performance_data.items():
                formatted += f"  {key.replace('_', ' ').title()}: {value}\n"
            formatted += "\n"
        if structured_response.risk_metrics:
            formatted += "Risk Metrics:\n"
            for key, value in structured_response.risk_metrics.items():
                formatted += f"  {key.replace('_', ' ').title()}: {value}\n"
            formatted += "\n"
        if structured_response.recommendations:
            formatted += "Recommendations:\n"
            for rec in structured_response.recommendations:
                formatted += f"  - {rec}\n"
            formatted += "\n"
        if structured_response.sources:
            formatted += "Sources:\n"
            for source in structured_response.sources:
                formatted += f"  - {source}\n"
            formatted += "\n"
        if structured_response.disclaimer:
            formatted += f"Disclaimer:\n  {structured_response.disclaimer}\n"
        return formatted.strip()

# Global instances
response_evaluator = ResponseEvaluator()
structured_generator = StructuredResponseGenerator() 