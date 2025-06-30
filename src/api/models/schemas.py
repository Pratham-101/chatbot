"""Pydantic schemas for API models."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class QueryRequest(BaseModel):
    """Request model for chat queries."""
    
    text: str = Field(..., min_length=1, max_length=1000, description="User query text")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    
    @validator("text")
    def validate_text(cls, v: str) -> str:
        """Validate and clean query text."""
        v = v.strip()
        if not v:
            raise ValueError("Query text cannot be empty")
        return v


class QualityMetrics(BaseModel):
    """Response quality metrics."""
    
    accuracy: float = Field(..., ge=0, le=10, description="Accuracy score (0-10)")
    completeness: float = Field(..., ge=0, le=10, description="Completeness score (0-10)")
    clarity: float = Field(..., ge=0, le=10, description="Clarity score (0-10)")
    relevance: float = Field(..., ge=0, le=10, description="Relevance score (0-10)")
    overall_score: float = Field(..., ge=0, le=10, description="Overall quality score (0-10)")
    feedback: str = Field(..., description="Quality feedback")


class FundDetails(BaseModel):
    """Fund details extracted from response."""
    
    name: Optional[str] = Field(None, description="Fund name")
    type: Optional[str] = Field(None, description="Fund type")
    category: Optional[str] = Field(None, description="Fund category")
    aum: Optional[str] = Field(None, description="Assets Under Management")
    nav: Optional[str] = Field(None, description="Net Asset Value")
    expense_ratio: Optional[str] = Field(None, description="Expense ratio")
    fund_manager: Optional[str] = Field(None, description="Fund manager name")


class PerformanceData(BaseModel):
    """Performance data extracted from response."""
    
    one_year_return: Optional[str] = Field(None, description="1-year return")
    three_year_return: Optional[str] = Field(None, description="3-year return")
    five_year_return: Optional[str] = Field(None, description="5-year return")
    since_inception: Optional[str] = Field(None, description="Since inception return")
    benchmark: Optional[str] = Field(None, description="Benchmark")


class RiskMetrics(BaseModel):
    """Risk metrics extracted from response."""
    
    risk_level: Optional[str] = Field(None, description="Risk level")
    volatility: Optional[str] = Field(None, description="Volatility")
    sharpe_ratio: Optional[str] = Field(None, description="Sharpe ratio")
    beta: Optional[str] = Field(None, description="Beta")


class StructuredData(BaseModel):
    """Structured data extracted from response."""
    
    summary: str = Field(..., description="Summary of the response")
    key_points: List[str] = Field(default_factory=list, description="Key points")
    fund_details: FundDetails = Field(default_factory=FundDetails, description="Fund details")
    performance_data: PerformanceData = Field(default_factory=PerformanceData, description="Performance data")
    risk_metrics: RiskMetrics = Field(default_factory=RiskMetrics, description="Risk metrics")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    sources: List[str] = Field(default_factory=list, description="Information sources")
    disclaimer: str = Field(default="Standard disclaimer applies", description="Disclaimer")


class QueryResponse(BaseModel):
    """Response model for chat queries."""
    
    answer: str = Field(..., description="Generated answer")
    response_time: float = Field(..., description="Response time in seconds")
    quality_metrics: QualityMetrics = Field(..., description="Response quality metrics")
    structured_data: StructuredData = Field(..., description="Structured data")
    raw_response: str = Field(..., description="Raw LLM response")
    request_id: Optional[str] = Field(None, description="Request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Dict[str, Any] = Field(default_factory=dict, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    services: Dict[str, str] = Field(default_factory=dict, description="Service statuses")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MetricsResponse(BaseModel):
    """Metrics response model."""
    
    total_requests: int = Field(..., description="Total requests processed")
    successful_requests: int = Field(..., description="Successful requests")
    failed_requests: int = Field(..., description="Failed requests")
    average_response_time: float = Field(..., description="Average response time")
    active_sessions: int = Field(..., description="Active sessions")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChatSession(BaseModel):
    """Chat session model."""
    
    session_id: str = Field(..., description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation time")
    last_activity: datetime = Field(default_factory=datetime.utcnow, description="Last activity time")
    message_count: int = Field(default=0, description="Number of messages in session")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChatMessage(BaseModel):
    """Chat message model."""
    
    message_id: str = Field(..., description="Message identifier")
    session_id: str = Field(..., description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    query: str = Field(..., description="User query")
    response: QueryResponse = Field(..., description="Bot response")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 