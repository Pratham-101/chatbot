"""Configuration management for the mutual fund chatbot."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "Mutual Fund Chatbot"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    
    # Security
    secret_key: str = Field(env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # LLM Configuration
    groq_api_key: str = Field(env="GROQ_API_KEY")
    groq_model: str = Field(default="llama3-8b-8192", env="GROQ_MODEL")
    groq_timeout: int = Field(default=30, env="GROQ_TIMEOUT")
    
    # Vector Store
    vector_store_path: str = Field(default="vector_store", env="VECTOR_STORE_PATH")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    # Web Search
    web_search_timeout: int = Field(default=10, env="WEB_SEARCH_TIMEOUT")
    web_search_max_results: int = Field(default=3, env="WEB_SEARCH_MAX_RESULTS")
    
    # Database (for future use)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # Redis (for caching and sessions)
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # Monitoring
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")
    
    # CORS
    cors_origins: list[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # File Storage
    upload_dir: str = Field(default="uploads", env="UPLOAD_DIR")
    max_file_size: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    
    @validator("secret_key", pre=True, always=True)
    def validate_secret_key(cls, v: str) -> str:
        """Ensure secret key is set and has minimum length."""
        if not v:
            raise ValueError("SECRET_KEY must be set")
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v
    
    @validator("groq_api_key", pre=True, always=True)
    def validate_groq_api_key(cls, v: str) -> str:
        """Ensure Groq API key is set."""
        if not v:
            raise ValueError("GROQ_API_KEY must be set")
        return v
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()


class DevelopmentSettings(Settings):
    """Development-specific settings."""
    debug: bool = True
    environment: str = "development"
    log_level: str = "DEBUG"


class ProductionSettings(Settings):
    """Production-specific settings."""
    debug: bool = False
    environment: str = "production"
    log_level: str = "WARNING"
    
    @validator("cors_origins")
    def validate_cors_origins_production(cls, v: list[str]) -> list[str]:
        """Ensure CORS origins are properly configured in production."""
        if "*" in v:
            raise ValueError("Wildcard CORS origins not allowed in production")
        return v


class TestingSettings(Settings):
    """Testing-specific settings."""
    debug: bool = True
    environment: str = "testing"
    log_level: str = "DEBUG"
    groq_api_key: str = "test_key"
    secret_key: str = "test_secret_key_32_characters_long"


def get_settings_by_environment() -> Settings:
    """Get settings based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings() 