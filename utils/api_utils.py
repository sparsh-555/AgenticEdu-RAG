"""
OpenAI API Utilities for Agentic Edu-RAG System

This module provides a robust abstraction layer for OpenAI API interactions,
implementing best practices for error handling, rate limiting, token management,
and response processing. The utilities are optimized for educational applications
requiring consistent, reliable LLM interactions.

Key Features:
1. Comprehensive Error Handling: Graceful degradation for API failures
2. Token Usage Tracking: Cost monitoring and optimization
3. Response Validation: Ensuring response quality and format
4. Rate Limiting: Preventing API quota exhaustion
5. Retry Logic: Automatic recovery from transient failures
6. Context Management: Correlation across multi-agent workflows

Design Patterns:
- Adapter Pattern: Abstract OpenAI API complexity
- Strategy Pattern: Different handling for different response types
- Observer Pattern: Event-driven logging and monitoring
- Circuit Breaker: Prevent cascading failures
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types import CreateEmbeddingResponse
from pydantic import BaseModel, Field
import tiktoken

from ..config.settings import get_settings
from .logging_utils import get_logger, LogContext, EventType, create_context


class APIErrorType(Enum):
    """Types of API errors for specific handling strategies."""
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    QUOTA_EXCEEDED = "quota_exceeded"
    MODEL_OVERLOADED = "model_overloaded"
    NETWORK_ERROR = "network_error"


@dataclass
class TokenUsage:
    """Token usage tracking for cost management."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float = 0.0
    
    def __post_init__(self):
        """Calculate estimated cost based on current OpenAI pricing."""
        # Approximate costs (should be updated based on current pricing)
        prompt_cost_per_1k = 0.03  # $0.03 per 1K prompt tokens for GPT-4
        completion_cost_per_1k = 0.06  # $0.06 per 1K completion tokens
        
        self.estimated_cost = (
            (self.prompt_tokens / 1000) * prompt_cost_per_1k +
            (self.completion_tokens / 1000) * completion_cost_per_1k
        )


class APIResponse(BaseModel):
    """Standardized response wrapper for all API interactions."""
    content: str = Field(..., description="Response content")
    model: str = Field(..., description="Model used for generation")
    token_usage: Optional[TokenUsage] = Field(default=None, description="Token usage statistics")
    response_time_ms: float = Field(..., description="API response time in milliseconds")
    success: bool = Field(default=True, description="Whether request was successful")
    error_type: Optional[str] = Field(default=None, description="Error type if failed")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    request_id: Optional[str] = Field(default=None, description="OpenAI request ID for tracking")
    
    # Educational context
    educational_appropriateness: Optional[float] = Field(default=None, description="Content appropriateness score")
    complexity_level: Optional[str] = Field(default=None, description="Detected complexity level")


class RateLimiter:
    """Simple rate limiter to prevent API quota exhaustion."""
    
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        with self.lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            # If at limit, wait until oldest request is > 1 minute old
            if len(self.requests) >= self.max_requests:
                oldest_request = min(self.requests)
                sleep_time = 60 - (now - oldest_request)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Record this request
            self.requests.append(now)


class OpenAIError(Exception):
    """Custom exception for OpenAI API errors with context."""
    
    def __init__(self, message: str, error_type: APIErrorType, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.error_type = error_type
        self.original_error = original_error


def retry_on_failure(max_retries: int = 3, backoff_factor: float = 1.0):
    """Decorator for automatic retry with exponential backoff."""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    # Exponential backoff
                    sleep_time = backoff_factor * (2 ** attempt)
                    time.sleep(sleep_time)
            
            raise last_exception
        return wrapper
    return decorator


class OpenAIClient:
    """
    Enhanced OpenAI client with educational system optimizations.
    
    This client provides a robust interface for educational AI applications,
    with built-in error handling, rate limiting, cost tracking, and response
    validation specifically designed for programming education use cases.
    """
    
    def __init__(self):
        """Initialize the enhanced OpenAI client."""
        self.settings = get_settings()
        self.logger = get_logger()
        
        # Initialize OpenAI clients
        self.client = OpenAI(api_key=self.settings.openai.api_key)
        self.async_client = AsyncOpenAI(api_key=self.settings.openai.api_key)
        
        # Rate limiting and performance management
        self.rate_limiter = RateLimiter(max_requests_per_minute=55)  # Conservative limit
        self.token_encoder = tiktoken.encoding_for_model(self.settings.openai.model)
        
        # Cost tracking
        self.total_tokens_used = 0
        self.total_estimated_cost = 0.0
        self.request_count = 0
        
        # Circuit breaker for API health
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        self.circuit_breaker_timeout = 300  # 5 minutes
        self.circuit_breaker_open_time = None
        
        self.logger.log_event(
            EventType.SYSTEM_START,
            "OpenAI client initialized",
            extra_data={
                "model": self.settings.openai.model,
                "embedding_model": self.settings.openai.embedding_model
            }
        )
    
    def _check_circuit_breaker(self):
        """Check if circuit breaker allows requests."""
        if self.circuit_breaker_open_time:
            if time.time() - self.circuit_breaker_open_time > self.circuit_breaker_timeout:
                # Reset circuit breaker
                self.circuit_breaker_open_time = None
                self.consecutive_failures = 0
                self.logger.log_event(
                    EventType.SYSTEM_START,
                    "Circuit breaker reset, resuming API requests"
                )
            else:
                raise OpenAIError(
                    "Circuit breaker open - API temporarily unavailable",
                    APIErrorType.SERVER_ERROR
                )
    
    def _handle_api_error(self, error: Exception) -> OpenAIError:
        """Convert OpenAI API errors to our custom error types."""
        error_message = str(error)
        
        if "rate limit" in error_message.lower():
            return OpenAIError(error_message, APIErrorType.RATE_LIMIT, error)
        elif "authentication" in error_message.lower() or "api key" in error_message.lower():
            return OpenAIError(error_message, APIErrorType.AUTHENTICATION, error)
        elif "invalid" in error_message.lower():
            return OpenAIError(error_message, APIErrorType.INVALID_REQUEST, error)
        elif "timeout" in error_message.lower():
            return OpenAIError(error_message, APIErrorType.TIMEOUT, error)
        elif "quota" in error_message.lower():
            return OpenAIError(error_message, APIErrorType.QUOTA_EXCEEDED, error)
        elif "overloaded" in error_message.lower():
            return OpenAIError(error_message, APIErrorType.MODEL_OVERLOADED, error)
        else:
            return OpenAIError(error_message, APIErrorType.SERVER_ERROR, error)
    
    def _track_success(self, token_usage: TokenUsage):
        """Track successful API call."""
        self.consecutive_failures = 0
        self.request_count += 1
        self.total_tokens_used += token_usage.total_tokens
        self.total_estimated_cost += token_usage.estimated_cost
    
    def _track_failure(self):
        """Track failed API call."""
        self.consecutive_failures += 1
        
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.circuit_breaker_open_time = time.time()
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Circuit breaker opened after {self.consecutive_failures} failures",
                level="ERROR"
            )
    
    @retry_on_failure(max_retries=3, backoff_factor=1.5)
    def create_chat_completion(self,
                              messages: List[Dict[str, str]],
                              temperature: Optional[float] = None,
                              max_tokens: Optional[int] = None,
                              context: Optional[LogContext] = None) -> APIResponse:
        """
        Create chat completion with comprehensive error handling and monitoring.
        
        Args:
            messages: List of message objects for the conversation
            temperature: Sampling temperature (uses config default if None)
            max_tokens: Maximum tokens to generate (uses config default if None)
            context: Logging context for correlation
            
        Returns:
            APIResponse with content and metadata
            
        Raises:
            OpenAIError: If API request fails after retries
        """
        start_time = time.time()
        context = context or create_context()
        
        # Check circuit breaker
        self._check_circuit_breaker()
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Prepare request parameters
        request_params = {
            "model": self.settings.openai.model,
            "messages": messages,
            "temperature": temperature or self.settings.openai.temperature,
            "max_tokens": max_tokens or self.settings.openai.max_tokens
        }
        
        try:
            # Log request
            self.logger.log_event(
                EventType.LLM_REQUEST,
                "Sending chat completion request",
                context=context,
                extra_data={
                    "model": request_params["model"],
                    "message_count": len(messages),
                    "temperature": request_params["temperature"]
                }
            )
            
            # Make API call
            completion: ChatCompletion = self.client.chat.completions.create(**request_params)
            
            # Process response
            content = completion.choices[0].message.content or ""
            
            # Extract token usage
            usage = completion.usage
            token_usage = TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0
            )
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Create response
            response = APIResponse(
                content=content,
                model=completion.model,
                token_usage=token_usage,
                response_time_ms=response_time_ms,
                request_id=getattr(completion, 'id', None)
            )
            
            # Add educational metadata
            response.educational_appropriateness = self._assess_educational_appropriateness(content)
            response.complexity_level = self._assess_complexity_level(content)
            
            # Track success
            self._track_success(token_usage)
            
            # Log response
            self.logger.log_event(
                EventType.LLM_RESPONSE,
                "Chat completion successful",
                context=context,
                extra_data={
                    "response_length": len(content),
                    "tokens_used": token_usage.total_tokens,
                    "response_time_ms": response_time_ms,
                    "estimated_cost": token_usage.estimated_cost
                }
            )
            
            return response
            
        except Exception as e:
            self._track_failure()
            
            # Convert to our error type
            openai_error = self._handle_api_error(e)
            
            # Log error
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Chat completion failed: {openai_error.error_type.value}",
                context=context,
                level="ERROR",
                extra_data={
                    "error_type": openai_error.error_type.value,
                    "error_message": str(openai_error)
                }
            )
            
            # Return error response instead of raising
            return APIResponse(
                content="I apologize, but I'm experiencing technical difficulties. Please try again.",
                model=self.settings.openai.model,
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_type=openai_error.error_type.value,
                error_message=str(openai_error)
            )
    
    @retry_on_failure(max_retries=2, backoff_factor=1.0)
    def create_embeddings(self,
                         texts: Union[str, List[str]],
                         context: Optional[LogContext] = None) -> List[List[float]]:
        """
        Create embeddings for text(s) with error handling.
        
        Args:
            texts: Single text or list of texts to embed
            context: Logging context for correlation
            
        Returns:
            List of embedding vectors
            
        Raises:
            OpenAIError: If embedding creation fails
        """
        start_time = time.time()
        context = context or create_context()
        
        # Check circuit breaker
        self._check_circuit_breaker()
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Ensure input is a list
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Log request
            self.logger.log_event(
                EventType.LLM_REQUEST,
                "Creating embeddings",
                context=context,
                extra_data={
                    "model": self.settings.openai.embedding_model,
                    "text_count": len(texts),
                    "total_characters": sum(len(text) for text in texts)
                }
            )
            
            # Make API call
            response: CreateEmbeddingResponse = self.client.embeddings.create(
                model=self.settings.openai.embedding_model,
                input=texts
            )
            
            # Extract embeddings
            embeddings = [embedding.embedding for embedding in response.data]
            
            # Track usage
            if response.usage:
                token_usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=0,
                    total_tokens=response.usage.total_tokens
                )
                self._track_success(token_usage)
            
            # Log success
            response_time_ms = (time.time() - start_time) * 1000
            self.logger.log_event(
                EventType.LLM_RESPONSE,
                "Embeddings created successfully",
                context=context,
                extra_data={
                    "embedding_count": len(embeddings),
                    "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                    "response_time_ms": response_time_ms
                }
            )
            
            return embeddings
            
        except Exception as e:
            self._track_failure()
            
            openai_error = self._handle_api_error(e)
            
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Embedding creation failed: {openai_error.error_type.value}",
                context=context,
                level="ERROR"
            )
            
            raise openai_error
    
    def _assess_educational_appropriateness(self, content: str) -> float:
        """
        Assess educational appropriateness of content.
        
        Args:
            content: Content to assess
            
        Returns:
            Appropriateness score from 0.0 to 1.0
        """
        # Simple heuristic assessment
        score = 1.0
        
        # Check for inappropriate content indicators
        inappropriate_indicators = [
            "hack", "cheat", "plagiarize", "copy paste",
            "do my homework", "write my code"
        ]
        
        content_lower = content.lower()
        for indicator in inappropriate_indicators:
            if indicator in content_lower:
                score -= 0.2
        
        # Positive indicators
        positive_indicators = [
            "understand", "learn", "practice", "explain",
            "step by step", "concept", "principle"
        ]
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in content_lower)
        score += min(positive_count * 0.1, 0.3)
        
        return max(0.0, min(1.0, score))
    
    def _assess_complexity_level(self, content: str) -> str:
        """
        Assess complexity level of content.
        
        Args:
            content: Content to assess
            
        Returns:
            Complexity level: "beginner", "intermediate", or "advanced"
        """
        # Simple heuristic based on technical terms and concepts
        advanced_terms = [
            "algorithm", "complexity", "recursion", "dynamic programming",
            "data structure", "optimization", "inheritance", "polymorphism",
            "design pattern", "concurrency", "threading"
        ]
        
        intermediate_terms = [
            "function", "class", "loop", "condition", "array",
            "list", "dictionary", "object", "method", "parameter"
        ]
        
        beginner_terms = [
            "variable", "print", "input", "if", "else",
            "basic", "simple", "start", "begin", "first"
        ]
        
        content_lower = content.lower()
        
        advanced_count = sum(1 for term in advanced_terms if term in content_lower)
        intermediate_count = sum(1 for term in intermediate_terms if term in content_lower)
        beginner_count = sum(1 for term in beginner_terms if term in content_lower)
        
        if advanced_count >= 2:
            return "advanced"
        elif intermediate_count >= 3 or (intermediate_count >= 1 and advanced_count >= 1):
            return "intermediate"
        else:
            return "beginner"
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the model's tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        try:
            return len(self.token_encoder.encode(text))
        except Exception:
            # Fallback: approximate 4 characters per token
            return len(text) // 4
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate cost for token usage.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            
        Returns:
            Estimated cost in USD
        """
        token_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
        return token_usage.estimated_cost
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        return {
            "total_requests": self.request_count,
            "total_tokens_used": self.total_tokens_used,
            "total_estimated_cost": self.total_estimated_cost,
            "average_tokens_per_request": (
                self.total_tokens_used / self.request_count 
                if self.request_count > 0 else 0
            ),
            "consecutive_failures": self.consecutive_failures,
            "circuit_breaker_status": (
                "open" if self.circuit_breaker_open_time else "closed"
            )
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.total_tokens_used = 0
        self.total_estimated_cost = 0.0
        self.request_count = 0
        self.consecutive_failures = 0
        self.circuit_breaker_open_time = None


# Global client instance
_openai_client: Optional[OpenAIClient] = None


def get_openai_client(reload: bool = False) -> OpenAIClient:
    """
    Get global OpenAI client instance (singleton pattern).
    
    Args:
        reload: Force creation of new client instance
        
    Returns:
        OpenAI client instance
    """
    global _openai_client
    if _openai_client is None or reload:
        _openai_client = OpenAIClient()
    return _openai_client


if __name__ == "__main__":
    # API utilities test
    try:
        client = get_openai_client()
        
        # Test chat completion
        test_messages = [
            {"role": "system", "content": "You are a helpful programming tutor."},
            {"role": "user", "content": "Explain what a binary search algorithm is."}
        ]
        
        response = client.create_chat_completion(test_messages)
        print(f"Response: {response.content[:100]}...")
        print(f"Tokens used: {response.token_usage.total_tokens if response.token_usage else 'N/A'}")
        print(f"Response time: {response.response_time_ms:.1f}ms")
        print(f"Educational appropriateness: {response.educational_appropriateness}")
        print(f"Complexity level: {response.complexity_level}")
        
        # Test embeddings
        test_texts = ["Binary search algorithm", "Python programming"]
        embeddings = client.create_embeddings(test_texts)
        print(f"Embeddings created: {len(embeddings)} vectors of dimension {len(embeddings[0])}")
        
        # Usage stats
        stats = client.get_usage_stats()
        print(f"Usage stats: {stats}")
        
        print("✅ API utilities test completed successfully!")
        
    except Exception as e:
        print(f"❌ API utilities test failed: {e}")
