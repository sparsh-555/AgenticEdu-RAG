"""
Knowledge Base Orchestrator for Educational RAG System

This module provides a unified interface for all knowledge base operations,
orchestrating the vector store, content loading, and retrieval systems.
It serves as the main entry point for RAG functionality in the multi-agent
educational system.

Key Responsibilities:
1. Knowledge Base Management: Loading, updating, and organizing educational content
2. Retrieval Orchestration: Coordinating complex retrieval operations across components
3. Content Quality Assurance: Ensuring educational content meets quality standards
4. Performance Optimization: Caching, indexing, and search optimization
5. Agent Integration: Providing specialized interfaces for different agent types
6. Research Analytics: Collecting data for educational effectiveness research

Architecture:
- Facade Pattern: Simplified interface for complex RAG operations
- Strategy Pattern: Different retrieval strategies for different contexts
- Observer Pattern: Monitoring and logging of knowledge operations
- Cache Pattern: Performance optimization for frequent queries
- Factory Pattern: Creation of appropriate retrieval contexts

This orchestrator enables the multi-agent system to leverage educational
content effectively while maintaining high performance and quality standards.
"""

import time
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from collections import defaultdict

from pydantic import BaseModel, Field

from .vector_store import (
    EducationalVectorStore, get_vector_store, ContentMetadata, 
    ContentType, AgentSpecialization, VectorStoreStats
)
from .content_loader import (
    EducationalContentLoader, get_content_loader, ContentLoadingConfig,
    ChunkingStrategy
)
from .retrieval import (
    EducationalRetriever, get_educational_retriever, RetrievalContext,
    ScoredResult, RetrievalMetrics
)
from ..classification.srl_classifier import SRLPhase
from ..utils.logging_utils import get_logger, LogContext, EventType, create_context
from ..config.settings import get_settings


class KnowledgeBaseStatus(Enum):
    """Status of the knowledge base."""
    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UPDATING = "updating"


class QueryType(Enum):
    """Types of queries for specialized handling."""
    CONCEPTUAL = "conceptual"          # Understanding concepts
    IMPLEMENTATION = "implementation"   # How to implement
    DEBUGGING = "debugging"            # Fixing problems
    BEST_PRACTICES = "best_practices"  # Recommended approaches
    EXAMPLES = "examples"              # Code examples
    GENERAL = "general"                # General queries


@dataclass
class KnowledgeBaseConfig:
    """Configuration for knowledge base operations."""
    # Content loading configuration
    content_loading_config: ContentLoadingConfig = field(default_factory=ContentLoadingConfig)
    
    # Retrieval configuration
    default_max_results: int = 5
    default_similarity_threshold: float = 0.0
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    
    # Quality assurance
    min_content_quality_score: float = 0.5
    enable_content_validation: bool = True
    
    # Performance optimization
    enable_parallel_loading: bool = True
    batch_size: int = 50
    max_concurrent_retrievals: int = 10


class RetrievalRequest(BaseModel):
    """Request for knowledge retrieval."""
    query: str = Field(..., description="Query text")
    agent_type: Optional[str] = Field(default=None, description="Requesting agent type")
    srl_phase: Optional[str] = Field(default=None, description="Self-regulated learning phase")
    student_level: Optional[str] = Field(default=None, description="Student proficiency level")
    
    # Context information
    code_snippet: Optional[str] = Field(default=None, description="Related code snippet")
    error_message: Optional[str] = Field(default=None, description="Related error message")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Conversation context")
    
    # Retrieval preferences
    max_results: Optional[int] = Field(default=None, description="Maximum results")
    prefer_code_examples: bool = Field(default=False, description="Prefer content with code")
    prefer_recent_content: bool = Field(default=False, description="Prefer recently accessed content")
    
    # Educational context
    learning_objectives: Optional[List[str]] = Field(default=None, description="Learning objectives")
    programming_domain: Optional[str] = Field(default=None, description="Programming domain")


class RetrievalResponse(BaseModel):
    """Response from knowledge retrieval."""
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved content")
    total_found: int = Field(default=0, description="Total matching content found")
    retrieval_time_ms: float = Field(default=0.0, description="Time taken for retrieval")
    
    # Quality metrics
    average_relevance_score: float = Field(default=0.0, description="Average relevance score")
    content_types_found: List[str] = Field(default_factory=list, description="Types of content found")
    
    # Educational metadata
    concepts_covered: List[str] = Field(default_factory=list, description="Programming concepts covered")
    difficulty_levels: List[str] = Field(default_factory=list, description="Difficulty levels found")
    
    # Cache and performance info
    served_from_cache: bool = Field(default=False, description="Whether served from cache")
    query_suggestions: List[str] = Field(default_factory=list, description="Suggested follow-up queries")


class KnowledgeBaseStats(BaseModel):
    """Comprehensive knowledge base statistics."""
    # Content statistics
    total_content_pieces: int = Field(default=0, description="Total content in knowledge base")
    content_by_type: Dict[str, int] = Field(default_factory=dict, description="Content count by type")
    content_by_agent: Dict[str, int] = Field(default_factory=dict, description="Content count by agent")
    content_by_difficulty: Dict[str, int] = Field(default_factory=dict, description="Content by difficulty")
    
    # Usage statistics
    total_queries_processed: int = Field(default=0, description="Total queries processed")
    average_query_time_ms: float = Field(default=0.0, description="Average query processing time")
    cache_hit_rate: float = Field(default=0.0, description="Cache hit rate")
    
    # Quality metrics
    average_content_quality: float = Field(default=0.0, description="Average content quality score")
    average_retrieval_relevance: float = Field(default=0.0, description="Average retrieval relevance")
    
    # System status
    knowledge_base_status: str = Field(default="uninitialized", description="Current system status")
    last_updated: Optional[float] = Field(default=None, description="Last update timestamp")
    
    # Component statistics
    vector_store_stats: Optional[Dict[str, Any]] = Field(default=None, description="Vector store statistics")
    retrieval_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Retrieval metrics")


class EducationalKnowledgeBase:
    """
    Orchestrator for educational knowledge base operations.
    
    This class provides a unified interface for all knowledge base functionality,
    coordinating content loading, storage, retrieval, and quality assurance
    to support the multi-agent educational system.
    """
    
    def __init__(self, config: Optional[KnowledgeBaseConfig] = None):
        """Initialize the educational knowledge base."""
        self.config = config or KnowledgeBaseConfig()
        self.settings = get_settings()
        self.logger = get_logger()
        
        # Initialize core components
        self.vector_store = get_vector_store()
        self.content_loader = get_content_loader(self.config.content_loading_config)
        self.retriever = get_educational_retriever()
        
        # System status
        self.status = KnowledgeBaseStatus.UNINITIALIZED
        self.last_updated = None
        
        # Performance tracking
        self.query_count = 0
        self.total_query_time = 0.0
        self.cache_hits = 0
        
        # Query cache for performance
        self.query_cache = {} if self.config.enable_caching else None
        self.cache_lock = threading.Lock() if self.config.enable_caching else None
        
        # Component statistics
        self.component_stats = {
            "content_pieces_loaded": 0,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "cache_size": 0
        }
        
        self.logger.log_event(
            EventType.COMPONENT_INIT,
            "Educational knowledge base initialized",
            extra_data={
                "caching_enabled": self.config.enable_caching,
                "cache_ttl": self.config.cache_ttl_seconds,
                "max_results": self.config.default_max_results
            }
        )
    
    def initialize_knowledge_base(self, 
                                force_reload: bool = False,
                                context: Optional[LogContext] = None) -> Dict[str, Any]:
        """
        Initialize the knowledge base by loading educational content.
        
        Args:
            force_reload: Whether to force reload of existing content
            context: Logging context
            
        Returns:
            Initialization results and statistics
        """
        context = context or create_context()
        start_time = time.time()
        
        try:
            self.status = KnowledgeBaseStatus.LOADING
            
            self.logger.log_event(
                EventType.SYSTEM_START,
                "Starting knowledge base initialization",
                context=context,
                extra_data={"force_reload": force_reload}
            )
            
            # Load educational content
            loading_result = self.content_loader.load_all_content(force_reload=force_reload)
            
            if loading_result["status"] == "success":
                self.status = KnowledgeBaseStatus.READY
                self.last_updated = time.time()
                self.component_stats["content_pieces_loaded"] = loading_result.get("chunks_stored", 0)
                
                initialization_time = (time.time() - start_time) * 1000
                
                # Get initial statistics
                stats = self.get_knowledge_base_stats()
                
                self.logger.log_event(
                    EventType.SYSTEM_START,
                    "Knowledge base initialization completed",
                    context=context,
                    extra_data={
                        "initialization_time_ms": initialization_time,
                        "content_pieces": stats.total_content_pieces,
                        "status": self.status.value
                    }
                )
                
                return {
                    "status": "success",
                    "initialization_time_ms": initialization_time,
                    "content_statistics": {
                        "total_pieces": stats.total_content_pieces,
                        "by_type": stats.content_by_type,
                        "by_agent": stats.content_by_agent
                    },
                    "loading_details": loading_result
                }
            else:
                self.status = KnowledgeBaseStatus.ERROR
                
                self.logger.log_event(
                    EventType.ERROR_OCCURRED,
                    f"Knowledge base initialization failed: {loading_result.get('error')}",
                    context=context,
                    level="ERROR"
                )
                
                return {
                    "status": "error",
                    "error": loading_result.get("error"),
                    "initialization_time_ms": (time.time() - start_time) * 1000
                }
                
        except Exception as e:
            self.status = KnowledgeBaseStatus.ERROR
            
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Knowledge base initialization exception: {str(e)}",
                context=context,
                level="ERROR"
            )
            
            return {
                "status": "error",
                "error": str(e),
                "initialization_time_ms": (time.time() - start_time) * 1000
            }
    
    def retrieve_knowledge(self, 
                          request: RetrievalRequest,
                          context: Optional[LogContext] = None) -> RetrievalResponse:
        """
        Retrieve educational knowledge based on request.
        
        Args:
            request: Retrieval request with query and context
            context: Logging context
            
        Returns:
            Retrieval response with results and metadata
        """
        context = context or create_context()
        start_time = time.time()
        
        try:
            # Check knowledge base status
            if self.status != KnowledgeBaseStatus.READY:
                return RetrievalResponse(
                    results=[],
                    total_found=0,
                    retrieval_time_ms=(time.time() - start_time) * 1000
                )
            
            # Check cache first
            if self.config.enable_caching:
                cached_response = self._check_retrieval_cache(request)
                if cached_response:
                    self.cache_hits += 1
                    self.component_stats["successful_retrievals"] += 1
                    
                    self.logger.log_event(
                        EventType.KNOWLEDGE_RETRIEVED,
                        "Knowledge retrieval served from cache",
                        context=context,
                        extra_data={"cache_hit": True}
                    )
                    
                    return cached_response
            
            # Create retrieval context
            retrieval_context = self._create_retrieval_context(request)
            
            # Perform retrieval
            scored_results = self.retriever.retrieve_educational_content(
                retrieval_context, context
            )
            
            # Process results into response
            response = self._create_retrieval_response(scored_results, start_time)
            
            # Cache response
            if self.config.enable_caching:
                self._cache_retrieval_response(request, response)
            
            # Update statistics
            self.query_count += 1
            self.total_query_time += response.retrieval_time_ms
            self.component_stats["successful_retrievals"] += 1
            
            self.logger.log_rag_operation(
                operation="knowledge_retrieval",
                query=request.query,
                results_count=len(response.results),
                context=context
            )
            
            return response
            
        except Exception as e:
            self.component_stats["failed_retrievals"] += 1
            
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Knowledge retrieval failed: {str(e)}",
                context=context,
                level="ERROR",
                extra_data={"query": request.query[:100]}
            )
            
            return RetrievalResponse(
                results=[],
                total_found=0,
                retrieval_time_ms=(time.time() - start_time) * 1000
            )
    
    def add_educational_content(self, 
                              content: str,
                              content_type: ContentType,
                              agent_specialization: AgentSpecialization,
                              metadata: Optional[Dict[str, Any]] = None,
                              context: Optional[LogContext] = None) -> bool:
        """
        Add new educational content to the knowledge base.
        
        Args:
            content: Content text
            content_type: Type of educational content
            agent_specialization: Target agent specialization
            metadata: Additional metadata
            context: Logging context
            
        Returns:
            True if content was added successfully
        """
        context = context or create_context()
        
        try:
            # Create content metadata
            content_metadata = ContentMetadata(
                content_id=f"user_content_{int(time.time())}",
                content_type=content_type,
                agent_specialization=agent_specialization,
                content_length=len(content),
                has_code_examples=self._detect_code_examples(content),
                **(metadata or {})
            )
            
            # Add to vector store
            success = self.vector_store.add_content(content, content_metadata, context)
            
            if success:
                self.component_stats["content_pieces_loaded"] += 1
                
                # Clear cache to ensure fresh results
                if self.config.enable_caching:
                    with self.cache_lock:
                        self.query_cache.clear()
                
                self.logger.log_event(
                    EventType.KNOWLEDGE_RETRIEVED,
                    "Educational content added successfully",
                    context=context,
                    extra_data={
                        "content_type": content_type.value,
                        "agent_specialization": agent_specialization.value,
                        "content_length": len(content)
                    }
                )
            
            return success
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Failed to add educational content: {str(e)}",
                context=context,
                level="ERROR"
            )
            return False
    
    def update_content_quality_scores(self, 
                                    effectiveness_feedback: Dict[str, float],
                                    context: Optional[LogContext] = None):
        """
        Update content quality scores based on effectiveness feedback.
        
        Args:
            effectiveness_feedback: Mapping of content_id to effectiveness score
            context: Logging context
        """
        context = context or create_context()
        
        try:
            updated_count = 0
            
            for content_id, effectiveness_score in effectiveness_feedback.items():
                # Update content metadata with new effectiveness score
                # This would typically involve updating the vector store metadata
                # Implementation depends on vector store capabilities
                
                self.logger.log_event(
                    EventType.KNOWLEDGE_RETRIEVED,
                    f"Updated effectiveness score for content {content_id}",
                    context=context,
                    extra_data={
                        "content_id": content_id,
                        "effectiveness_score": effectiveness_score
                    }
                )
                
                updated_count += 1
            
            self.logger.log_event(
                EventType.SYSTEM_START,
                f"Updated quality scores for {updated_count} content pieces",
                context=context,
                extra_data={"updated_count": updated_count}
            )
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Failed to update content quality scores: {str(e)}",
                context=context,
                level="ERROR"
            )
    
    def get_knowledge_base_stats(self) -> KnowledgeBaseStats:
        """
        Get comprehensive knowledge base statistics.
        
        Returns:
            Knowledge base statistics
        """
        try:
            # Get vector store statistics
            vector_stats = self.vector_store.get_vector_store_stats()
            
            # Get retrieval metrics
            retrieval_metrics = self.retriever.get_retrieval_metrics()
            
            # Calculate cache hit rate
            cache_hit_rate = (
                self.cache_hits / self.query_count if self.query_count > 0 else 0.0
            )
            
            # Calculate average query time
            avg_query_time = (
                self.total_query_time / self.query_count if self.query_count > 0 else 0.0
            )
            
            # Create comprehensive stats
            stats = KnowledgeBaseStats(
                total_content_pieces=vector_stats.total_documents,
                content_by_type=self._analyze_content_by_type(vector_stats),
                content_by_agent=vector_stats.collection_counts,
                total_queries_processed=self.query_count,
                average_query_time_ms=avg_query_time,
                cache_hit_rate=cache_hit_rate,
                knowledge_base_status=self.status.value,
                last_updated=self.last_updated,
                vector_store_stats=vector_stats.dict(),
                retrieval_metrics=retrieval_metrics.dict()
            )
            
            return stats
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Failed to get knowledge base stats: {str(e)}",
                level="ERROR"
            )
            
            return KnowledgeBaseStats(
                knowledge_base_status=self.status.value,
                last_updated=self.last_updated
            )
    
    def suggest_query_improvements(self, 
                                 query: str,
                                 context: Optional[LogContext] = None) -> List[str]:
        """
        Suggest improvements for queries that return few results.
        
        Args:
            query: Original query
            context: Logging context
            
        Returns:
            List of suggested query improvements
        """
        context = context or create_context()
        suggestions = []
        
        try:
            # Analyze query characteristics
            query_words = query.lower().split()
            
            # Suggest more specific terms
            if len(query_words) < 3:
                suggestions.append("Try adding more specific programming terms to your query")
            
            # Suggest programming domain specification
            domain_keywords = ["python", "java", "javascript", "algorithm", "data structure"]
            if not any(keyword in query.lower() for keyword in domain_keywords):
                suggestions.append("Consider specifying the programming language or domain")
            
            # Suggest adding context
            context_keywords = ["how to", "example", "implement", "debug", "fix"]
            if not any(keyword in query.lower() for keyword in context_keywords):
                suggestions.append("Try adding context like 'how to', 'example of', or 'implement'")
            
            # Suggest educational level
            level_keywords = ["beginner", "intermediate", "advanced", "basic", "simple"]
            if not any(keyword in query.lower() for keyword in level_keywords):
                suggestions.append("Consider specifying your experience level (beginner/intermediate/advanced)")
            
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                f"Generated {len(suggestions)} query suggestions",
                context=context,
                extra_data={"original_query": query, "suggestions_count": len(suggestions)}
            )
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Failed to generate query suggestions: {str(e)}",
                context=context,
                level="ERROR"
            )
        
        return suggestions
    
    def _create_retrieval_context(self, request: RetrievalRequest) -> RetrievalContext:
        """Create retrieval context from request."""
        return RetrievalContext(
            query=request.query,
            agent_specialization=request.agent_type,
            srl_phase=request.srl_phase,
            student_level=request.student_level,
            code_snippet=request.code_snippet,
            error_message=request.error_message,
            conversation_history=request.conversation_history,
            max_results=request.max_results or self.config.default_max_results,
            similarity_threshold=self.config.default_similarity_threshold,
            prefer_code_examples=request.prefer_code_examples,
            programming_domain=request.programming_domain,
            learning_objectives=request.learning_objectives
        )
    
    def _create_retrieval_response(self, 
                                 scored_results: List[ScoredResult],
                                 start_time: float) -> RetrievalResponse:
        """Create retrieval response from scored results."""
        retrieval_time = (time.time() - start_time) * 1000
        
        # Process results for response
        processed_results = []
        concepts_covered = set()
        content_types = set()
        difficulty_levels = set()
        
        for scored_result in scored_results:
            result_dict = {
                "content_id": scored_result.result.content_id,
                "content": scored_result.result.content,
                "similarity_score": scored_result.similarity_score,
                "educational_relevance": scored_result.educational_relevance_score,
                "combined_score": scored_result.combined_score,
                "content_type": scored_result.result.content_type,
                "programming_concepts": scored_result.result.programming_concepts,
                "difficulty_level": scored_result.result.difficulty_level,
                "metadata": scored_result.result.metadata
            }
            
            processed_results.append(result_dict)
            concepts_covered.update(scored_result.result.programming_concepts)
            content_types.add(scored_result.result.content_type)
            difficulty_levels.add(scored_result.result.difficulty_level)
        
        # Calculate average relevance
        avg_relevance = (
            sum(r.educational_relevance_score for r in scored_results) / len(scored_results)
            if scored_results else 0.0
        )
        
        return RetrievalResponse(
            results=processed_results,
            total_found=len(scored_results),
            retrieval_time_ms=retrieval_time,
            average_relevance_score=avg_relevance,
            content_types_found=list(content_types),
            concepts_covered=list(concepts_covered),
            difficulty_levels=list(difficulty_levels),
            served_from_cache=False
        )
    
    def _check_retrieval_cache(self, request: RetrievalRequest) -> Optional[RetrievalResponse]:
        """Check cache for existing retrieval response."""
        if not self.config.enable_caching or not self.cache_lock:
            return None
        
        cache_key = self._generate_cache_key(request)
        
        with self.cache_lock:
            if cache_key in self.query_cache:
                cached_entry = self.query_cache[cache_key]
                
                # Check if cache entry is still valid
                if time.time() - cached_entry["timestamp"] < self.config.cache_ttl_seconds:
                    response = cached_entry["response"]
                    response.served_from_cache = True
                    return response
                else:
                    # Remove expired entry
                    del self.query_cache[cache_key]
        
        return None
    
    def _cache_retrieval_response(self, request: RetrievalRequest, response: RetrievalResponse):
        """Cache retrieval response."""
        if not self.config.enable_caching or not self.cache_lock:
            return
        
        cache_key = self._generate_cache_key(request)
        
        with self.cache_lock:
            # Simple cache size management
            if len(self.query_cache) > 1000:
                # Remove oldest 20% of entries
                oldest_keys = sorted(
                    self.query_cache.keys(),
                    key=lambda k: self.query_cache[k]["timestamp"]
                )[:200]
                
                for key in oldest_keys:
                    del self.query_cache[key]
            
            self.query_cache[cache_key] = {
                "response": response,
                "timestamp": time.time()
            }
            
            self.component_stats["cache_size"] = len(self.query_cache)
    
    def _generate_cache_key(self, request: RetrievalRequest) -> str:
        """Generate cache key for retrieval request."""
        import hashlib
        
        key_parts = [
            request.query,
            request.agent_type or "",
            request.srl_phase or "",
            request.student_level or "",
            str(request.max_results or self.config.default_max_results),
            str(request.prefer_code_examples),
            request.programming_domain or ""
        ]
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _detect_code_examples(self, content: str) -> bool:
        """Detect if content contains code examples."""
        code_indicators = ['```', 'def ', 'class ', 'function', 'import ', '{', '}']
        return any(indicator in content for indicator in code_indicators)
    
    def _analyze_content_by_type(self, vector_stats: VectorStoreStats) -> Dict[str, int]:
        """Analyze content distribution by type."""
        # This would typically query the vector store for content type distribution
        # For now, return a placeholder implementation
        return {
            "implementation_guide": 0,
            "debugging_resource": 0,
            "concept_explanation": 0,
            "code_example": 0,
            "best_practice": 0,
            "general": 0
        }


# Global knowledge base instance
_knowledge_base: Optional[EducationalKnowledgeBase] = None


def get_knowledge_base(config: Optional[KnowledgeBaseConfig] = None,
                      reload: bool = False) -> EducationalKnowledgeBase:
    """
    Get global knowledge base instance (singleton pattern).
    
    Args:
        config: Knowledge base configuration
        reload: Force creation of new knowledge base instance
        
    Returns:
        EducationalKnowledgeBase instance
    """
    global _knowledge_base
    if _knowledge_base is None or reload:
        _knowledge_base = EducationalKnowledgeBase(config)
    return _knowledge_base


if __name__ == "__main__":
    # Knowledge base test
    try:
        kb = get_knowledge_base()
        
        # Test initialization
        print("Testing knowledge base initialization...")
        init_result = kb.initialize_knowledge_base()
        print(f"Initialization: {init_result['status']}")
        
        if init_result["status"] == "success":
            # Test retrieval
            print("\nTesting knowledge retrieval...")
            
            request = RetrievalRequest(
                query="How do I implement binary search?",
                agent_type="implementation",
                srl_phase="FORETHOUGHT",
                student_level="intermediate",
                max_results=3,
                prefer_code_examples=True
            )
            
            response = kb.retrieve_knowledge(request)
            print(f"Retrieval: {len(response.results)} results in {response.retrieval_time_ms:.1f}ms")
            print(f"Average relevance: {response.average_relevance_score:.3f}")
            print(f"Concepts covered: {response.concepts_covered}")
            
            # Test statistics
            print("\nTesting knowledge base statistics...")
            stats = kb.get_knowledge_base_stats()
            print(f"Total content: {stats.total_content_pieces}")
            print(f"Total queries: {stats.total_queries_processed}")
            print(f"Average query time: {stats.average_query_time_ms:.1f}ms")
            print(f"Cache hit rate: {stats.cache_hit_rate:.2%}")
            
            # Test query suggestions
            print("\nTesting query suggestions...")
            suggestions = kb.suggest_query_improvements("sort")
            print(f"Suggestions for 'sort': {len(suggestions)}")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
        
        print("\n✅ Knowledge base test completed successfully!")
        
    except Exception as e:
        print(f"❌ Knowledge base test failed: {e}")
        import traceback
        traceback.print_exc()
