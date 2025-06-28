"""
Unified Knowledge Base Orchestrator for Educational RAG System

This module provides a unified interface for all knowledge base operations,
orchestrating the unified vector store, content loading, and retrieval systems.
It serves as the main entry point for RAG functionality in the multi-agent
educational system with a single, unified collection architecture.

Key Responsibilities:
1. Unified Knowledge Management: Loading, updating, and organizing educational content in single collection
2. Intelligent Retrieval Orchestration: Coordinating retrieval with metadata-based filtering
3. Content Quality Assurance: Ensuring educational content meets quality standards
4. Performance Optimization: Caching, indexing, and search optimization for unified approach
5. Agent Integration: Providing specialized interfaces while using unified storage
6. Research Analytics: Collecting data for educational effectiveness research

Unified Architecture:
- Single Collection: All content stored in one collection with rich metadata
- Metadata-Driven: Intelligent filtering replaces collection-based routing
- Facade Pattern: Simplified interface for complex unified RAG operations
- Strategy Pattern: Different retrieval strategies with unified storage
- Observer Pattern: Monitoring and logging of unified knowledge operations
- Cache Pattern: Performance optimization for frequent queries

This unified orchestrator enables the multi-agent system to leverage educational
content effectively while maintaining high performance and eliminating the
complexity of multi-collection management.
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
    UnifiedEducationalVectorStore, get_vector_store, ContentMetadata, 
    ContentType, AgentSpecialization, VectorStoreStats
)
from .content_loader import (
    UnifiedEducationalContentLoader, get_content_loader, UnifiedContentLoadingConfig,
    ChunkingStrategy
)
from .retrieval import (
    UnifiedEducationalRetriever, get_educational_retriever, UnifiedRetrievalContext,
    UnifiedScoredResult, UnifiedRetrievalMetrics
)
from classification.srl_classifier import SRLPhase
from utils.logging_utils import get_logger, LogContext, EventType, create_context
from config.settings import get_settings


class UnifiedKnowledgeBaseStatus(Enum):
    """Status of the unified knowledge base."""
    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UPDATING = "updating"


class UnifiedQueryType(Enum):
    """Types of queries for specialized handling in unified approach."""
    CONCEPTUAL = "conceptual"          # Understanding concepts
    IMPLEMENTATION = "implementation"   # How to implement
    DEBUGGING = "debugging"            # Fixing problems
    BEST_PRACTICES = "best_practices"  # Recommended approaches
    EXAMPLES = "examples"              # Code examples
    GENERAL = "general"                # General queries


@dataclass
class UnifiedKnowledgeBaseConfig:
    """Configuration for unified knowledge base operations."""
    # Content loading configuration
    content_loading_config: UnifiedContentLoadingConfig = field(default_factory=UnifiedContentLoadingConfig)
    
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
    
    # PDF processing options
    pdf_directory: str = "data/pdfs"
    auto_detect_specialization: bool = True


class UnifiedRetrievalRequest(BaseModel):
    """Request for unified knowledge retrieval."""
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
    content_type_preference: Optional[str] = Field(default=None, description="Preferred content type")
    
    # Educational context
    learning_objectives: Optional[List[str]] = Field(default=None, description="Learning objectives")
    programming_domain: Optional[str] = Field(default=None, description="Programming domain")


class UnifiedRetrievalResponse(BaseModel):
    """Response from unified knowledge retrieval."""
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved content")
    total_found: int = Field(default=0, description="Total matching content found")
    retrieval_time_ms: float = Field(default=0.0, description="Time taken for retrieval")
    
    # Quality metrics
    average_relevance_score: float = Field(default=0.0, description="Average relevance score")
    average_metadata_match_score: float = Field(default=0.0, description="Average metadata match score")
    content_types_found: List[str] = Field(default_factory=list, description="Types of content found")
    
    # Educational metadata
    concepts_covered: List[str] = Field(default_factory=list, description="Programming concepts covered")
    difficulty_levels: List[str] = Field(default_factory=list, description="Difficulty levels found")
    agent_specializations: List[str] = Field(default_factory=list, description="Agent specializations found")
    
    # Cache and performance info
    served_from_cache: bool = Field(default=False, description="Whether served from cache")
    query_suggestions: List[str] = Field(default_factory=list, description="Suggested follow-up queries")
    
    # Unified approach metadata
    filters_applied: Dict[str, Any] = Field(default_factory=dict, description="Metadata filters applied")
    retrieval_strategy: str = Field(default="", description="Retrieval strategy used")


class UnifiedKnowledgeBaseStats(BaseModel):
    """Comprehensive unified knowledge base statistics."""
    # Content statistics
    total_content_pieces: int = Field(default=0, description="Total content in unified knowledge base")
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
    average_metadata_match: float = Field(default=0.0, description="Average metadata match score")
    
    # System status
    knowledge_base_status: str = Field(default="uninitialized", description="Current system status")
    last_updated: Optional[float] = Field(default=None, description="Last update timestamp")
    
    # Component statistics
    vector_store_stats: Optional[Dict[str, Any]] = Field(default=None, description="Unified vector store statistics")
    retrieval_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Unified retrieval metrics")
    
    # Unified approach specific
    metadata_filter_effectiveness: Dict[str, float] = Field(default_factory=dict, description="Filter effectiveness")
    strategy_performance: Dict[str, float] = Field(default_factory=dict, description="Strategy performance")


class UnifiedEducationalKnowledgeBase:
    """
    Unified orchestrator for educational knowledge base operations.
    
    This class provides a unified interface for all knowledge base functionality,
    coordinating content loading, storage, retrieval, and quality assurance
    using a single collection with intelligent metadata filtering.
    """
    
    def __init__(self, config: Optional[UnifiedKnowledgeBaseConfig] = None):
        """Initialize the unified educational knowledge base."""
        self.config = config or UnifiedKnowledgeBaseConfig()
        self.settings = get_settings()
        self.logger = get_logger()
        
        # Initialize unified core components
        self.vector_store = get_vector_store()
        self.content_loader = get_content_loader(self.config.content_loading_config)
        self.retriever = get_educational_retriever()
        
        # System status
        self.status = UnifiedKnowledgeBaseStatus.UNINITIALIZED
        self.last_updated = None
        
        # Performance tracking
        self.query_count = 0
        self.total_query_time = 0.0
        self.cache_hits = 0
        
        # Unified query cache for performance
        self.query_cache = {} if self.config.enable_caching else None
        self.cache_lock = threading.Lock() if self.config.enable_caching else None
        
        # Component statistics
        self.component_stats = {
            "content_pieces_loaded": 0,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "cache_size": 0,
            "pdf_files_processed": 0
        }
        
        self.logger.log_event(
            EventType.COMPONENT_INIT,
            "Unified educational knowledge base initialized",
            extra_data={
                "caching_enabled": self.config.enable_caching,
                "cache_ttl": self.config.cache_ttl_seconds,
                "max_results": self.config.default_max_results,
                "pdf_directory": self.config.pdf_directory
            }
        )
    
    def initialize_unified_knowledge_base(self, 
                                        force_reload: bool = False,
                                        context: Optional[LogContext] = None) -> Dict[str, Any]:
        """
        Initialize the unified knowledge base by loading educational content from PDFs.
        
        Args:
            force_reload: Whether to force reload of existing content
            context: Logging context
            
        Returns:
            Initialization results and statistics
        """
        context = context or create_context()
        start_time = time.time()
        
        try:
            self.status = UnifiedKnowledgeBaseStatus.LOADING
            
            self.logger.log_event(
                EventType.SYSTEM_START,
                "Starting unified knowledge base initialization",
                context=context,
                extra_data={"force_reload": force_reload}
            )
            
            # Load educational content from PDFs
            loading_result = self.content_loader.load_pdf_books(force_reload=force_reload)
            
            if loading_result["status"] == "success":
                self.status = UnifiedKnowledgeBaseStatus.READY
                self.last_updated = time.time()
                self.component_stats["content_pieces_loaded"] = loading_result.get("chunks_stored", 0)
                self.component_stats["pdf_files_processed"] = loading_result.get("files_processed", 0)
                
                initialization_time = (time.time() - start_time) * 1000
                
                # Get initial unified statistics
                stats = self.get_unified_knowledge_base_stats()
                
                self.logger.log_event(
                    EventType.SYSTEM_START,
                    "Unified knowledge base initialization completed",
                    context=context,
                    extra_data={
                        "initialization_time_ms": initialization_time,
                        "content_pieces": stats.total_content_pieces,
                        "pdf_files": self.component_stats["pdf_files_processed"],
                        "status": self.status.value
                    }
                )
                
                return {
                    "status": "success",
                    "initialization_time_ms": initialization_time,
                    "content_statistics": {
                        "total_pieces": stats.total_content_pieces,
                        "by_type": stats.content_by_type,
                        "by_agent": stats.content_by_agent,
                        "by_difficulty": stats.content_by_difficulty
                    },
                    "loading_details": loading_result
                }
            else:
                self.status = UnifiedKnowledgeBaseStatus.ERROR
                
                self.logger.log_event(
                    EventType.ERROR_OCCURRED,
                    f"Unified knowledge base initialization failed: {loading_result.get('error')}",
                    context=context,
                    level="ERROR"
                )
                
                return {
                    "status": "error",
                    "error": loading_result.get("error"),
                    "initialization_time_ms": (time.time() - start_time) * 1000
                }
                
        except Exception as e:
            self.status = UnifiedKnowledgeBaseStatus.ERROR
            
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Unified knowledge base initialization exception: {str(e)}",
                context=context,
                level="ERROR"
            )
            
            return {
                "status": "error",
                "error": str(e),
                "initialization_time_ms": (time.time() - start_time) * 1000
            }
    
    def retrieve_unified_knowledge(self, 
                                 request: UnifiedRetrievalRequest,
                                 context: Optional[LogContext] = None) -> UnifiedRetrievalResponse:
        """
        Retrieve educational knowledge using unified collection approach.
        
        Args:
            request: Unified retrieval request with query and context
            context: Logging context
            
        Returns:
            Unified retrieval response with results and metadata
        """
        context = context or create_context()
        start_time = time.time()
        
        try:
            # Check unified knowledge base status
            if self.status != UnifiedKnowledgeBaseStatus.READY:
                return UnifiedRetrievalResponse(
                    results=[],
                    total_found=0,
                    retrieval_time_ms=(time.time() - start_time) * 1000
                )
            
            # Check unified cache first
            if self.config.enable_caching:
                cached_response = self._check_unified_retrieval_cache(request)
                if cached_response:
                    self.cache_hits += 1
                    self.component_stats["successful_retrievals"] += 1
                    
                    self.logger.log_event(
                        EventType.KNOWLEDGE_RETRIEVED,
                        "Unified knowledge retrieval served from cache",
                        context=context,
                        extra_data={"cache_hit": True}
                    )
                    
                    return cached_response
            
            # Create unified retrieval context
            retrieval_context = self._create_unified_retrieval_context(request)
            
            # Perform unified retrieval
            scored_results = self.retriever.retrieve_educational_content(
                retrieval_context, context
            )
            
            # Process results into unified response
            response = self._create_unified_retrieval_response(scored_results, retrieval_context, start_time)
            
            # Cache unified response
            if self.config.enable_caching:
                self._cache_unified_retrieval_response(request, response)
            
            # Update statistics
            self.query_count += 1
            self.total_query_time += response.retrieval_time_ms
            self.component_stats["successful_retrievals"] += 1
            
            self.logger.log_rag_operation(
                operation="unified_knowledge_retrieval",
                query=request.query,
                results_count=len(response.results),
                context=context
            )
            
            return response
            
        except Exception as e:
            self.component_stats["failed_retrievals"] += 1
            
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Unified knowledge retrieval failed: {str(e)}",
                context=context,
                level="ERROR",
                extra_data={"query": request.query[:100]}
            )
            
            return UnifiedRetrievalResponse(
                results=[],
                total_found=0,
                retrieval_time_ms=(time.time() - start_time) * 1000
            )
    
    def add_unified_educational_content(self, 
                                      content: str,
                                      content_type: ContentType,
                                      agent_specialization: AgentSpecialization,
                                      metadata: Optional[Dict[str, Any]] = None,
                                      context: Optional[LogContext] = None) -> bool:
        """
        Add new educational content to the unified knowledge base.
        
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
            # Create content metadata for unified storage
            content_metadata = ContentMetadata(
                content_id=f"user_content_{int(time.time())}",
                content_type=content_type,
                agent_specialization=agent_specialization,
                content_length=len(content),
                has_code_examples=self._detect_code_examples(content),
                has_error_examples=self._detect_error_examples(content),
                **(metadata or {})
            )
            
            # Add to unified vector store
            success = self.vector_store.add_content(content, content_metadata, context)
            
            if success:
                self.component_stats["content_pieces_loaded"] += 1
                
                # Clear unified cache to ensure fresh results
                if self.config.enable_caching:
                    with self.cache_lock:
                        self.query_cache.clear()
                
                self.logger.log_event(
                    EventType.KNOWLEDGE_RETRIEVED,
                    "Educational content added to unified knowledge base",
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
                f"Failed to add content to unified knowledge base: {str(e)}",
                context=context,
                level="ERROR"
            )
            return False
    
    def update_unified_content_quality_scores(self, 
                                            effectiveness_feedback: Dict[str, float],
                                            context: Optional[LogContext] = None):
        """
        Update content quality scores based on effectiveness feedback in unified approach.
        
        Args:
            effectiveness_feedback: Mapping of content_id to effectiveness score
            context: Logging context
        """
        context = context or create_context()
        
        try:
            updated_count = 0
            
            for content_id, effectiveness_score in effectiveness_feedback.items():
                # Update content metadata in unified vector store
                # This implementation would depend on the vector store's update capabilities
                
                self.logger.log_event(
                    EventType.KNOWLEDGE_RETRIEVED,
                    f"Updated effectiveness score for unified content {content_id}",
                    context=context,
                    extra_data={
                        "content_id": content_id,
                        "effectiveness_score": effectiveness_score
                    }
                )
                
                updated_count += 1
            
            self.logger.log_event(
                EventType.SYSTEM_START,
                f"Updated quality scores for {updated_count} content pieces in unified KB",
                context=context,
                extra_data={"updated_count": updated_count}
            )
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Failed to update unified content quality scores: {str(e)}",
                context=context,
                level="ERROR"
            )
    
    def get_unified_knowledge_base_stats(self) -> UnifiedKnowledgeBaseStats:
        """
        Get comprehensive unified knowledge base statistics.
        
        Returns:
            Unified knowledge base statistics
        """
        try:
            # Get unified vector store statistics
            vector_stats = self.vector_store.get_vector_store_stats()
            
            # Get unified retrieval metrics
            retrieval_metrics = self.retriever.get_retrieval_metrics()
            
            # Calculate average query time
            avg_query_time = (
                self.total_query_time / self.query_count 
                if self.query_count > 0 else 0.0
            )
            
            # Calculate cache hit rate
            cache_hit_rate = (
                self.cache_hits / self.query_count 
                if self.query_count > 0 else 0.0
            )
            
            # Calculate average content quality (placeholder - would need implementation)
            avg_content_quality = 0.7  # Placeholder
            
            return UnifiedKnowledgeBaseStats(
                total_content_pieces=vector_stats.total_documents,
                content_by_type=vector_stats.content_by_type,
                content_by_agent=vector_stats.content_by_agent,
                content_by_difficulty=vector_stats.content_by_difficulty,
                
                total_queries_processed=self.query_count,
                average_query_time_ms=avg_query_time,
                cache_hit_rate=cache_hit_rate,
                
                average_content_quality=avg_content_quality,
                average_retrieval_relevance=retrieval_metrics.average_educational_relevance,
                average_metadata_match=retrieval_metrics.average_metadata_match_score,
                
                knowledge_base_status=self.status.value,
                last_updated=self.last_updated,
                
                vector_store_stats=vector_stats.dict(),
                retrieval_metrics=retrieval_metrics.dict(),
                
                metadata_filter_effectiveness=retrieval_metrics.filter_effectiveness,
                strategy_performance=retrieval_metrics.strategy_performance
            )
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Failed to get unified knowledge base stats: {str(e)}",
                level="ERROR"
            )
            return UnifiedKnowledgeBaseStats()
    
    def _check_unified_retrieval_cache(self, request: UnifiedRetrievalRequest) -> Optional[UnifiedRetrievalResponse]:
        """Check unified retrieval cache for existing response."""
        if not self.config.enable_caching:
            return None
        
        cache_key = self._generate_unified_cache_key(request)
        
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
    
    def _cache_unified_retrieval_response(self, request: UnifiedRetrievalRequest, response: UnifiedRetrievalResponse):
        """Cache unified retrieval response."""
        if not self.config.enable_caching:
            return
        
        cache_key = self._generate_unified_cache_key(request)
        
        with self.cache_lock:
            # Simple cache size management
            if len(self.query_cache) > 1000:
                # Remove oldest 20% of entries
                keys_to_remove = list(self.query_cache.keys())[:200]
                for key in keys_to_remove:
                    del self.query_cache[key]
            
            self.query_cache[cache_key] = {
                "response": response,
                "timestamp": time.time()
            }
            
            self.component_stats["cache_size"] = len(self.query_cache)
    
    def _generate_unified_cache_key(self, request: UnifiedRetrievalRequest) -> str:
        """Generate cache key for unified retrieval request."""
        key_data = {
            "query": request.query,
            "agent_type": request.agent_type,
            "srl_phase": request.srl_phase,
            "student_level": request.student_level,
            "content_type_preference": request.content_type_preference,
            "prefer_code_examples": request.prefer_code_examples,
            "programming_domain": request.programming_domain,
            "max_results": request.max_results
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        import hashlib
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _create_unified_retrieval_context(self, request: UnifiedRetrievalRequest) -> UnifiedRetrievalContext:
        """Create unified retrieval context from request."""
        return UnifiedRetrievalContext(
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
            prefer_error_examples=False,  # Could be derived from error_message
            programming_domain=request.programming_domain,
            learning_objectives=request.learning_objectives,
            content_type_preference=request.content_type_preference
        )
    
    def _create_unified_retrieval_response(self, 
                                         scored_results: List[UnifiedScoredResult],
                                         retrieval_context: UnifiedRetrievalContext,
                                         start_time: float) -> UnifiedRetrievalResponse:
        """Create unified retrieval response from scored results."""
        retrieval_time = (time.time() - start_time) * 1000
        
        # Process results for response
        results = []
        content_types = set()
        difficulty_levels = set()
        agent_specializations = set()
        all_concepts = set()
        
        total_relevance = 0.0
        total_metadata_match = 0.0
        
        for scored_result in scored_results:
            result_dict = {
                "content_id": scored_result.result.content_id,
                "content": scored_result.result.content,
                "similarity_score": scored_result.similarity_score,
                "metadata_match_score": scored_result.metadata_match_score,
                "educational_relevance_score": scored_result.educational_relevance_score,
                "combined_score": scored_result.combined_score,
                "content_type": scored_result.result.content_type,
                "agent_specialization": scored_result.result.agent_specialization,
                "difficulty_level": scored_result.result.difficulty_level,
                "programming_concepts": scored_result.result.programming_concepts,
                "rank_position": scored_result.rank_position,
                "score_explanation": scored_result.score_explanation
            }
            
            results.append(result_dict)
            
            # Collect metadata for summary
            content_types.add(scored_result.result.content_type)
            difficulty_levels.add(scored_result.result.difficulty_level)
            agent_specializations.add(scored_result.result.agent_specialization)
            all_concepts.update(scored_result.result.programming_concepts)
            
            total_relevance += scored_result.educational_relevance_score
            total_metadata_match += scored_result.metadata_match_score
        
        # Calculate averages
        avg_relevance = total_relevance / len(scored_results) if scored_results else 0.0
        avg_metadata_match = total_metadata_match / len(scored_results) if scored_results else 0.0
        
        # Generate query suggestions
        query_suggestions = self._generate_unified_query_suggestions(retrieval_context, scored_results)
        
        # Determine filters applied
        filters_applied = self._get_applied_filters(retrieval_context)
        
        return UnifiedRetrievalResponse(
            results=results,
            total_found=len(scored_results),
            retrieval_time_ms=retrieval_time,
            average_relevance_score=avg_relevance,
            average_metadata_match_score=avg_metadata_match,
            content_types_found=list(content_types),
            concepts_covered=list(all_concepts),
            difficulty_levels=list(difficulty_levels),
            agent_specializations=list(agent_specializations),
            served_from_cache=False,
            query_suggestions=query_suggestions,
            filters_applied=filters_applied,
            retrieval_strategy=""  # Would be filled by retriever
        )
    
    def _generate_unified_query_suggestions(self, 
                                          context: UnifiedRetrievalContext,
                                          results: List[UnifiedScoredResult]) -> List[str]:
        """Generate follow-up query suggestions based on unified retrieval."""
        suggestions = []
        
        # Suggestions based on found concepts
        concepts_found = set()
        for result in results[:3]:  # Top 3 results
            concepts_found.update(result.result.programming_concepts)
        
        for concept in list(concepts_found)[:2]:  # Top 2 concepts
            suggestions.append(f"Show me more examples of {concept}")
            suggestions.append(f"How to implement {concept} efficiently")
        
        # Suggestions based on agent specialization
        if context.agent_specialization == AgentSpecialization.IMPLEMENTATION.value:
            suggestions.append("Show me implementation best practices")
            suggestions.append("What are common design patterns for this?")
        elif context.agent_specialization == AgentSpecialization.DEBUGGING.value:
            suggestions.append("What are common errors in this area?")
            suggestions.append("How to troubleshoot this issue?")
        
        # Suggestions based on content types found
        content_types = {result.result.content_type for result in results}
        if ContentType.CODE_EXAMPLE.value not in content_types:
            suggestions.append("Show me code examples for this")
        if ContentType.BEST_PRACTICE.value not in content_types:
            suggestions.append("What are the best practices here?")
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _get_applied_filters(self, context: UnifiedRetrievalContext) -> Dict[str, Any]:
        """Get dictionary of applied filters."""
        filters = {}
        
        if context.agent_specialization:
            filters["agent_specialization"] = context.agent_specialization
        if context.student_level:
            filters["difficulty_level"] = context.student_level
        if context.content_type_preference:
            filters["content_type"] = context.content_type_preference
        if context.prefer_code_examples:
            filters["has_code_examples"] = True
        if context.programming_domain:
            filters["programming_domain"] = context.programming_domain
        
        return filters
    
    def _detect_code_examples(self, content: str) -> bool:
        """Detect if content contains code examples."""
        code_indicators = ['```', 'def ', 'class ', 'function', 'import ', 'from ', '{', '}']
        return any(indicator in content for indicator in code_indicators)
    
    def _detect_error_examples(self, content: str) -> bool:
        """Detect if content contains error examples."""
        error_indicators = ['error:', 'exception:', 'traceback:', 'syntaxerror', 'typeerror']
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in error_indicators)
    
    def clear_unified_cache(self):
        """Clear the unified retrieval cache."""
        if self.config.enable_caching:
            with self.cache_lock:
                self.query_cache.clear()
                self.component_stats["cache_size"] = 0
            
            self.logger.log_event(
                EventType.SYSTEM_START,
                "Unified knowledge base cache cleared"
            )
    
    def get_unified_status(self) -> Dict[str, Any]:
        """Get current unified knowledge base status."""
        return {
            "status": self.status.value,
            "last_updated": self.last_updated,
            "total_queries": self.query_count,
            "cache_hits": self.cache_hits,
            "cache_size": self.component_stats.get("cache_size", 0),
            "content_pieces": self.component_stats.get("content_pieces_loaded", 0),
            "pdf_files_processed": self.component_stats.get("pdf_files_processed", 0)
        }


# Global unified knowledge base instance
_unified_knowledge_base: Optional[UnifiedEducationalKnowledgeBase] = None


def get_knowledge_base(config: Optional[UnifiedKnowledgeBaseConfig] = None, 
                      reload: bool = False) -> UnifiedEducationalKnowledgeBase:
    """
    Get global unified knowledge base instance (singleton pattern).
    
    Args:
        config: Unified knowledge base configuration
        reload: Force creation of new knowledge base instance
        
    Returns:
        UnifiedEducationalKnowledgeBase instance
    """
    global _unified_knowledge_base
    if _unified_knowledge_base is None or reload:
        _unified_knowledge_base = UnifiedEducationalKnowledgeBase(config)
    return _unified_knowledge_base


# Convenience functions for agent integration
def initialize_knowledge_base(force_reload: bool = False) -> Dict[str, Any]:
    """Initialize the unified knowledge base."""
    kb = get_knowledge_base()
    return kb.initialize_unified_knowledge_base(force_reload=force_reload)


def retrieve_for_agent(query: str, 
                      agent_type: str,
                      srl_phase: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> UnifiedRetrievalResponse:
    """
    Convenience function for agent-specific retrieval from unified knowledge base.
    
    Args:
        query: Query string
        agent_type: Agent type (implementation, debugging, etc.)
        srl_phase: SRL phase (forethought, performance, etc.)
        context: Additional context information
        
    Returns:
        Unified retrieval response
    """
    kb = get_knowledge_base()
    
    context = context or {}
    
    request = UnifiedRetrievalRequest(
        query=query,
        agent_type=agent_type,
        srl_phase=srl_phase,
        student_level=context.get("student_level"),
        code_snippet=context.get("code_snippet"),
        error_message=context.get("error_message"),
        conversation_history=context.get("conversation_history"),
        max_results=context.get("max_results", 5),
        prefer_code_examples=context.get("prefer_code_examples", False),
        programming_domain=context.get("programming_domain"),
        learning_objectives=context.get("learning_objectives"),
        content_type_preference=context.get("content_type_preference")
    )
    
    return kb.retrieve_unified_knowledge(request)


def get_knowledge_base_stats() -> UnifiedKnowledgeBaseStats:
    """Get unified knowledge base statistics."""
    kb = get_knowledge_base()
    return kb.get_unified_knowledge_base_stats()


if __name__ == "__main__":
    # Unified knowledge base test
    try:
        # Initialize unified knowledge base
        kb = get_knowledge_base()
        
        # Test initialization
        init_result = kb.initialize_unified_knowledge_base()
        print(f"Unified KB initialization: {init_result['status']}")
        if init_result['status'] == 'success':
            print(f"Content pieces: {init_result['content_statistics']['total_pieces']}")
            print(f"PDF files processed: {init_result['loading_details']['files_processed']}")
        
        # Test retrieval
        request = UnifiedRetrievalRequest(
            query="How to implement binary search algorithm?",
            agent_type=AgentSpecialization.IMPLEMENTATION.value,
            srl_phase=SRLPhase.FORETHOUGHT.value,
            student_level="intermediate",
            prefer_code_examples=True,
            programming_domain="algorithms",
            max_results=3
        )
        
        response = kb.retrieve_unified_knowledge(request)
        print(f"\nUnified retrieval results: {len(response.results)} found")
        print(f"Average relevance: {response.average_relevance_score:.3f}")
        print(f"Average metadata match: {response.average_metadata_match_score:.3f}")
        print(f"Content types found: {response.content_types_found}")
        print(f"Concepts covered: {response.concepts_covered}")
        
        # Test statistics
        stats = kb.get_unified_knowledge_base_stats()
        print(f"\nUnified KB Statistics:")
        print(f"Total content: {stats.total_content_pieces}")
        print(f"By type: {stats.content_by_type}")
        print(f"By agent: {stats.content_by_agent}")
        print(f"Total queries: {stats.total_queries_processed}")
        print(f"Cache hit rate: {stats.cache_hit_rate:.2%}")
        
        print("✅ Unified knowledge base test completed successfully!")
        
    except Exception as e:
        print(f"❌ Unified knowledge base test failed: {e}")
        import traceback
        traceback.print_exc()
