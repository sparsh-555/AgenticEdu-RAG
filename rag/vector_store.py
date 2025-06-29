"""
Unified Vector Store Implementation for Educational RAG System

This module provides a unified vector database abstraction using ChromaDB,
optimized for educational content storage and retrieval. The implementation
uses a single collection with rich metadata for intelligent content filtering
rather than separate agent-specific collections.

Key Features:
1. Unified Collection: Single collection with comprehensive metadata tagging
2. Rich Metadata Filtering: Context-aware retrieval using metadata filters
3. Educational Content Optimization: Specialized for programming education
4. Performance Optimization: Efficient similarity search with smart caching
5. Content Management: Adding, updating, and removing educational content
6. Quality Monitoring: Performance metrics and educational effectiveness tracking

Design Principles:
- Unified Architecture: Single source of truth for all educational content
- Metadata-Driven: Rich metadata enables intelligent content filtering
- Educational Focus: Optimized for programming education use cases
- Performance: Efficient operations for real-time query processing
- Scalability: Support for large educational content collections
"""

import time
import hashlib
import json
import uuid
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import threading

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from pydantic import BaseModel, Field

from config.settings import get_settings
from utils.logging_utils import get_logger, LogContext, EventType, create_context
from utils.api_utils import get_openai_client


class ContentType(Enum):
    """Types of educational content for specialized handling."""
    IMPLEMENTATION_GUIDE = "implementation_guide"
    DEBUGGING_RESOURCE = "debugging_resource"
    CONCEPT_EXPLANATION = "concept_explanation"
    CODE_EXAMPLE = "code_example"
    EXERCISE = "exercise"
    BEST_PRACTICE = "best_practice"
    COMMON_ERROR = "common_error"
    TROUBLESHOOTING = "troubleshooting"
    GENERAL = "general"


class AgentSpecialization(Enum):
    """Agent specializations for content tagging."""
    IMPLEMENTATION = "implementation"
    DEBUGGING = "debugging"
    SHARED = "shared"


@dataclass
class ContentMetadata:
    """Comprehensive metadata for educational content pieces."""
    content_id: str
    content_type: ContentType
    agent_specialization: AgentSpecialization
    
    # Educational metadata
    programming_concepts: List[str] = field(default_factory=list)
    difficulty_level: str = "intermediate"  # beginner, intermediate, advanced
    programming_language: Optional[str] = None
    topic_tags: List[str] = field(default_factory=list)
    
    # Content characteristics
    content_length: int = 0
    has_code_examples: bool = False
    has_error_examples: bool = False
    
    # Usage metadata
    retrieval_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    effectiveness_score: float = 0.5
    
    # Source information
    source_file: Optional[str] = None
    created_timestamp: float = field(default_factory=time.time)
    updated_timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for storage, handling None values."""
        # Filter out None values and convert lists to strings
        result = {}
        
        result["content_id"] = self.content_id
        result["content_type"] = self.content_type.value
        result["agent_specialization"] = self.agent_specialization.value
        result["programming_concepts"] = ",".join(self.programming_concepts) if self.programming_concepts else ""
        result["difficulty_level"] = self.difficulty_level or "medium"
        result["programming_language"] = self.programming_language or "general"
        result["topic_tags"] = ",".join(self.topic_tags) if self.topic_tags else ""
        result["content_length"] = self.content_length or 0
        result["has_code_examples"] = bool(self.has_code_examples)
        result["has_error_examples"] = bool(self.has_error_examples)
        result["retrieval_count"] = self.retrieval_count or 0
        result["last_accessed"] = self.last_accessed or 0.0
        result["effectiveness_score"] = self.effectiveness_score or 0.0
        result["source_file"] = self.source_file or ""
        result["created_timestamp"] = self.created_timestamp or 0.0
        result["updated_timestamp"] = self.updated_timestamp or 0.0
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentMetadata':
        """Create metadata from dictionary."""
        return cls(
            content_id=data.get("content_id", f"unknown_{int(time.time())}"),
            content_type=ContentType(data.get("content_type", "general")),
            agent_specialization=AgentSpecialization(data.get("agent_specialization", "shared")),
            programming_concepts=data.get("programming_concepts", []),
            difficulty_level=data.get("difficulty_level", "intermediate"),
            programming_language=data.get("programming_language"),
            topic_tags=data.get("topic_tags", []),
            content_length=data.get("content_length", 0),
            has_code_examples=data.get("has_code_examples", False),
            has_error_examples=data.get("has_error_examples", False),
            retrieval_count=data.get("retrieval_count", 0),
            last_accessed=data.get("last_accessed", time.time()),
            effectiveness_score=data.get("effectiveness_score", 0.5),
            source_file=data.get("source_file"),
            created_timestamp=data.get("created_timestamp", time.time()),
            updated_timestamp=data.get("updated_timestamp", time.time())
        )


class RetrievalResult(BaseModel):
    """Result from unified vector store retrieval."""
    content_id: str = Field(..., description="Unique content identifier")
    content: str = Field(..., description="Retrieved content text")
    similarity_score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Content metadata")
    
    # Educational context
    content_type: str = Field(..., description="Type of educational content")
    agent_specialization: str = Field(..., description="Agent specialization")
    programming_concepts: List[str] = Field(default_factory=list, description="Programming concepts covered")
    difficulty_level: str = Field(default="intermediate", description="Content difficulty level")


class VectorStoreStats(BaseModel):
    """Statistics for unified vector store performance monitoring."""
    total_documents: int = Field(default=0, description="Total documents stored")
    content_by_type: Dict[str, int] = Field(default_factory=dict, description="Documents by content type")
    content_by_agent: Dict[str, int] = Field(default_factory=dict, description="Documents by agent specialization")
    content_by_difficulty: Dict[str, int] = Field(default_factory=dict, description="Documents by difficulty level")
    total_queries: int = Field(default=0, description="Total queries processed")
    average_query_time_ms: float = Field(default=0.0, description="Average query processing time")
    cache_hit_rate: float = Field(default=0.0, description="Cache hit rate for queries")
    storage_size_mb: float = Field(default=0.0, description="Estimated storage size")


class UnifiedEducationalVectorStore:
    """
    Unified vector store for educational content using ChromaDB.
    
    This implementation uses a single collection with rich metadata for
    intelligent content filtering, replacing the previous agent-specific
    collection architecture with a more flexible and maintainable approach.
    """
    
    def __init__(self):
        """Initialize the unified educational vector store."""
        self.settings = get_settings()
        self.logger = get_logger()
        self.openai_client = get_openai_client()
        
        # Initialize ChromaDB client
        self._initialize_chroma_client()
        
        # Single unified collection
        self.collection = None
        self._initialize_collection()
        
        # Performance tracking
        self.query_count = 0
        self.total_query_time = 0.0
        self.cache_hits = 0
        
        # Simple query cache for performance
        self.query_cache = {}
        self.cache_lock = threading.Lock()
        
        self.logger.log_event(
            EventType.COMPONENT_INIT,
            "Unified educational vector store initialized",
            extra_data={
                "persist_directory": self.settings.chroma.persist_directory,
                "collection_name": self.settings.chroma.collection_name
            }
        )
    
    def _initialize_chroma_client(self):
        """Initialize ChromaDB client with proper configuration."""
        try:
            # Create persistent directory if it doesn't exist
            persist_dir = Path(self.settings.chroma.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize Chroma with persistent storage
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
            )
            
            self.logger.log_event(
                EventType.SYSTEM_START,
                "ChromaDB client initialized successfully",
                extra_data={"persist_directory": str(persist_dir)}
            )
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Failed to initialize ChromaDB client: {str(e)}",
                level="ERROR"
            )
            raise RuntimeError(f"ChromaDB initialization failed: {str(e)}")
    
    def _initialize_collection(self):
        """Initialize single unified collection for all educational content."""
        try:
            collection_name = self.settings.chroma.collection_name
            
            # Create or get unified collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "description": "Unified educational content collection",
                    "architecture": "unified_with_metadata_filtering",
                    "created_at": time.time()
                }
            )
            
            self.logger.log_event(
                EventType.COMPONENT_INIT,
                f"Unified collection '{collection_name}' initialized",
                extra_data={"collection_name": collection_name}
            )
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Failed to initialize unified collection: {str(e)}",
                level="ERROR"
            )
            raise
    
    def add_content(self, 
                   content: str,
                   metadata: ContentMetadata,
                   context: Optional[LogContext] = None) -> bool:
        """
        Add educational content to the unified collection.
        
        Args:
            content: Text content to store
            metadata: Educational metadata for the content
            context: Logging context
            
        Returns:
            True if content was added successfully, False otherwise
        """
        context = context or create_context()
        start_time = time.time()
        
        try:
            # Generate embedding for content
            embedding = self._generate_embedding(content)
            
            # Prepare document data
            document_id = metadata.content_id
            document_metadata = metadata.to_dict()
            
            # Add to unified collection
            self.collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[document_metadata],
                ids=[document_id]
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                f"Content added to unified collection",
                context=context,
                extra_data={
                    "content_id": document_id,
                    "content_type": metadata.content_type.value,
                    "agent_specialization": metadata.agent_specialization.value,
                    "content_length": len(content),
                    "processing_time_ms": processing_time
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Failed to add content: {str(e)}",
                context=context,
                level="ERROR",
                extra_data={"content_id": metadata.content_id}
            )
            return False
    
    def search_similar_content(self,
                             query: str,
                             agent_specialization: Optional[AgentSpecialization] = None,
                             content_type_filter: Optional[ContentType] = None,
                             difficulty_level: Optional[str] = None,
                             programming_concepts: Optional[List[str]] = None,
                             max_results: int = 5,
                             similarity_threshold: float = 0.0,
                             context: Optional[LogContext] = None) -> List[RetrievalResult]:
        """
        Search for similar educational content using unified collection with metadata filtering.
        
        Args:
            query: Search query
            agent_specialization: Filter by agent specialization
            content_type_filter: Filter by content type
            difficulty_level: Filter by difficulty level
            programming_concepts: Filter by programming concepts
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity score
            context: Logging context
            
        Returns:
            List of retrieval results sorted by similarity
        """
        context = context or create_context()
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(
            query, agent_specialization, content_type_filter, 
            difficulty_level, programming_concepts, max_results, similarity_threshold
        )
        
        with self.cache_lock:
            if cache_key in self.query_cache:
                self.cache_hits += 1
                cached_result = self.query_cache[cache_key]
                
                self.logger.log_event(
                    EventType.KNOWLEDGE_RETRIEVED,
                    "Query served from cache",
                    context=context,
                    extra_data={"cache_hit": True, "results_count": len(cached_result)}
                )
                
                return cached_result
        
        try:
            # DEBUGGING: Log search parameters
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                "VECTOR STORE SEARCH - Starting search",
                extra_data={
                    "query": query[:100],
                    "agent_specialization": agent_specialization.value if agent_specialization else None,
                    "content_type_filter": content_type_filter.value if content_type_filter else None,
                    "difficulty_level": difficulty_level,
                    "programming_concepts": programming_concepts,
                    "max_results": max_results,
                    "similarity_threshold": similarity_threshold
                }
            )
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Build metadata filter for unified collection
            where_filter = self._build_unified_metadata_filter(
                agent_specialization, content_type_filter, difficulty_level, programming_concepts
            )
            
            # DEBUGGING: Log the metadata filter being applied
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                "VECTOR STORE SEARCH - Metadata filter built",
                extra_data={
                    "where_filter": where_filter,
                    "has_filter": where_filter is not None,
                    "filter_keys": list(where_filter.keys()) if where_filter else []
                }
            )
            
            # Perform similarity search on unified collection
            # RESEARCH FIX: Get more results to ensure good candidates after filtering
            search_count = max(max_results * 4, 15)  # Get significantly more for research demo
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=search_count,
                where=where_filter if where_filter else None
            )
            
            # DEBUGGING: Log raw database results
            raw_count = len(results['ids'][0]) if results['ids'] and results['ids'][0] else 0
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                "VECTOR STORE SEARCH - Raw database results",
                extra_data={
                    "raw_results_count": raw_count,
                    "requested_n_results": max_results * 2,
                    "has_results": raw_count > 0
                }
            )
            
            # Process results
            processed_results = self._process_unified_search_results(
                results, similarity_threshold
            )
            
            # DEBUGGING: Log after processing
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                "VECTOR STORE SEARCH - After processing",
                extra_data={
                    "processed_results_count": len(processed_results),
                    "similarity_threshold": similarity_threshold,
                    "filtered_by_similarity": raw_count - len(processed_results)
                }
            )
            
            # Sort by similarity and limit results
            processed_results.sort(key=lambda x: x.similarity_score, reverse=True)
            final_results = processed_results[:max_results]
            
            # DEBUGGING: Log final results
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                "VECTOR STORE SEARCH - Final results",
                extra_data={
                    "final_results_count": len(final_results),
                    "max_results_requested": max_results,
                    "top_similarity_scores": [r.similarity_score for r in final_results[:3]]
                }
            )
            
            # Update usage statistics
            self._update_content_usage_stats(final_results)
            
            # Cache results
            with self.cache_lock:
                if len(self.query_cache) > 1000:  # Simple cache size management
                    # Remove oldest 20% of entries
                    keys_to_remove = list(self.query_cache.keys())[:200]
                    for key in keys_to_remove:
                        del self.query_cache[key]
                
                self.query_cache[cache_key] = final_results
            
            # Track performance
            processing_time = (time.time() - start_time) * 1000
            self.query_count += 1
            self.total_query_time += processing_time
            
            self.logger.log_rag_operation(
                operation="unified_similarity_search",
                query=query,
                results_count=len(final_results),
                context=context
            )
            
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                f"Unified search completed: {len(final_results)} results",
                context=context,
                extra_data={
                    "processing_time_ms": processing_time,
                    "filters_applied": {
                        "agent_specialization": agent_specialization.value if agent_specialization else None,
                        "content_type": content_type_filter.value if content_type_filter else None,
                        "difficulty_level": difficulty_level,
                        "programming_concepts": programming_concepts
                    },
                    "similarity_threshold": similarity_threshold,
                    "max_results": max_results
                }
            )
            
            return final_results
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Unified similarity search failed: {str(e)}",
                context=context,
                level="ERROR",
                extra_data={"query_preview": query[:100]}
            )
            return []
    
    def update_content(self,
                      content_id: str,
                      new_content: Optional[str] = None,
                      new_metadata: Optional[ContentMetadata] = None,
                      context: Optional[LogContext] = None) -> bool:
        """
        Update existing content in unified collection.
        
        Args:
            content_id: ID of content to update
            new_content: New content text (optional)
            new_metadata: New metadata (optional)
            context: Logging context
            
        Returns:
            True if content was updated successfully
        """
        context = context or create_context()
        
        try:
            # Check if content exists in unified collection
            existing = self.collection.get(ids=[content_id])
            
            if not existing['ids']:
                self.logger.log_event(
                    EventType.WARNING_ISSUED,
                    f"Content not found for update: {content_id}",
                    context=context,
                    level="WARNING"
                )
                return False
            
            # Prepare update data
            update_data = {}
            
            if new_content:
                update_data['documents'] = [new_content]
                update_data['embeddings'] = [self._generate_embedding(new_content)]
            
            if new_metadata:
                new_metadata.updated_timestamp = time.time()
                update_data['metadatas'] = [new_metadata.to_dict()]
            
            if update_data:
                self.collection.update(
                    ids=[content_id],
                    **update_data
                )
            
            # Clear cache to ensure consistency
            with self.cache_lock:
                self.query_cache.clear()
            
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                f"Content updated in unified collection",
                context=context,
                extra_data={"content_id": content_id}
            )
            
            return True
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Failed to update content: {str(e)}",
                context=context,
                level="ERROR",
                extra_data={"content_id": content_id}
            )
            return False
    
    def delete_content(self, content_id: str, context: Optional[LogContext] = None) -> bool:
        """
        Delete content from the unified collection.
        
        Args:
            content_id: ID of content to delete
            context: Logging context
            
        Returns:
            True if content was deleted successfully
        """
        context = context or create_context()
        
        try:
            # Check if content exists
            existing = self.collection.get(ids=[content_id])
            
            if not existing['ids']:
                self.logger.log_event(
                    EventType.WARNING_ISSUED,
                    f"Content not found for deletion: {content_id}",
                    context=context,
                    level="WARNING"
                )
                return False
            
            # Delete the content
            self.collection.delete(ids=[content_id])
            
            # Clear cache
            with self.cache_lock:
                self.query_cache.clear()
            
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                f"Content deleted from unified collection",
                context=context,
                extra_data={"content_id": content_id}
            )
            
            return True
                
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Failed to delete content: {str(e)}",
                context=context,
                level="ERROR",
                extra_data={"content_id": content_id}
            )
            return False
    
    def get_vector_store_stats(self) -> VectorStoreStats:
        """
        Get comprehensive unified vector store statistics.
        
        Returns:
            Vector store statistics
        """
        try:
            total_documents = self.collection.count()
            
            # Get all documents to analyze metadata distribution
            all_docs = self.collection.get(include=['metadatas'])
            metadatas = all_docs.get('metadatas', [])
            
            # Analyze content distribution
            content_by_type = {}
            content_by_agent = {}
            content_by_difficulty = {}
            
            for metadata in metadatas:
                # Content type distribution
                content_type = metadata.get('content_type', 'unknown')
                content_by_type[content_type] = content_by_type.get(content_type, 0) + 1
                
                # Agent specialization distribution
                agent_spec = metadata.get('agent_specialization', 'unknown')
                content_by_agent[agent_spec] = content_by_agent.get(agent_spec, 0) + 1
                
                # Difficulty level distribution
                difficulty = metadata.get('difficulty_level', 'unknown')
                content_by_difficulty[difficulty] = content_by_difficulty.get(difficulty, 0) + 1
            
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
            
            return VectorStoreStats(
                total_documents=total_documents,
                content_by_type=content_by_type,
                content_by_agent=content_by_agent,
                content_by_difficulty=content_by_difficulty,
                total_queries=self.query_count,
                average_query_time_ms=avg_query_time,
                cache_hit_rate=cache_hit_rate,
                storage_size_mb=0.0  # ChromaDB doesn't provide easy size calculation
            )
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Failed to get vector store stats: {str(e)}",
                level="ERROR"
            )
            return VectorStoreStats()
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI."""
        try:
            embeddings = self.openai_client.create_embeddings([text])
            return embeddings[0]
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Embedding generation failed: {str(e)}",
                level="ERROR"
            )
            # Return zero embedding as fallback
            return [0.0] * 1536  # OpenAI embedding dimension
    
    def _build_unified_metadata_filter(self,
                                     agent_specialization: Optional[AgentSpecialization],
                                     content_type: Optional[ContentType],
                                     difficulty_level: Optional[str],
                                     programming_concepts: Optional[List[str]]) -> Optional[Dict[str, Any]]:
        """Build metadata filter for unified collection search using research-appropriate tiered filtering."""
        # DEBUGGING: Log input parameters
        self.logger.log_event(
            EventType.KNOWLEDGE_RETRIEVED,
            "BUILDING METADATA FILTER - Input parameters",
            extra_data={
                "agent_specialization": agent_specialization.value if agent_specialization else None,
                "content_type": content_type.value if content_type else None,
                "difficulty_level": difficulty_level,
                "programming_concepts": programming_concepts
            }
        )
        
        # RESEARCH FIX: Use tiered filtering approach for better research demonstration
        # Tier 1: Prefer agent-specific content but allow shared content
        # Tier 2: Allow adjacent difficulty levels  
        # Tier 3: No hard filtering to ensure minimum results
        
        filters = []
        
        # Agent specialization: Include target agent AND shared content
        if agent_specialization:
            # Allow both the specific agent content AND shared content
            agent_filter = {
                "$or": [
                    {"agent_specialization": agent_specialization.value},
                    {"agent_specialization": AgentSpecialization.SHARED.value}
                ]
            }
            filters.append(agent_filter)
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                "METADATA FILTER - Agent specialization filter added (including shared)",
                extra_data={"filter": agent_filter}
            )
        
        # Content type: Only filter if very specific request, otherwise allow all educational content
        if content_type and content_type in [ContentType.DEBUGGING_RESOURCE, ContentType.IMPLEMENTATION_GUIDE]:
            # Only apply strict content type filtering for very specific requests
            filters.append({"content_type": content_type.value})
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                "METADATA FILTER - Content type filter added (specific request)",
                extra_data={"content_type": content_type.value}
            )
        elif content_type:
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                "METADATA FILTER - Content type not filtered (allowing educational variety)",
                extra_data={"requested_type": content_type.value}
            )
        
        # Difficulty level: Use adjacent levels for more inclusive filtering  
        if difficulty_level:
            difficulty_mapping = {
                "beginner": ["beginner", "intermediate"],
                "intermediate": ["beginner", "intermediate", "advanced"], 
                "advanced": ["intermediate", "advanced"]
            }
            allowed_levels = difficulty_mapping.get(difficulty_level, [difficulty_level])
            
            if len(allowed_levels) > 1:
                difficulty_filter = {"difficulty_level": {"$in": allowed_levels}}
                filters.append(difficulty_filter)
                self.logger.log_event(
                    EventType.KNOWLEDGE_RETRIEVED,
                    "METADATA FILTER - Difficulty level filter added (inclusive)",
                    extra_data={"allowed_levels": allowed_levels}
                )
            else:
                self.logger.log_event(
                    EventType.KNOWLEDGE_RETRIEVED,
                    "METADATA FILTER - Single difficulty level, allowing broader content"
                )
        
        # Programming concepts: Only filter if we have very specific concepts
        if programming_concepts and len(programming_concepts) <= 2:
            # Only filter on concepts if we have 1-2 specific ones
            concept_filter = {
                "$or": [
                    {"programming_concepts": {"$regex": f".*{concept}.*"}}
                    for concept in programming_concepts[:2]
                ]
            }
            filters.append(concept_filter)
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                "METADATA FILTER - Programming concepts filter added",
                extra_data={"concepts": programming_concepts[:2]}
            )
        elif programming_concepts:
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                "METADATA FILTER - Too many concepts, allowing broader content",
                extra_data={"concept_count": len(programming_concepts)}
            )
        
        # Build final filter - use OR logic for more inclusive results
        final_filter = None
        if not filters:
            final_filter = None  # No filtering - get best similarity matches
        elif len(filters) == 1:
            final_filter = filters[0]
        else:
            # Use OR logic between different filter types for more inclusive results
            final_filter = {"$or": filters}
        
        # DEBUGGING: Log final filter
        self.logger.log_event(
            EventType.KNOWLEDGE_RETRIEVED,
            "RESEARCH METADATA FILTER - Final filter built",
            extra_data={
                "final_filter": final_filter,
                "filter_count": len(filters),
                "is_inclusive": True,
                "approach": "tiered_research_filtering"
            }
        )
        
        return final_filter
    
    def _process_unified_search_results(self,
                                      results: Dict[str, Any],
                                      similarity_threshold: float) -> List[RetrievalResult]:
        """Process raw search results from unified collection."""
        processed_results = []
        
        # DEBUG: Log the raw ChromaDB results structure
        self.logger.log_event(
            EventType.KNOWLEDGE_RETRIEVED,
            "Processing raw ChromaDB search results",
            extra_data={
                "results_structure": {
                    "ids_count": len(results.get('ids', [[]])[0]) if results.get('ids') else 0,
                    "documents_count": len(results.get('documents', [[]])[0]) if results.get('documents') else 0,
                    "metadatas_count": len(results.get('metadatas', [[]])[0]) if results.get('metadatas') else 0,
                    "distances_count": len(results.get('distances', [[]])[0]) if results.get('distances') else 0,
                    "has_metadatas": 'metadatas' in results and results['metadatas'] is not None,
                    "metadatas_sample": results.get('metadatas', [[]])[0][:2] if results.get('metadatas') and results['metadatas'][0] else []
                },
                "similarity_threshold": similarity_threshold
            }
        )
        
        if not results['ids'] or not results['ids'][0]:
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                "No results returned from ChromaDB search",
                extra_data={"raw_results": results}
            )
            return processed_results
        
        for i, content_id in enumerate(results['ids'][0]):
            try:
                # Calculate similarity score (ChromaDB returns distances, convert to similarity)
                distance = results['distances'][0][i]
                similarity_score = 1.0 - distance  # Convert distance to similarity
                
                if similarity_score < similarity_threshold:
                    continue
                
                # Extract content and metadata
                content = results['documents'][0][i]
                
                # Safely extract metadata with multiple fallback levels
                metadata = {}
                if (results.get('metadatas') and 
                    len(results['metadatas']) > 0 and 
                    results['metadatas'][0] and 
                    len(results['metadatas'][0]) > i):
                    metadata = results['metadatas'][0][i] or {}
                
                safe_metadata = metadata or {}
                
                # DEBUG: Log the raw metadata structure
                self.logger.log_event(
                    EventType.KNOWLEDGE_RETRIEVED,
                    f"Processing result {i}: content_id={content_id}",
                    extra_data={
                        "metadata_keys": list(safe_metadata.keys()),
                        "metadata_sample": {k: str(v)[:100] for k, v in safe_metadata.items()},
                        "metadata_type": type(metadata).__name__,
                        "has_agent_specialization": "agent_specialization" in safe_metadata
                    }
                )
                
                # Handle programming_concepts - convert from string to list if needed
                programming_concepts = safe_metadata.get('programming_concepts', [])
                if isinstance(programming_concepts, str):
                    programming_concepts = [c.strip() for c in programming_concepts.split(',') if c.strip()]
                elif programming_concepts is None:
                    programming_concepts = []
                
                # Create retrieval result with safe metadata access  
                retrieval_result = RetrievalResult(
                    content_id=content_id,
                    content=content,
                    similarity_score=similarity_score,
                    metadata=safe_metadata,
                    content_type=safe_metadata.get('content_type', 'general'),
                    agent_specialization=safe_metadata.get('agent_specialization', 'shared'),
                    programming_concepts=programming_concepts,
                    difficulty_level=safe_metadata.get('difficulty_level', 'intermediate')
                )
                
            except Exception as e:
                # DEBUG: Log any errors in processing individual results
                self.logger.log_event(
                    EventType.ERROR_OCCURRED,
                    f"Error processing search result {i}: {str(e)}",
                    level="ERROR",
                    extra_data={
                        "content_id": content_id if 'content_id' in locals() else "unknown",
                        "metadata": metadata if 'metadata' in locals() else {},
                        "error_type": type(e).__name__
                    }
                )
                continue
            
            processed_results.append(retrieval_result)
        
        return processed_results
    
    def _update_content_usage_stats(self, results: List[RetrievalResult]):
        """Update usage statistics for retrieved content."""
        current_time = time.time()
        
        for result in results:
            try:
                # Get current metadata
                existing = self.collection.get(ids=[result.content_id])
                if existing['ids']:
                    # Update metadata
                    metadata = existing['metadatas'][0]
                    metadata['retrieval_count'] = metadata.get('retrieval_count', 0) + 1
                    metadata['last_accessed'] = current_time
                    
                    self.collection.update(
                        ids=[result.content_id],
                        metadatas=[metadata]
                    )
            except Exception:
                continue  # Skip if update fails
    
    def _generate_cache_key(self, *args) -> str:
        """Generate cache key for query parameters."""
        key_data = json.dumps([str(arg) for arg in args], sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()


# Global unified vector store instance
_unified_vector_store: Optional[UnifiedEducationalVectorStore] = None


def get_vector_store(reload: bool = False) -> UnifiedEducationalVectorStore:
    """
    Get global unified vector store instance (singleton pattern).
    
    Args:
        reload: Force creation of new vector store instance
        
    Returns:
        UnifiedEducationalVectorStore instance
    """
    global _unified_vector_store
    if _unified_vector_store is None or reload:
        _unified_vector_store = UnifiedEducationalVectorStore()
    return _unified_vector_store


if __name__ == "__main__":
    # Unified vector store test
    try:
        vector_store = get_vector_store()
        
        # Test content addition
        test_metadata = ContentMetadata(
            content_id="test_unified_content_1",
            content_type=ContentType.IMPLEMENTATION_GUIDE,
            agent_specialization=AgentSpecialization.IMPLEMENTATION,
            programming_concepts=["binary_search", "algorithms"],
            difficulty_level="intermediate",
            topic_tags=["searching", "efficiency"]
        )
        
        test_content = """
        Binary search is an efficient algorithm for finding an item from a sorted list.
        It works by repeatedly dividing the search interval in half and comparing the
        target value to the middle element of the interval.
        """
        
        # Add content to unified collection
        success = vector_store.add_content(test_content, test_metadata)
        print(f"Content addition: {'✅ Success' if success else '❌ Failed'}")
        
        # Test unified search with filtering
        search_results = vector_store.search_similar_content(
            query="How do I implement binary search?",
            agent_specialization=AgentSpecialization.IMPLEMENTATION,
            content_type_filter=ContentType.IMPLEMENTATION_GUIDE,
            max_results=3
        )
        
        print(f"Unified search results: {len(search_results)} found")
        for result in search_results:
            print(f"  - {result.content_id}: {result.similarity_score:.3f}")
        
        # Test statistics
        stats = vector_store.get_vector_store_stats()
        print(f"Unified vector store stats:")
        print(f"  - Total documents: {stats.total_documents}")
        print(f"  - By type: {stats.content_by_type}")
        print(f"  - By agent: {stats.content_by_agent}")
        print(f"  - By difficulty: {stats.content_by_difficulty}")
        print(f"  - Total queries: {stats.total_queries}")
        
        print("✅ Unified vector store test completed successfully!")
        
    except Exception as e:
        print(f"❌ Unified vector store test failed: {e}")
        import traceback
        traceback.print_exc()
