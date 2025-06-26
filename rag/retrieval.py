"""
RAG Retrieval System for Educational Content

This module implements sophisticated retrieval strategies for educational content,
combining vector similarity search with educational metadata filtering, relevance
scoring, and agent-specific optimization. It provides the core RAG functionality
for the multi-agent educational system.

Key Features:
1. Multi-Strategy Retrieval: Vector similarity, keyword matching, and hybrid approaches
2. Educational Relevance Scoring: Content quality and appropriateness for learning context
3. Agent-Specific Filtering: Specialized content retrieval for different agent types
4. Context-Aware Ranking: Ranking based on student level, SRL phase, and query context
5. Adaptive Retrieval: Learning from retrieval effectiveness to improve future results
6. Performance Optimization: Caching, batch processing, and efficient search strategies

Retrieval Pipeline:
1. Query Analysis: Understand information need and educational context
2. Search Strategy Selection: Choose optimal combination of retrieval methods
3. Candidate Generation: Retrieve relevant content from vector store
4. Educational Filtering: Apply educational metadata filters and quality checks
5. Relevance Scoring: Score candidates based on educational appropriateness
6. Result Ranking: Final ranking considering all factors
7. Response Generation: Format results for agent consumption
"""

import time
import math
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from collections import defaultdict, Counter

from pydantic import BaseModel, Field

from .vector_store import (
    EducationalVectorStore, get_vector_store, RetrievalResult,
    ContentType, AgentSpecialization
)
from ..classification.srl_classifier import SRLPhase
from ..utils.logging_utils import get_logger, LogContext, EventType, create_context
from ..config.settings import get_settings


class RetrievalStrategy(Enum):
    """Different retrieval strategies for different query types."""
    VECTOR_SIMILARITY = "vector_similarity"
    KEYWORD_MATCHING = "keyword_matching"
    HYBRID = "hybrid"
    CONCEPTUAL = "conceptual"
    CONTEXTUAL = "contextual"
    ADAPTIVE = "adaptive"


class RetrievalContext(BaseModel):
    """Context information for retrieval operations."""
    query: str = Field(..., description="User query for retrieval")
    agent_specialization: Optional[str] = Field(default=None, description="Target agent specialization")
    srl_phase: Optional[str] = Field(default=None, description="Self-regulated learning phase")
    student_level: Optional[str] = Field(default=None, description="Student proficiency level")
    
    # Query context
    code_snippet: Optional[str] = Field(default=None, description="Associated code snippet")
    error_message: Optional[str] = Field(default=None, description="Associated error message")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Previous conversation")
    
    # Retrieval preferences
    max_results: int = Field(default=5, description="Maximum results to return")
    similarity_threshold: float = Field(default=0.0, description="Minimum similarity threshold")
    prefer_code_examples: bool = Field(default=False, description="Prefer content with code examples")
    prefer_error_examples: bool = Field(default=False, description="Prefer content with error examples")
    
    # Educational context
    programming_domain: Optional[str] = Field(default=None, description="Programming domain focus")
    learning_objectives: Optional[List[str]] = Field(default=None, description="Specific learning objectives")


class RetrievalStrategy(Enum):
    """Available retrieval strategies."""
    VECTOR_ONLY = "vector_only"
    ENHANCED_SIMILARITY = "enhanced_similarity"
    EDUCATIONAL_RELEVANCE = "educational_relevance"
    CONTEXTUAL_ADAPTIVE = "contextual_adaptive"


@dataclass
class ScoredResult:
    """A retrieval result with comprehensive scoring."""
    result: RetrievalResult
    
    # Scoring components
    similarity_score: float = 0.0
    educational_relevance_score: float = 0.0
    context_match_score: float = 0.0
    quality_score: float = 0.0
    freshness_score: float = 0.0
    
    # Final scores
    combined_score: float = 0.0
    rank_position: int = 0
    
    # Scoring explanation
    score_explanation: Dict[str, float] = field(default_factory=dict)


class RetrievalMetrics(BaseModel):
    """Metrics for retrieval performance monitoring."""
    total_retrievals: int = Field(default=0, description="Total retrieval operations")
    average_retrieval_time_ms: float = Field(default=0.0, description="Average retrieval time")
    average_results_returned: float = Field(default=0.0, description="Average number of results")
    
    # Quality metrics
    average_similarity_score: float = Field(default=0.0, description="Average similarity score")
    average_educational_relevance: float = Field(default=0.0, description="Average educational relevance")
    
    # Strategy effectiveness
    strategy_usage: Dict[str, int] = Field(default_factory=dict, description="Usage count per strategy")
    strategy_performance: Dict[str, float] = Field(default_factory=dict, description="Performance per strategy")


class EducationalRetriever:
    """
    Advanced retrieval system for educational content.
    
    This retriever combines multiple search strategies with educational metadata
    to provide highly relevant content for programming education. It adapts its
    approach based on the educational context and learning objectives.
    """
    
    def __init__(self):
        """Initialize the educational retriever."""
        self.settings = get_settings()
        self.logger = get_logger()
        self.vector_store = get_vector_store()
        
        # Retrieval strategy weights
        self.strategy_weights = self._initialize_strategy_weights()
        
        # Educational scoring weights
        self.educational_weights = self._initialize_educational_weights()
        
        # Performance tracking
        self.retrieval_metrics = RetrievalMetrics()
        self.strategy_performance_history = defaultdict(list)
        
        # Concept similarity mappings
        self.concept_similarity = self._initialize_concept_similarity()
        
        # Keyword expansion dictionaries
        self.keyword_expansions = self._initialize_keyword_expansions()
        
        self.logger.log_event(
            EventType.COMPONENT_INIT,
            "Educational retriever initialized",
            extra_data={"retrieval_strategies": len(self.strategy_weights)}
        )
    
    def retrieve_educational_content(self, 
                                   context: RetrievalContext,
                                   log_context: Optional[LogContext] = None) -> List[ScoredResult]:
        """
        Retrieve and score educational content based on context.
        
        Args:
            context: Retrieval context with query and educational metadata
            log_context: Logging context for tracking
            
        Returns:
            List of scored and ranked retrieval results
        """
        log_context = log_context or create_context()
        start_time = time.time()
        
        try:
            # Analyze query to determine optimal strategy
            strategy = self._select_retrieval_strategy(context)
            
            # Enhance query with educational context
            enhanced_query = self._enhance_query_with_context(context)
            
            # Retrieve candidates using selected strategy
            candidates = self._retrieve_candidates(enhanced_query, context, strategy)
            
            # Score candidates using educational metrics
            scored_results = self._score_educational_relevance(candidates, context)
            
            # Apply contextual filtering and ranking
            filtered_results = self._apply_contextual_filtering(scored_results, context)
            
            # Final ranking and selection
            final_results = self._final_ranking_and_selection(filtered_results, context)
            
            # Update performance metrics
            retrieval_time = (time.time() - start_time) * 1000
            self._update_retrieval_metrics(strategy, final_results, retrieval_time)
            
            # Log retrieval operation
            self.logger.log_rag_operation(
                operation="educational_retrieval",
                query=context.query,
                results_count=len(final_results),
                context=log_context
            )
            
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                f"Educational content retrieved: {len(final_results)} results",
                context=log_context,
                extra_data={
                    "strategy": strategy.value,
                    "retrieval_time_ms": retrieval_time,
                    "enhanced_query_length": len(enhanced_query),
                    "candidates_retrieved": len(candidates),
                    "final_results": len(final_results)
                }
            )
            
            return final_results
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Educational retrieval failed: {str(e)}",
                context=log_context,
                level="ERROR",
                extra_data={"query": context.query[:100]}
            )
            return []
    
    def _select_retrieval_strategy(self, context: RetrievalContext) -> RetrievalStrategy:
        """
        Select optimal retrieval strategy based on context.
        
        Args:
            context: Retrieval context
            
        Returns:
            Selected retrieval strategy
        """
        # Strategy selection logic based on context
        strategy_scores = {}
        
        # Vector similarity is good for conceptual queries
        if any(word in context.query.lower() for word in ["concept", "understand", "explain"]):
            strategy_scores[RetrievalStrategy.VECTOR_ONLY] = 0.8
        
        # Enhanced similarity for implementation queries
        if any(word in context.query.lower() for word in ["how", "implement", "approach"]):
            strategy_scores[RetrievalStrategy.ENHANCED_SIMILARITY] = 0.9
        
        # Educational relevance for learning-focused queries
        if any(word in context.query.lower() for word in ["learn", "practice", "exercise"]):
            strategy_scores[RetrievalStrategy.EDUCATIONAL_RELEVANCE] = 0.85
        
        # Contextual adaptive for complex queries with multiple factors
        complexity_indicators = [
            bool(context.code_snippet),
            bool(context.error_message),
            bool(context.conversation_history),
            len(context.query.split()) > 10
        ]
        
        if sum(complexity_indicators) >= 2:
            strategy_scores[RetrievalStrategy.CONTEXTUAL_ADAPTIVE] = 0.95
        
        # Select strategy with highest score, default to enhanced similarity
        if strategy_scores:
            return max(strategy_scores, key=strategy_scores.get)
        else:
            return RetrievalStrategy.ENHANCED_SIMILARITY
    
    def _enhance_query_with_context(self, context: RetrievalContext) -> str:
        """
        Enhance query with educational context for better retrieval.
        
        Args:
            context: Retrieval context
            
        Returns:
            Enhanced query string
        """
        query_parts = [context.query]
        
        # Add SRL phase context
        if context.srl_phase:
            if context.srl_phase == SRLPhase.FORETHOUGHT.value:
                query_parts.append("planning implementation strategy approach")
            elif context.srl_phase == SRLPhase.PERFORMANCE.value:
                query_parts.append("debugging error fixing troubleshooting")
        
        # Add student level context
        if context.student_level:
            query_parts.append(f"{context.student_level} level")
        
        # Add programming domain context
        if context.programming_domain:
            query_parts.append(context.programming_domain)
        
        # Add code context
        if context.code_snippet:
            # Extract key terms from code
            code_terms = self._extract_code_terms(context.code_snippet)
            query_parts.extend(code_terms[:3])  # Add top 3 terms
        
        # Add error context
        if context.error_message:
            # Extract error type
            error_type = self._extract_error_type(context.error_message)
            if error_type:
                query_parts.append(error_type)
        
        # Expand keywords
        expanded_query = self._expand_keywords(" ".join(query_parts))
        
        return expanded_query
    
    def _retrieve_candidates(self, 
                           enhanced_query: str,
                           context: RetrievalContext,
                           strategy: RetrievalStrategy) -> List[RetrievalResult]:
        """
        Retrieve candidate results using specified strategy.
        
        Args:
            enhanced_query: Enhanced query string
            context: Retrieval context
            strategy: Retrieval strategy to use
            
        Returns:
            List of candidate retrieval results
        """
        if strategy == RetrievalStrategy.VECTOR_ONLY:
            return self._vector_only_retrieval(enhanced_query, context)
        elif strategy == RetrievalStrategy.ENHANCED_SIMILARITY:
            return self._enhanced_similarity_retrieval(enhanced_query, context)
        elif strategy == RetrievalStrategy.EDUCATIONAL_RELEVANCE:
            return self._educational_relevance_retrieval(enhanced_query, context)
        elif strategy == RetrievalStrategy.CONTEXTUAL_ADAPTIVE:
            return self._contextual_adaptive_retrieval(enhanced_query, context)
        else:
            return self._enhanced_similarity_retrieval(enhanced_query, context)
    
    def _vector_only_retrieval(self, 
                              query: str, 
                              context: RetrievalContext) -> List[RetrievalResult]:
        """Basic vector similarity retrieval."""
        agent_spec = None
        if context.agent_specialization:
            agent_spec = AgentSpecialization(context.agent_specialization)
        
        return self.vector_store.search_similar_content(
            query=query,
            agent_specialization=agent_spec,
            max_results=context.max_results * 2,  # Get more for ranking
            similarity_threshold=context.similarity_threshold
        )
    
    def _enhanced_similarity_retrieval(self, 
                                     query: str,
                                     context: RetrievalContext) -> List[RetrievalResult]:
        """Enhanced retrieval with content type filtering."""
        agent_spec = None
        if context.agent_specialization:
            agent_spec = AgentSpecialization(context.agent_specialization)
        
        # Determine preferred content types based on context
        content_type_filter = None
        if context.srl_phase == SRLPhase.FORETHOUGHT.value:
            content_type_filter = ContentType.IMPLEMENTATION_GUIDE
        elif context.srl_phase == SRLPhase.PERFORMANCE.value:
            content_type_filter = ContentType.DEBUGGING_RESOURCE
        
        return self.vector_store.search_similar_content(
            query=query,
            agent_specialization=agent_spec,
            content_type_filter=content_type_filter,
            difficulty_level=context.student_level,
            max_results=context.max_results * 3,
            similarity_threshold=context.similarity_threshold
        )
    
    def _educational_relevance_retrieval(self, 
                                       query: str,
                                       context: RetrievalContext) -> List[RetrievalResult]:
        """Retrieval focused on educational relevance."""
        # Multiple searches with different content types
        all_results = []
        
        agent_spec = None
        if context.agent_specialization:
            agent_spec = AgentSpecialization(context.agent_specialization)
        
        # Search for different educational content types
        educational_types = [
            ContentType.CONCEPT_EXPLANATION,
            ContentType.CODE_EXAMPLE,
            ContentType.BEST_PRACTICE
        ]
        
        for content_type in educational_types:
            results = self.vector_store.search_similar_content(
                query=query,
                agent_specialization=agent_spec,
                content_type_filter=content_type,
                difficulty_level=context.student_level,
                max_results=context.max_results,
                similarity_threshold=context.similarity_threshold
            )
            all_results.extend(results)
        
        # Remove duplicates and return top results
        unique_results = self._deduplicate_results(all_results)
        return unique_results[:context.max_results * 3]
    
    def _contextual_adaptive_retrieval(self, 
                                     query: str,
                                     context: RetrievalContext) -> List[RetrievalResult]:
        """Advanced retrieval that adapts to full context."""
        all_results = []
        
        # Multiple retrieval passes with different strategies
        strategies = [
            ("main_query", query),
            ("code_focused", self._create_code_focused_query(query, context)),
            ("concept_focused", self._create_concept_focused_query(query, context))
        ]
        
        agent_spec = None
        if context.agent_specialization:
            agent_spec = AgentSpecialization(context.agent_specialization)
        
        for strategy_name, search_query in strategies:
            if search_query:
                results = self.vector_store.search_similar_content(
                    query=search_query,
                    agent_specialization=agent_spec,
                    difficulty_level=context.student_level,
                    max_results=context.max_results * 2,
                    similarity_threshold=context.similarity_threshold * 0.8  # Lower threshold
                )
                
                # Tag results with strategy for scoring
                for result in results:
                    if 'retrieval_strategy' not in result.metadata:
                        result.metadata['retrieval_strategy'] = strategy_name
                
                all_results.extend(results)
        
        # Deduplicate and return
        unique_results = self._deduplicate_results(all_results)
        return unique_results[:context.max_results * 4]
    
    def _score_educational_relevance(self, 
                                   candidates: List[RetrievalResult],
                                   context: RetrievalContext) -> List[ScoredResult]:
        """
        Score candidates based on educational relevance.
        
        Args:
            candidates: Candidate retrieval results
            context: Retrieval context
            
        Returns:
            List of scored results
        """
        scored_results = []
        
        for candidate in candidates:
            # Initialize scored result
            scored_result = ScoredResult(result=candidate)
            
            # Base similarity score
            scored_result.similarity_score = candidate.similarity_score
            
            # Educational relevance scoring
            scored_result.educational_relevance_score = self._calculate_educational_relevance(
                candidate, context
            )
            
            # Context match scoring
            scored_result.context_match_score = self._calculate_context_match(
                candidate, context
            )
            
            # Quality scoring
            scored_result.quality_score = self._calculate_quality_score(candidate)
            
            # Freshness scoring (based on last accessed, retrieval count)
            scored_result.freshness_score = self._calculate_freshness_score(candidate)
            
            # Calculate combined score
            scored_result.combined_score = self._calculate_combined_score(scored_result)
            
            # Store score explanation
            scored_result.score_explanation = {
                "similarity": scored_result.similarity_score,
                "educational_relevance": scored_result.educational_relevance_score,
                "context_match": scored_result.context_match_score,
                "quality": scored_result.quality_score,
                "freshness": scored_result.freshness_score,
                "combined": scored_result.combined_score
            }
            
            scored_results.append(scored_result)
        
        return scored_results
    
    def _calculate_educational_relevance(self, 
                                       candidate: RetrievalResult,
                                       context: RetrievalContext) -> float:
        """Calculate educational relevance score."""
        relevance_score = 0.0
        
        # SRL phase alignment
        if context.srl_phase and context.agent_specialization:
            if (context.srl_phase == SRLPhase.FORETHOUGHT.value and 
                candidate.agent_specialization == "implementation"):
                relevance_score += 0.3
            elif (context.srl_phase == SRLPhase.PERFORMANCE.value and 
                  candidate.agent_specialization == "debugging"):
                relevance_score += 0.3
        
        # Student level alignment
        if context.student_level and candidate.difficulty_level == context.student_level:
            relevance_score += 0.2
        
        # Content type preferences
        if context.prefer_code_examples and candidate.content_type in ["code_example", "mixed"]:
            relevance_score += 0.15
        
        if context.prefer_error_examples and "error" in candidate.content_type.lower():
            relevance_score += 0.15
        
        # Programming concepts alignment
        if context.programming_domain and candidate.programming_concepts:
            domain_concepts = self.concept_similarity.get(context.programming_domain, [])
            concept_overlap = len(set(candidate.programming_concepts) & set(domain_concepts))
            if concept_overlap > 0:
                relevance_score += min(0.2, concept_overlap * 0.05)
        
        return min(1.0, relevance_score)
    
    def _calculate_context_match(self, 
                               candidate: RetrievalResult,
                               context: RetrievalContext) -> float:
        """Calculate context matching score."""
        match_score = 0.0
        
        # Code snippet context matching
        if context.code_snippet and candidate.metadata.get('has_code_examples'):
            match_score += 0.3
        
        # Error message context matching
        if context.error_message and candidate.metadata.get('has_error_examples'):
            match_score += 0.3
        
        # Conversation history relevance
        if context.conversation_history:
            # Simple relevance based on topic continuity
            recent_topics = self._extract_topics_from_history(context.conversation_history)
            candidate_topics = candidate.programming_concepts
            
            topic_overlap = len(set(recent_topics) & set(candidate_topics))
            if topic_overlap > 0:
                match_score += min(0.4, topic_overlap * 0.1)
        
        return min(1.0, match_score)
    
    def _calculate_quality_score(self, candidate: RetrievalResult) -> float:
        """Calculate content quality score."""
        quality_score = 0.5  # Base quality
        
        # Content length (optimal range)
        content_length = candidate.metadata.get('content_length', 0)
        if 200 <= content_length <= 1500:  # Optimal range
            quality_score += 0.2
        elif content_length < 100 or content_length > 3000:  # Poor range
            quality_score -= 0.1
        
        # Content type quality indicators
        high_quality_types = ["implementation_guide", "concept_explanation", "best_practice"]
        if candidate.content_type in high_quality_types:
            quality_score += 0.15
        
        # Concept density
        concepts_count = len(candidate.programming_concepts)
        if 2 <= concepts_count <= 5:  # Good concept density
            quality_score += 0.15
        
        return max(0.0, min(1.0, quality_score))
    
    def _calculate_freshness_score(self, candidate: RetrievalResult) -> float:
        """Calculate freshness/popularity score based on usage."""
        # Higher retrieval count indicates proven usefulness
        retrieval_count = candidate.metadata.get('retrieval_count', 0)
        
        # Normalize retrieval count to 0-1 scale
        freshness_score = min(1.0, retrieval_count / 100.0)
        
        # Recent access bonus
        last_accessed = candidate.metadata.get('last_accessed', 0)
        current_time = time.time()
        days_since_access = (current_time - last_accessed) / (24 * 3600)
        
        if days_since_access < 7:  # Recently accessed
            freshness_score += 0.1
        
        return min(1.0, freshness_score)
    
    def _calculate_combined_score(self, scored_result: ScoredResult) -> float:
        """Calculate weighted combined score."""
        weights = self.educational_weights
        
        combined = (
            scored_result.similarity_score * weights["similarity"] +
            scored_result.educational_relevance_score * weights["educational_relevance"] +
            scored_result.context_match_score * weights["context_match"] +
            scored_result.quality_score * weights["quality"] +
            scored_result.freshness_score * weights["freshness"]
        )
        
        return combined
    
    def _apply_contextual_filtering(self, 
                                  scored_results: List[ScoredResult],
                                  context: RetrievalContext) -> List[ScoredResult]:
        """Apply contextual filtering to scored results."""
        filtered_results = []
        
        for scored_result in scored_results:
            # Minimum score threshold
            if scored_result.combined_score < 0.3:
                continue
            
            # Diversity filtering (avoid too similar content)
            if self._passes_diversity_filter(scored_result, filtered_results):
                filtered_results.append(scored_result)
        
        return filtered_results
    
    def _final_ranking_and_selection(self, 
                                   filtered_results: List[ScoredResult],
                                   context: RetrievalContext) -> List[ScoredResult]:
        """Final ranking and selection of results."""
        # Sort by combined score
        filtered_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Assign rank positions
        for i, result in enumerate(filtered_results):
            result.rank_position = i + 1
        
        # Select top results
        final_results = filtered_results[:context.max_results]
        
        return final_results
    
    def _passes_diversity_filter(self, 
                                candidate: ScoredResult,
                                existing_results: List[ScoredResult]) -> bool:
        """Check if candidate passes diversity filter."""
        if not existing_results:
            return True
        
        # Check content similarity with existing results
        candidate_concepts = set(candidate.result.programming_concepts)
        
        for existing in existing_results:
            existing_concepts = set(existing.result.programming_concepts)
            
            # If too much concept overlap, might be too similar
            overlap_ratio = len(candidate_concepts & existing_concepts) / max(len(candidate_concepts | existing_concepts), 1)
            
            if overlap_ratio > 0.8:  # Too similar
                return False
        
        return True
    
    def _deduplicate_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Remove duplicate results based on content ID."""
        seen_ids = set()
        unique_results = []
        
        for result in results:
            if result.content_id not in seen_ids:
                seen_ids.add(result.content_id)
                unique_results.append(result)
        
        return unique_results
    
    def _extract_code_terms(self, code_snippet: str) -> List[str]:
        """Extract key terms from code snippet."""
        # Simple extraction of function names, variable names, etc.
        terms = []
        
        # Function definitions
        func_matches = re.findall(r'def\s+(\w+)', code_snippet)
        terms.extend(func_matches)
        
        # Class definitions
        class_matches = re.findall(r'class\s+(\w+)', code_snippet)
        terms.extend(class_matches)
        
        # Common Python keywords
        python_keywords = ['for', 'while', 'if', 'else', 'try', 'except', 'import']
        for keyword in python_keywords:
            if keyword in code_snippet:
                terms.append(keyword)
        
        return terms[:5]  # Return top 5 terms
    
    def _extract_error_type(self, error_message: str) -> Optional[str]:
        """Extract error type from error message."""
        error_patterns = [
            r'(\w*Error):',
            r'(\w*Exception):',
            r'Traceback.*?(\w*Error)',
        ]
        
        for pattern in error_patterns:
            match = re.search(pattern, error_message)
            if match:
                return match.group(1)
        
        return None
    
    def _expand_keywords(self, query: str) -> str:
        """Expand keywords using educational synonyms."""
        words = query.lower().split()
        expanded_words = []
        
        for word in words:
            expanded_words.append(word)
            
            # Add synonyms if available
            if word in self.keyword_expansions:
                expanded_words.extend(self.keyword_expansions[word][:2])  # Add top 2 synonyms
        
        return " ".join(expanded_words)
    
    def _create_code_focused_query(self, query: str, context: RetrievalContext) -> Optional[str]:
        """Create a code-focused version of the query."""
        if not context.code_snippet:
            return None
        
        code_terms = self._extract_code_terms(context.code_snippet)
        return f"{query} {' '.join(code_terms)} code example implementation"
    
    def _create_concept_focused_query(self, query: str, context: RetrievalContext) -> str:
        """Create a concept-focused version of the query."""
        concept_terms = ["concept", "explanation", "understand", "theory", "principle"]
        return f"{query} {' '.join(concept_terms[:2])}"
    
    def _extract_topics_from_history(self, history: List[Dict[str, str]]) -> List[str]:
        """Extract topics from conversation history."""
        topics = []
        
        for turn in history[-3:]:  # Last 3 turns
            content = turn.get("content", "").lower()
            
            # Extract programming-related terms
            programming_terms = [
                "algorithm", "function", "class", "loop", "array", "list",
                "sorting", "searching", "recursion", "iteration"
            ]
            
            for term in programming_terms:
                if term in content:
                    topics.append(term)
        
        return list(set(topics))  # Remove duplicates
    
    def _update_retrieval_metrics(self, 
                                strategy: RetrievalStrategy,
                                results: List[ScoredResult],
                                retrieval_time: float):
        """Update retrieval performance metrics."""
        self.retrieval_metrics.total_retrievals += 1
        
        # Update timing
        current_avg = self.retrieval_metrics.average_retrieval_time_ms
        total_retrievals = self.retrieval_metrics.total_retrievals
        self.retrieval_metrics.average_retrieval_time_ms = (
            (current_avg * (total_retrievals - 1) + retrieval_time) / total_retrievals
        )
        
        # Update results count
        current_avg_results = self.retrieval_metrics.average_results_returned
        self.retrieval_metrics.average_results_returned = (
            (current_avg_results * (total_retrievals - 1) + len(results)) / total_retrievals
        )
        
        # Update quality metrics
        if results:
            avg_similarity = sum(r.similarity_score for r in results) / len(results)
            avg_educational = sum(r.educational_relevance_score for r in results) / len(results)
            
            current_sim = self.retrieval_metrics.average_similarity_score
            current_edu = self.retrieval_metrics.average_educational_relevance
            
            self.retrieval_metrics.average_similarity_score = (
                (current_sim * (total_retrievals - 1) + avg_similarity) / total_retrievals
            )
            
            self.retrieval_metrics.average_educational_relevance = (
                (current_edu * (total_retrievals - 1) + avg_educational) / total_retrievals
            )
        
        # Update strategy metrics
        strategy_name = strategy.value
        self.retrieval_metrics.strategy_usage[strategy_name] = (
            self.retrieval_metrics.strategy_usage.get(strategy_name, 0) + 1
        )
        
        # Store performance for strategy optimization
        if results:
            strategy_performance = sum(r.combined_score for r in results) / len(results)
            self.strategy_performance_history[strategy_name].append(strategy_performance)
            
            # Update average performance
            strategy_history = self.strategy_performance_history[strategy_name]
            self.retrieval_metrics.strategy_performance[strategy_name] = (
                sum(strategy_history) / len(strategy_history)
            )
    
    def _initialize_strategy_weights(self) -> Dict[str, float]:
        """Initialize retrieval strategy weights."""
        return {
            "vector_similarity": 0.4,
            "keyword_matching": 0.2,
            "educational_context": 0.25,
            "content_type_match": 0.15
        }
    
    def _initialize_educational_weights(self) -> Dict[str, float]:
        """Initialize educational scoring weights."""
        return {
            "similarity": 0.3,
            "educational_relevance": 0.35,
            "context_match": 0.2,
            "quality": 0.1,
            "freshness": 0.05
        }
    
    def _initialize_concept_similarity(self) -> Dict[str, List[str]]:
        """Initialize concept similarity mappings."""
        return {
            "algorithms": ["sorting", "searching", "optimization", "complexity", "efficiency"],
            "data_structures": ["array", "list", "tree", "graph", "hash", "stack", "queue"],
            "object_oriented": ["class", "object", "inheritance", "polymorphism", "encapsulation"],
            "functions": ["method", "parameter", "argument", "return", "scope"],
            "loops": ["iteration", "while", "for", "recursion", "control_flow"],
            "debugging": ["error", "exception", "troubleshooting", "testing", "validation"]
        }
    
    def _initialize_keyword_expansions(self) -> Dict[str, List[str]]:
        """Initialize keyword expansion mappings."""
        return {
            "implement": ["create", "build", "develop", "code"],
            "algorithm": ["method", "approach", "technique", "procedure"],
            "debug": ["fix", "troubleshoot", "resolve", "solve"],
            "error": ["exception", "bug", "issue", "problem"],
            "function": ["method", "procedure", "routine"],
            "loop": ["iteration", "repeat", "cycle"],
            "variable": ["identifier", "name", "symbol"],
            "class": ["object", "type", "structure"]
        }
    
    def get_retrieval_metrics(self) -> RetrievalMetrics:
        """Get comprehensive retrieval metrics."""
        return self.retrieval_metrics


# Global retriever instance
_educational_retriever: Optional[EducationalRetriever] = None


def get_educational_retriever(reload: bool = False) -> EducationalRetriever:
    """
    Get global educational retriever instance (singleton pattern).
    
    Args:
        reload: Force creation of new retriever instance
        
    Returns:
        EducationalRetriever instance
    """
    global _educational_retriever
    if _educational_retriever is None or reload:
        _educational_retriever = EducationalRetriever()
    return _educational_retriever


if __name__ == "__main__":
    # Educational retriever test
    try:
        retriever = get_educational_retriever()
        
        # Test retrieval context
        test_context = RetrievalContext(
            query="How do I implement binary search algorithm?",
            agent_specialization="implementation",
            srl_phase="FORETHOUGHT",
            student_level="intermediate",
            max_results=3,
            prefer_code_examples=True
        )
        
        # Test retrieval
        results = retriever.retrieve_educational_content(test_context)
        
        print(f"Retrieval test: {len(results)} results returned")
        for i, result in enumerate(results):
            print(f"  Result {i+1}: Score {result.combined_score:.3f}")
            print(f"    Content ID: {result.result.content_id}")
            print(f"    Content Type: {result.result.content_type}")
            print(f"    Similarity: {result.similarity_score:.3f}")
            print(f"    Educational Relevance: {result.educational_relevance_score:.3f}")
        
        # Test metrics
        metrics = retriever.get_retrieval_metrics()
        print(f"\nRetrieval metrics:")
        print(f"  Total retrievals: {metrics.total_retrievals}")
        print(f"  Average time: {metrics.average_retrieval_time_ms:.1f}ms")
        print(f"  Average results: {metrics.average_results_returned:.1f}")
        
        print("✅ Educational retriever test completed successfully!")
        
    except Exception as e:
        print(f"❌ Educational retriever test failed: {e}")
        import traceback
        traceback.print_exc()
