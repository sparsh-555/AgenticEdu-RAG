"""
Self-Regulated Learning (SRL) Phase Classifier

This module implements the core classification system that determines which phase of
Self-Regulated Learning a student query represents. Based on educational research,
it distinguishes between:

1. FORETHOUGHT PHASE: Planning, strategy selection, goal setting (→ Implementation Agent)
2. PERFORMANCE PHASE: Self-monitoring, error correction, help-seeking (→ Debugging Agent)

Theoretical Foundation:
This classifier is based on Zimmerman's cyclical model of Self-Regulated Learning,
which identifies three phases of learning processes. By automatically detecting
which phase a student is in, we can provide appropriately aligned educational support.

Key Research Concepts Applied:
- Forethought Phase: Goal setting, strategic planning, self-efficacy beliefs
- Performance Phase: Self-monitoring, self-control, help-seeking behavior
- Phase-Specific Support: Different pedagogical approaches for each phase

The classifier uses advanced prompt engineering and LLM reasoning to achieve
research-quality classification accuracy for educational applications.
"""

import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re

from pydantic import BaseModel, Field, validator

from .classification_prompts import (
    get_classification_prompt, 
    get_few_shot_prompt,
    get_confidence_adjusted_prompt,
    CLASSIFICATION_EXAMPLES
)
from ..utils.api_utils import get_openai_client, ClassificationResult, OpenAIError
from ..utils.logging_utils import get_logger, LogContext, EventType, create_context
from ..config.settings import get_settings


class SRLPhase(Enum):
    """Self-Regulated Learning phases based on Zimmerman's model."""
    FORETHOUGHT = "FORETHOUGHT"    # Planning, strategy selection, goal setting
    PERFORMANCE = "PERFORMANCE"    # Self-monitoring, error correction, help-seeking


class ClassificationStrategy(Enum):
    """Different strategies for classification based on query complexity."""
    STANDARD = "standard"          # Standard classification prompt
    FEW_SHOT = "few_shot"         # Few-shot learning with examples
    MULTI_STAGE = "multi_stage"   # Multiple classification attempts
    DOMAIN_SPECIFIC = "domain_specific"  # Domain-aware classification
    CONVERSATION_AWARE = "conversation_aware"  # Context from previous turns


@dataclass
class ClassificationContext:
    """Context information for SRL classification."""
    query: str
    code_snippet: Optional[str] = None
    error_message: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    student_level: Optional[str] = None  # "beginner", "intermediate", "advanced"
    domain: Optional[str] = None  # "algorithms", "data_structures", "web_dev", etc.
    previous_classifications: Optional[List[str]] = None
    session_context: Optional[Dict[str, Any]] = None


class EnhancedClassificationResult(BaseModel):
    """Enhanced classification result with educational metadata."""
    classification: SRLPhase = Field(..., description="Predicted SRL phase")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    reasoning: str = Field(..., description="Explanation of classification decision")
    indicators: List[str] = Field(default_factory=list, description="Key indicators found")
    
    # Educational metadata
    query_complexity: str = Field(default="medium", description="Complexity assessment")
    domain_detected: Optional[str] = Field(default=None, description="Programming domain identified")
    student_intent: str = Field(default="unknown", description="Inferred student intent")
    educational_priority: str = Field(default="medium", description="Educational priority level")
    
    # Technical metadata
    classification_strategy: str = Field(default="standard", description="Strategy used")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    fallback_used: bool = Field(default=False, description="Whether fallback was needed")
    
    @validator('classification', pre=True)
    def validate_classification(cls, v):
        if isinstance(v, str):
            return SRLPhase(v)
        return v


class SRLClassifier:
    """
    Self-Regulated Learning phase classifier for educational query routing.
    
    This classifier applies SRL theory to automatically determine whether a student
    query represents forethought phase (planning/implementation) or performance 
    phase (debugging/monitoring) learning needs.
    
    Key Features:
    - Multiple classification strategies for different query types
    - Confidence assessment and validation
    - Educational metadata extraction
    - Fallback mechanisms for reliability
    - Performance monitoring and optimization
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 use_few_shot: bool = True,
                 enable_multi_stage: bool = True,
                 cache_classifications: bool = True):
        """
        Initialize the SRL classifier with configuration options.
        
        Args:
            confidence_threshold: Minimum confidence for accepting classification
            use_few_shot: Whether to use few-shot examples for better accuracy
            enable_multi_stage: Whether to use multi-stage classification for uncertain cases
            cache_classifications: Whether to cache results for similar queries
        """
        self.settings = get_settings()
        self.logger = get_logger()
        self.openai_client = get_openai_client()
        
        self.confidence_threshold = confidence_threshold
        self.use_few_shot = use_few_shot
        self.enable_multi_stage = enable_multi_stage
        self.cache_classifications = cache_classifications
        
        # Performance tracking
        self.classification_count = 0
        self.total_processing_time = 0.0
        self.accuracy_metrics = []
        
        # Classification cache for performance optimization
        self._classification_cache = {} if cache_classifications else None
        
        self.logger.log_event(
            EventType.SYSTEM_START,
            "SRL Classifier initialized",
            extra_data={
                "confidence_threshold": confidence_threshold,
                "few_shot_enabled": use_few_shot,
                "multi_stage_enabled": enable_multi_stage
            }
        )
    
    def classify_query(self, 
                      context: ClassificationContext,
                      log_context: Optional[LogContext] = None) -> EnhancedClassificationResult:
        """
        Classify a student query into SRL phases with comprehensive analysis.
        
        This is the main entry point for classification, implementing a multi-strategy
        approach that adapts to query complexity and available context.
        
        Args:
            context: Classification context with query and metadata
            log_context: Logging context for tracking
            
        Returns:
            Enhanced classification result with educational metadata
            
        Raises:
            RuntimeError: If classification fails completely
        """
        start_time = time.time()
        log_context = log_context or create_context()
        
        try:
            # Check cache first if enabled
            if self._classification_cache:
                cached_result = self._check_cache(context)
                if cached_result:
                    self.logger.log_event(
                        EventType.SRL_CLASSIFICATION,
                        "Classification served from cache",
                        context=log_context,
                        extra_data={"cached": True}
                    )
                    return cached_result
            
            # Determine optimal classification strategy
            strategy = self._determine_classification_strategy(context)
            
            # Perform classification using selected strategy
            result = self._classify_with_strategy(context, strategy, log_context)
            
            # Validate and potentially refine result
            if result.confidence < self.confidence_threshold and self.enable_multi_stage:
                result = self._multi_stage_classification(context, result, log_context)
            
            # Add processing metadata
            processing_time = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time
            result.classification_strategy = strategy.value
            
            # Extract educational metadata
            self._enhance_with_educational_metadata(context, result)
            
            # Cache result if enabled
            if self._classification_cache:
                self._cache_result(context, result)
            
            # Log classification
            self._log_classification(context, result, log_context)
            
            # Update performance tracking
            self.classification_count += 1
            self.total_processing_time += processing_time
            
            return result
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"SRL classification failed: {str(e)}",
                context=log_context,
                level="ERROR",
                extra_data={"query_preview": context.query[:100]}
            )
            
            # Return fallback classification
            return self._create_fallback_result(context, str(e))
    
    def _determine_classification_strategy(self, context: ClassificationContext) -> ClassificationStrategy:
        """
        Determine the optimal classification strategy based on available context.
        
        Args:
            context: Classification context
            
        Returns:
            Recommended classification strategy
        """
        # Use conversation-aware strategy if we have conversation history
        if context.conversation_history and len(context.conversation_history) > 0:
            return ClassificationStrategy.CONVERSATION_AWARE
        
        # Use domain-specific strategy if we can identify the domain
        if self._detect_programming_domain(context.query):
            return ClassificationStrategy.DOMAIN_SPECIFIC
        
        # Use few-shot strategy for complex or ambiguous queries
        if self.use_few_shot and (len(context.query) > 200 or self._is_ambiguous_query(context.query)):
            return ClassificationStrategy.FEW_SHOT
        
        # Use multi-stage for queries with mixed indicators
        if self._has_mixed_indicators(context.query):
            return ClassificationStrategy.MULTI_STAGE
        
        # Default to standard strategy
        return ClassificationStrategy.STANDARD
    
    def _classify_with_strategy(self, 
                               context: ClassificationContext,
                               strategy: ClassificationStrategy,
                               log_context: LogContext) -> EnhancedClassificationResult:
        """
        Perform classification using the specified strategy.
        
        Args:
            context: Classification context
            strategy: Classification strategy to use
            log_context: Logging context
            
        Returns:
            Classification result
        """
        if strategy == ClassificationStrategy.FEW_SHOT:
            return self._few_shot_classification(context, log_context)
        elif strategy == ClassificationStrategy.DOMAIN_SPECIFIC:
            return self._domain_specific_classification(context, log_context)
        elif strategy == ClassificationStrategy.CONVERSATION_AWARE:
            return self._conversation_aware_classification(context, log_context)
        elif strategy == ClassificationStrategy.MULTI_STAGE:
            return self._multi_stage_classification(context, None, log_context)
        else:
            return self._standard_classification(context, log_context)
    
    def _standard_classification(self, 
                               context: ClassificationContext,
                               log_context: LogContext) -> EnhancedClassificationResult:
        """
        Perform standard SRL classification using the base prompt.
        
        Args:
            context: Classification context
            log_context: Logging context
            
        Returns:
            Classification result
        """
        prompt = get_classification_prompt("standard")
        return self._execute_classification(context, prompt, log_context, "standard")
    
    def _few_shot_classification(self, 
                               context: ClassificationContext,
                               log_context: LogContext) -> EnhancedClassificationResult:
        """
        Perform few-shot classification using examples for better accuracy.
        
        Args:
            context: Classification context
            log_context: Logging context
            
        Returns:
            Classification result
        """
        prompt = get_few_shot_prompt(num_examples=3)
        return self._execute_classification(context, prompt, log_context, "few_shot")
    
    def _domain_specific_classification(self, 
                                      context: ClassificationContext,
                                      log_context: LogContext) -> EnhancedClassificationResult:
        """
        Perform domain-specific classification with specialized context.
        
        Args:
            context: Classification context
            log_context: Logging context
            
        Returns:
            Classification result
        """
        domain = self._detect_programming_domain(context.query)
        prompt = get_classification_prompt("domain_specific", domain=domain)
        result = self._execute_classification(context, prompt, log_context, "domain_specific")
        result.domain_detected = domain
        return result
    
    def _conversation_aware_classification(self, 
                                         context: ClassificationContext,
                                         log_context: LogContext) -> EnhancedClassificationResult:
        """
        Perform classification considering conversation history.
        
        Args:
            context: Classification context
            log_context: Logging context
            
        Returns:
            Classification result
        """
        # Format conversation history
        history_text = ""
        if context.conversation_history:
            for i, turn in enumerate(context.conversation_history[-3:]):  # Last 3 turns
                history_text += f"Turn {i+1}: {turn}\n"
        
        prompt = get_classification_prompt("conversation", conversation_history=history_text)
        return self._execute_classification(context, prompt, log_context, "conversation_aware")
    
    def _multi_stage_classification(self, 
                                  context: ClassificationContext,
                                  initial_result: Optional[EnhancedClassificationResult],
                                  log_context: LogContext) -> EnhancedClassificationResult:
        """
        Perform multi-stage classification for uncertain or complex cases.
        
        Args:
            context: Classification context
            initial_result: Initial classification result (if any)
            log_context: Logging context
            
        Returns:
            Refined classification result
        """
        # If we don't have an initial result, get one
        if initial_result is None:
            initial_result = self._standard_classification(context, log_context)
        
        # If confidence is already high, return the result
        if initial_result.confidence >= 0.9:
            return initial_result
        
        # Perform secondary classification with edge case handling
        prompt = get_classification_prompt("edge_case")
        secondary_result = self._execute_classification(context, prompt, log_context, "multi_stage_secondary")
        
        # Compare results and choose the more confident one
        if secondary_result.confidence > initial_result.confidence:
            secondary_result.classification_strategy = "multi_stage"
            return secondary_result
        else:
            initial_result.classification_strategy = "multi_stage"
            return initial_result
    
    def _execute_classification(self, 
                              context: ClassificationContext,
                              prompt: str,
                              log_context: LogContext,
                              strategy_name: str) -> EnhancedClassificationResult:
        """
        Execute the actual classification using the OpenAI API.
        
        Args:
            context: Classification context
            prompt: Classification prompt to use
            log_context: Logging context
            strategy_name: Name of strategy being used
            
        Returns:
            Classification result
            
        Raises:
            RuntimeError: If classification fails
        """
        # Construct the user message with all available context
        user_parts = [f"Query: {context.query}"]
        
        if context.code_snippet:
            user_parts.append(f"Code snippet:\n```\n{context.code_snippet}\n```")
        
        if context.error_message:
            user_parts.append(f"Error message: {context.error_message}")
        
        if context.student_level:
            user_parts.append(f"Student level: {context.student_level}")
        
        user_message = "\n\n".join(user_parts)
        
        # Prepare messages for the API call
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message}
        ]
        
        try:
            # Make API call with specific context for classification
            with self.logger.performance_timer("srl_classification", log_context):
                api_response = self.openai_client.create_chat_completion(
                    messages=messages,
                    temperature=0.1,  # Low temperature for consistent classification
                    context=log_context
                )
            
            # Parse the response
            result = self._parse_classification_response(api_response.content, strategy_name)
            
            # Add token usage information
            if hasattr(api_response, 'usage'):
                result.token_usage = {
                    "prompt_tokens": api_response.usage.prompt_tokens,
                    "completion_tokens": api_response.usage.completion_tokens,
                    "total_tokens": api_response.usage.total_tokens
                }
            
            return result
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Classification execution failed with {strategy_name}: {str(e)}",
                context=log_context,
                level="ERROR"
            )
            
            # Try fallback classification
            return self._fallback_classification(context, str(e))
    
    def _parse_classification_response(self, 
                                     response_content: str,
                                     strategy_name: str) -> EnhancedClassificationResult:
        """
        Parse the LLM response into a structured classification result.
        
        Args:
            response_content: Raw response from LLM
            strategy_name: Strategy used for classification
            
        Returns:
            Parsed classification result
        """
        try:
            # Try to parse as JSON first
            if "{" in response_content and "}" in response_content:
                # Extract JSON from response
                json_start = response_content.find("{")
                json_end = response_content.rfind("}") + 1
                json_str = response_content[json_start:json_end]
                
                data = json.loads(json_str)
                
                return EnhancedClassificationResult(
                    classification=SRLPhase(data["classification"]),
                    confidence=float(data["confidence"]),
                    reasoning=data.get("reasoning", ""),
                    indicators=data.get("indicators", []),
                    classification_strategy=strategy_name
                )
            
            # Fallback: look for classification keywords
            response_upper = response_content.upper()
            if "FORETHOUGHT" in response_upper:
                classification = SRLPhase.FORETHOUGHT
                confidence = 0.6  # Lower confidence for fallback parsing
            elif "PERFORMANCE" in response_upper:
                classification = SRLPhase.PERFORMANCE
                confidence = 0.6
            else:
                # Default to forethought for educational scaffolding
                classification = SRLPhase.FORETHOUGHT
                confidence = 0.3
            
            return EnhancedClassificationResult(
                classification=classification,
                confidence=confidence,
                reasoning="Parsed from fallback keyword detection",
                indicators=["fallback_parsing"],
                classification_strategy=strategy_name,
                fallback_used=True
            )
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Failed to parse classification response: {str(e)}",
                level="WARNING",
                extra_data={"response_preview": response_content[:200]}
            )
            
            # Ultimate fallback
            return EnhancedClassificationResult(
                classification=SRLPhase.FORETHOUGHT,  # Default to implementation support
                confidence=0.2,
                reasoning=f"Parsing failed, defaulting to forethought phase: {str(e)}",
                indicators=["parsing_error"],
                classification_strategy=strategy_name,
                fallback_used=True
            )
    
    def _fallback_classification(self, 
                               context: ClassificationContext,
                               error_message: str) -> EnhancedClassificationResult:
        """
        Perform simple rule-based classification as fallback.
        
        Args:
            context: Classification context
            error_message: Error that triggered fallback
            
        Returns:
            Fallback classification result
        """
        query_lower = context.query.lower()
        
        # Simple keyword-based classification
        debugging_keywords = [
            "error", "bug", "fix", "debug", "wrong", "doesn't work",
            "not working", "broken", "crash", "exception", "fails"
        ]
        
        implementation_keywords = [
            "how to", "implement", "approach", "strategy", "plan",
            "design", "algorithm", "best way", "structure"
        ]
        
        # Check for debugging indicators
        debug_score = sum(1 for keyword in debugging_keywords if keyword in query_lower)
        impl_score = sum(1 for keyword in implementation_keywords if keyword in query_lower)
        
        # Presence of error message strongly indicates debugging
        if context.error_message:
            debug_score += 3
        
        # Presence of code snippet might indicate debugging
        if context.code_snippet:
            debug_score += 1
        
        if debug_score > impl_score:
            classification = SRLPhase.PERFORMANCE
            confidence = min(0.6 + (debug_score - impl_score) * 0.1, 0.8)
        else:
            classification = SRLPhase.FORETHOUGHT
            confidence = min(0.6 + (impl_score - debug_score) * 0.1, 0.8)
        
        return EnhancedClassificationResult(
            classification=classification,
            confidence=confidence,
            reasoning=f"Fallback rule-based classification due to: {error_message}",
            indicators=[f"debug_score:{debug_score}", f"impl_score:{impl_score}"],
            classification_strategy="fallback_rules",
            fallback_used=True
        )
    
    def _enhance_with_educational_metadata(self, 
                                         context: ClassificationContext,
                                         result: EnhancedClassificationResult):
        """
        Add educational metadata to the classification result.
        
        Args:
            context: Classification context
            result: Classification result to enhance (modified in place)
        """
        # Assess query complexity
        result.query_complexity = self._assess_query_complexity(context)
        
        # Detect programming domain if not already done
        if not result.domain_detected:
            result.domain_detected = self._detect_programming_domain(context.query)
        
        # Infer student intent
        result.student_intent = self._infer_student_intent(context, result)
        
        # Assess educational priority
        result.educational_priority = self._assess_educational_priority(context, result)
    
    def _assess_query_complexity(self, context: ClassificationContext) -> str:
        """Assess the complexity level of the query."""
        complexity_score = 0
        
        # Length-based complexity
        if len(context.query) > 300:
            complexity_score += 2
        elif len(context.query) > 150:
            complexity_score += 1
        
        # Code presence increases complexity
        if context.code_snippet:
            complexity_score += 2
            if len(context.code_snippet) > 500:
                complexity_score += 1
        
        # Error messages add complexity
        if context.error_message:
            complexity_score += 1
        
        # Multiple concepts indicate complexity
        concept_indicators = ["algorithm", "data structure", "object", "class", "function", "loop"]
        concept_count = sum(1 for concept in concept_indicators if concept in context.query.lower())
        complexity_score += min(concept_count, 3)
        
        if complexity_score >= 6:
            return "high"
        elif complexity_score >= 3:
            return "medium"
        else:
            return "low"
    
    def _detect_programming_domain(self, query: str) -> Optional[str]:
        """Detect the programming domain of the query."""
        query_lower = query.lower()
        
        domain_keywords = {
            "algorithms": ["algorithm", "sorting", "searching", "complexity", "big o"],
            "data_structures": ["array", "list", "stack", "queue", "tree", "graph", "hash"],
            "object_oriented": ["class", "object", "inheritance", "polymorphism", "encapsulation"],
            "web_development": ["html", "css", "javascript", "web", "http", "api", "server"],
            "databases": ["sql", "database", "query", "table", "join", "mysql", "postgresql"],
            "machine_learning": ["ml", "ai", "neural", "model", "training", "prediction"],
            "game_development": ["game", "sprite", "physics", "collision", "graphics"],
            "mobile_development": ["mobile", "android", "ios", "app", "swift", "kotlin"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        
        return None
    
    def _infer_student_intent(self, 
                            context: ClassificationContext,
                            result: EnhancedClassificationResult) -> str:
        """Infer the student's learning intent."""
        query_lower = context.query.lower()
        
        # Intent patterns
        if any(phrase in query_lower for phrase in ["learn", "understand", "explain"]):
            return "conceptual_learning"
        elif any(phrase in query_lower for phrase in ["homework", "assignment", "project"]):
            return "academic_work"
        elif any(phrase in query_lower for phrase in ["practice", "exercise", "challenge"]):
            return "skill_practice"
        elif any(phrase in query_lower for phrase in ["work", "job", "career", "interview"]):
            return "professional_development"
        elif result.classification == SRLPhase.PERFORMANCE:
            return "problem_solving"
        else:
            return "skill_building"
    
    def _assess_educational_priority(self, 
                                   context: ClassificationContext,
                                   result: EnhancedClassificationResult) -> str:
        """Assess the educational priority level."""
        priority_score = 0
        
        # High priority for clear errors or urgent help-seeking
        if context.error_message:
            priority_score += 2
        
        # High priority for frustrated language
        frustrated_indicators = ["stuck", "frustrated", "don't understand", "confused", "help"]
        if any(indicator in context.query.lower() for indicator in frustrated_indicators):
            priority_score += 2
        
        # Medium priority for planning needs
        if result.classification == SRLPhase.FORETHOUGHT:
            priority_score += 1
        
        # Lower priority for general questions
        if any(phrase in context.query.lower() for phrase in ["best practice", "opinion", "prefer"]):
            priority_score -= 1
        
        if priority_score >= 3:
            return "high"
        elif priority_score >= 1:
            return "medium"
        else:
            return "low"
    
    def _is_ambiguous_query(self, query: str) -> bool:
        """Check if query contains ambiguous language that might need few-shot examples."""
        ambiguous_indicators = [
            "help", "issue", "problem", "trouble", "stuck", "confused",
            "something", "anything", "somehow", "maybe", "might", "could"
        ]
        return any(indicator in query.lower() for indicator in ambiguous_indicators)
    
    def _has_mixed_indicators(self, query: str) -> bool:
        """Check if query has both forethought and performance indicators."""
        query_lower = query.lower()
        
        forethought_indicators = ["how to", "approach", "strategy", "plan", "design"]
        performance_indicators = ["error", "fix", "debug", "wrong", "doesn't work"]
        
        has_forethought = any(ind in query_lower for ind in forethought_indicators)
        has_performance = any(ind in query_lower for ind in performance_indicators)
        
        return has_forethought and has_performance
    
    def _check_cache(self, context: ClassificationContext) -> Optional[EnhancedClassificationResult]:
        """Check if we have a cached result for similar query."""
        if not self._classification_cache:
            return None
        
        # Simple cache key based on query and major context elements
        cache_key = self._generate_cache_key(context)
        return self._classification_cache.get(cache_key)
    
    def _cache_result(self, context: ClassificationContext, result: EnhancedClassificationResult):
        """Cache the classification result."""
        if not self._classification_cache:
            return
        
        cache_key = self._generate_cache_key(context)
        self._classification_cache[cache_key] = result
        
        # Simple cache size management
        if len(self._classification_cache) > 1000:
            # Remove oldest 20% of entries (simple FIFO)
            keys_to_remove = list(self._classification_cache.keys())[:200]
            for key in keys_to_remove:
                del self._classification_cache[key]
    
    def _generate_cache_key(self, context: ClassificationContext) -> str:
        """Generate a cache key for the classification context."""
        key_parts = [
            context.query[:100],  # First 100 chars of query
            "code" if context.code_snippet else "nocode",
            "error" if context.error_message else "noerror",
            context.student_level or "unknown"
        ]
        return "|".join(key_parts)
    
    def _create_fallback_result(self, context: ClassificationContext, error: str) -> EnhancedClassificationResult:
        """Create a fallback result when all classification attempts fail."""
        return EnhancedClassificationResult(
            classification=SRLPhase.FORETHOUGHT,  # Default to implementation support
            confidence=0.1,
            reasoning=f"System fallback due to classification failure: {error}",
            indicators=["system_fallback"],
            classification_strategy="emergency_fallback",
            fallback_used=True,
            query_complexity="unknown",
            student_intent="unknown",
            educational_priority="medium"
        )
    
    def _log_classification(self, 
                          context: ClassificationContext,
                          result: EnhancedClassificationResult,
                          log_context: LogContext):
        """Log the classification result for analysis and monitoring."""
        self.logger.log_query_processing(
            query=context.query,
            classification=result.classification.value,
            confidence=result.confidence,
            context=log_context
        )
        
        # Log detailed educational metadata
        self.logger.log_event(
            EventType.SRL_CLASSIFICATION,
            f"SRL phase classified as {result.classification.value}",
            context=log_context,
            extra_data={
                "classification": result.classification.value,
                "confidence": result.confidence,
                "strategy": result.classification_strategy,
                "complexity": result.query_complexity,
                "domain": result.domain_detected,
                "intent": result.student_intent,
                "priority": result.educational_priority,
                "fallback_used": result.fallback_used,
                "indicators": result.indicators,
                "processing_time_ms": result.processing_time_ms
            }
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the classifier."""
        avg_processing_time = (
            self.total_processing_time / self.classification_count 
            if self.classification_count > 0 else 0
        )
        
        cache_stats = {}
        if self._classification_cache:
            cache_stats = {
                "cache_size": len(self._classification_cache),
                "cache_enabled": True
            }
        else:
            cache_stats = {"cache_enabled": False}
        
        return {
            "classification_count": self.classification_count,
            "average_processing_time_ms": avg_processing_time,
            "total_processing_time_ms": self.total_processing_time,
            "confidence_threshold": self.confidence_threshold,
            **cache_stats
        }


# Global classifier instance
_srl_classifier: Optional[SRLClassifier] = None


def get_srl_classifier(reload: bool = False) -> SRLClassifier:
    """
    Get global SRL classifier instance (singleton pattern).
    
    Args:
        reload: Force creation of new classifier instance
        
    Returns:
        SRLClassifier instance
    """
    global _srl_classifier
    if _srl_classifier is None or reload:
        _srl_classifier = SRLClassifier()
    return _srl_classifier


if __name__ == "__main__":
    # SRL classifier test
    try:
        classifier = get_srl_classifier()
        
        # Test cases representing different SRL phases
        test_cases = [
            ClassificationContext(
                query="How do I implement a binary search algorithm in Python?",
                student_level="intermediate"
            ),
            ClassificationContext(
                query="My binary search function returns the wrong index",
                code_snippet="def binary_search(arr, target):\n    # buggy implementation",
                error_message=None
            ),
            ClassificationContext(
                query="I'm getting an IndexError in my list manipulation code",
                error_message="IndexError: list index out of range"
            )
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\nTest Case {i+1}:")
            print(f"Query: {test_case.query}")
            
            result = classifier.classify_query(test_case)
            print(f"Classification: {result.classification.value}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Strategy: {result.classification_strategy}")
            print(f"Reasoning: {result.reasoning}")
        
        # Performance stats
        print(f"\nPerformance Stats: {classifier.get_performance_stats()}")
        print("✅ SRL classifier test completed successfully!")
        
    except Exception as e:
        print(f"❌ SRL classifier test failed: {e}")
