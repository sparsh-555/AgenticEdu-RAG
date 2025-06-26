"""
Base Agent Architecture for Agentic Edu-RAG System

This module defines the foundational architecture for all specialized agents in our
multi-agent educational system. It implements the Template Method design pattern
to ensure consistent behavior while allowing specialized customization.

Key Architectural Principles:
1. Template Method Pattern: Defines the skeleton of agent operations while allowing
   subclasses to override specific steps for specialized behavior
2. Dependency Injection: Agents receive their dependencies (LLM client, RAG system)
   rather than creating them, enabling better testing and flexibility
3. Observable Behavior: All agent actions are logged and monitored for research analysis
4. Error Resilience: Comprehensive error handling with graceful degradation
5. Educational Focus: Built-in support for SRL-aligned educational interactions

Design Patterns Demonstrated:
- Template Method: Base processing flow with customizable steps
- Strategy Pattern: Different response generation strategies for different agent types
- Observer Pattern: Event-driven logging and monitoring
- Dependency Injection: Loose coupling through constructor injection

This architecture ensures that all agents follow consistent patterns while enabling
the specialized behavior required for effective Self-Regulated Learning support.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import json

from pydantic import BaseModel, Field

from ..config.settings import get_settings, load_prompt_templates
from ..utils.logging_utils import get_logger, LogContext, EventType, create_context
from ..utils.api_utils import get_openai_client, APIResponse, OpenAIClient


class AgentType(Enum):
    """Enumeration of available agent types for type safety and routing."""
    ORCHESTRATOR = "orchestrator"
    IMPLEMENTATION = "implementation"
    DEBUGGING = "debugging"


class ResponseType(Enum):
    """Types of responses agents can generate for different educational contexts."""
    GUIDANCE = "guidance"          # Strategic guidance and planning support
    EXPLANATION = "explanation"    # Concept explanation and teaching
    DEBUGGING_HELP = "debugging_help"  # Step-by-step debugging assistance
    CODE_REVIEW = "code_review"    # Code analysis and improvement suggestions
    ENCOURAGEMENT = "encouragement"  # Motivational and confidence-building


@dataclass
class AgentInput:
    """
    Standardized input structure for all agent interactions.
    
    This ensures consistent interfaces across all agent types while providing
    the context necessary for educational interactions.
    """
    query: str
    code_snippet: Optional[str] = None
    error_message: Optional[str] = None
    context: Optional[LogContext] = None
    previous_interactions: Optional[List[Dict[str, Any]]] = None
    srl_phase: Optional[str] = None  # "forethought" or "performance"
    student_level: Optional[str] = None  # "beginner", "intermediate", "advanced"
    
    def __post_init__(self):
        """Ensure context is available for logging and correlation."""
        if self.context is None:
            self.context = create_context()


class AgentResponse(BaseModel):
    """
    Standardized response structure for all agent interactions.
    
    This provides consistent output formatting and metadata collection
    for research analysis and system optimization.
    """
    content: str = Field(..., description="Main response content for the student")
    response_type: ResponseType = Field(..., description="Type of response provided")
    agent_type: AgentType = Field(..., description="Type of agent that generated response")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Agent's confidence in response")
    educational_metadata: Dict[str, Any] = Field(default_factory=dict, description="Educational context and metrics")
    rag_context: Optional[List[str]] = Field(default=None, description="Retrieved context used in response")
    processing_time_ms: float = Field(..., description="Time taken to generate response")
    token_usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage for cost tracking")
    suggested_follow_up: Optional[List[str]] = Field(default=None, description="Suggested next questions or actions")
    
    class Config:
        use_enum_values = True


class BaseAgent(ABC):
    """
    Abstract base class for all educational agents in the system.
    
    This class implements the Template Method pattern, defining the standard
    workflow for agent processing while allowing subclasses to customize
    specific steps for their specialized educational functions.
    
    The standard workflow consists of:
    1. Input validation and preprocessing
    2. Context retrieval from RAG system
    3. Response generation using specialized prompts
    4. Post-processing and educational metadata enrichment
    5. Logging and performance monitoring
    
    Subclasses must implement:
    - get_agent_type(): Return the specific agent type
    - get_specialized_prompts(): Return agent-specific prompt templates
    - process_specialized_response(): Customize response generation logic
    - validate_specialized_input(): Validate agent-specific input requirements
    """
    
    def __init__(self, 
                 openai_client: Optional[OpenAI] = None,
                 rag_system = None,  # Will be properly typed when RAG system is implemented
                 enable_rag: bool = True):
        """
        Initialize the base agent with required dependencies.
        
        Args:
            openai_client: OpenAI client for LLM interactions
            rag_system: RAG system for knowledge retrieval
            enable_rag: Whether to use RAG for context enhancement
        """
        self.settings = get_settings()
        self.logger = get_logger()
        self.openai_client = openai_client or get_openai_client()
        self.rag_system = rag_system
        self.enable_rag = enable_rag
        
        # Load prompt templates for this agent type
        self.prompt_templates = load_prompt_templates()
        
        # Performance tracking
        self.interaction_count = 0
        self.total_processing_time = 0.0
        
        self.logger.log_event(
            EventType.SYSTEM_START,
            f"{self.get_agent_type().value} agent initialized",
            extra_data={
                "agent_type": self.get_agent_type().value,
                "rag_enabled": enable_rag
            }
        )
    
    @abstractmethod
    def get_agent_type(self) -> AgentType:
        """Return the specific type of this agent for routing and logging."""
        pass
    
    @abstractmethod
    def get_specialized_prompts(self) -> Dict[str, str]:
        """
        Return agent-specific prompt templates.
        
        Returns:
            Dictionary of prompt templates for this agent type
        """
        pass
    
    @abstractmethod
    def validate_specialized_input(self, agent_input: AgentInput) -> Tuple[bool, Optional[str]]:
        """
        Validate input specific to this agent type.
        
        Args:
            agent_input: Input to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def process_specialized_response(self, 
                                   agent_input: AgentInput,
                                   rag_context: Optional[List[str]],
                                   base_response: str) -> AgentResponse:
        """
        Process and customize the response for this agent type.
        
        Args:
            agent_input: Original input from student
            rag_context: Retrieved context from RAG system
            base_response: Base response from LLM
            
        Returns:
            Specialized AgentResponse with educational metadata
        """
        pass
    
    def process_query(self, agent_input: AgentInput) -> AgentResponse:
        """
        Main processing method implementing the Template Method pattern.
        
        This method defines the standard workflow that all agents follow,
        while delegating specialized behavior to abstract methods.
        
        Workflow:
        1. Validate input
        2. Retrieve relevant context (if RAG enabled)
        3. Generate response using LLM
        4. Process and enrich response
        5. Log interaction for research analysis
        
        Args:
            agent_input: Standardized input containing query and context
            
        Returns:
            AgentResponse with content and educational metadata
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If response generation fails
        """
        start_time = time.time()
        
        try:
            # Step 1: Validate input
            self._validate_input(agent_input)
            
            # Step 2: Retrieve relevant context (Template Method delegation)
            rag_context = self._retrieve_context(agent_input) if self.enable_rag else None
            
            # Step 3: Generate base response
            base_response = self._generate_base_response(agent_input, rag_context)
            
            # Step 4: Process specialized response (Template Method delegation)
            response = self.process_specialized_response(agent_input, rag_context, base_response)
            
            # Step 5: Add processing metadata
            processing_time = (time.time() - start_time) * 1000
            response.processing_time_ms = processing_time
            
            # Step 6: Log interaction
            self._log_interaction(agent_input, response)
            
            # Update performance tracking
            self.interaction_count += 1
            self.total_processing_time += processing_time
            
            return response
            
        except Exception as e:
            self.logger.log_event(
                EventType.AGENT_ERROR,
                f"Agent processing failed: {str(e)}",
                context=agent_input.context,
                level="ERROR",
                extra_data={
                    "agent_type": self.get_agent_type().value,
                    "query_preview": agent_input.query[:100]
                }
            )
            raise RuntimeError(f"Agent processing failed: {str(e)}")
    
    def _validate_input(self, agent_input: AgentInput):
        """
        Validate input using both base and specialized validation.
        
        Args:
            agent_input: Input to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Base validation
        if not agent_input.query or not agent_input.query.strip():
            raise ValueError("Query cannot be empty")
        
        if len(agent_input.query) > 10000:  # Reasonable limit
            raise ValueError("Query too long (max 10,000 characters)")
        
        # Specialized validation (Template Method delegation)
        is_valid, error_message = self.validate_specialized_input(agent_input)
        if not is_valid:
            raise ValueError(f"Specialized validation failed: {error_message}")
    
    def _retrieve_context(self, agent_input: AgentInput) -> Optional[List[str]]:
        """
        Retrieve relevant context from RAG system if available.
        
        Args:
            agent_input: Input containing query for context retrieval
            
        Returns:
            List of relevant context strings or None if RAG disabled
        """
        if not self.rag_system:
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                "RAG system not available, proceeding without context",
                context=agent_input.context,
                level="DEBUG"
            )
            return None
        
        try:
            # This will be implemented when RAG system is available
            # For now, return placeholder
            context = []
            
            self.logger.log_rag_operation(
                operation="context_retrieval",
                query=agent_input.query,
                results_count=len(context),
                context=agent_input.context
            )
            
            return context if context else None
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Context retrieval failed: {str(e)}",
                context=agent_input.context,
                level="WARNING"
            )
            return None
    
    def _generate_base_response(self, 
                               agent_input: AgentInput,
                               rag_context: Optional[List[str]]) -> str:
        """
        Generate base response using LLM with agent-specific prompts.
        
        Args:
            agent_input: Input containing query and context
            rag_context: Retrieved context from RAG system
            
        Returns:
            Base response string from LLM
        """
        # Get specialized prompts for this agent type
        prompts = self.get_specialized_prompts()
        system_prompt = prompts.get("system", "You are a helpful educational assistant.")
        
        # Construct messages for chat completion
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add RAG context if available
        if rag_context:
            context_text = "\n\n".join(rag_context)
            context_message = f"Relevant educational context:\n{context_text}"
            messages.append({"role": "system", "content": context_message})
        
        # Add previous interactions if available
        if agent_input.previous_interactions:
            for interaction in agent_input.previous_interactions[-3:]:  # Last 3 interactions
                if "user" in interaction:
                    messages.append({"role": "user", "content": interaction["user"]})
                if "assistant" in interaction:
                    messages.append({"role": "assistant", "content": interaction["assistant"]})
        
        # Construct user message with all available context
        user_content_parts = [f"Student query: {agent_input.query}"]
        
        if agent_input.code_snippet:
            user_content_parts.append(f"Code snippet:\n```\n{agent_input.code_snippet}\n```")
        
        if agent_input.error_message:
            user_content_parts.append(f"Error message: {agent_input.error_message}")
        
        if agent_input.srl_phase:
            user_content_parts.append(f"Learning phase: {agent_input.srl_phase}")
        
        if agent_input.student_level:
            user_content_parts.append(f"Student level: {agent_input.student_level}")
        
        user_message = "\n\n".join(user_content_parts)
        messages.append({"role": "user", "content": user_message})
        
        # Generate response using OpenAI client
        api_response = self.openai_client.create_chat_completion(
            messages=messages,
            context=agent_input.context
        )
        
        return api_response.content
    
    def _log_interaction(self, agent_input: AgentInput, response: AgentResponse):
        """
        Log the agent interaction for research analysis and monitoring.
        
        Args:
            agent_input: Input that was processed
            response: Response that was generated
        """
        self.logger.log_agent_interaction(
            agent_type=self.get_agent_type().value,
            action="response_generated",
            context=agent_input.context,
            performance_data=None  # Could add PerformanceMetrics here
        )
        
        # Log educational metadata for SRL research
        self.logger.log_event(
            EventType.LEARNING_INTERACTION,
            f"Educational response generated",
            context=agent_input.context,
            extra_data={
                "agent_type": self.get_agent_type().value,
                "response_type": response.response_type.value,
                "confidence": response.confidence,
                "srl_phase": agent_input.srl_phase,
                "student_level": agent_input.student_level,
                "query_length": len(agent_input.query),
                "response_length": len(response.content),
                "rag_context_used": response.rag_context is not None,
                "processing_time_ms": response.processing_time_ms
            }
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for this agent instance.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_processing_time = (
            self.total_processing_time / self.interaction_count 
            if self.interaction_count > 0 else 0
        )
        
        return {
            "agent_type": self.get_agent_type().value,
            "interaction_count": self.interaction_count,
            "total_processing_time_ms": self.total_processing_time,
            "average_processing_time_ms": avg_processing_time,
            "rag_enabled": self.enable_rag
        }
    
    def create_educational_metadata(self, 
                                   agent_input: AgentInput,
                                   confidence: float,
                                   concepts_covered: Optional[List[str]] = None,
                                   learning_objectives: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create educational metadata for research and analysis purposes.
        
        Args:
            agent_input: Original input from student
            confidence: Agent's confidence in the response
            concepts_covered: Programming concepts addressed in response
            learning_objectives: Educational objectives supported
            
        Returns:
            Dictionary with educational metadata
        """
        return {
            "srl_phase": agent_input.srl_phase,
            "student_level": agent_input.student_level,
            "agent_confidence": confidence,
            "concepts_covered": concepts_covered or [],
            "learning_objectives": learning_objectives or [],
            "interaction_complexity": self._assess_interaction_complexity(agent_input),
            "code_analysis": {
                "has_code": agent_input.code_snippet is not None,
                "has_error": agent_input.error_message is not None,
                "code_length": len(agent_input.code_snippet) if agent_input.code_snippet else 0
            },
            "query_analysis": {
                "query_length": len(agent_input.query),
                "question_words": self._count_question_words(agent_input.query),
                "technical_terms": self._count_technical_terms(agent_input.query)
            }
        }
    
    def _assess_interaction_complexity(self, agent_input: AgentInput) -> str:
        """Assess the complexity level of the interaction for educational analysis."""
        complexity_score = 0
        
        # Query length and complexity
        if len(agent_input.query) > 200:
            complexity_score += 1
        
        # Presence of code
        if agent_input.code_snippet:
            complexity_score += 2
            if len(agent_input.code_snippet) > 500:
                complexity_score += 1
        
        # Error messages indicate debugging complexity
        if agent_input.error_message:
            complexity_score += 2
        
        # Multiple previous interactions indicate ongoing conversation
        if agent_input.previous_interactions and len(agent_input.previous_interactions) > 2:
            complexity_score += 1
        
        if complexity_score >= 5:
            return "high"
        elif complexity_score >= 3:
            return "medium"
        else:
            return "low"
    
    def _count_question_words(self, text: str) -> int:
        """Count question words in text for interaction analysis."""
        question_words = ["how", "what", "why", "when", "where", "which", "who"]
        text_lower = text.lower()
        return sum(1 for word in question_words if word in text_lower)
    
    def _count_technical_terms(self, text: str) -> int:
        """Count programming-related technical terms for complexity assessment."""
        technical_terms = [
            "function", "class", "variable", "loop", "if", "else", "return", "import",
            "array", "list", "dictionary", "string", "integer", "boolean", "algorithm",
            "recursion", "iteration", "object", "method", "parameter", "argument"
        ]
        text_lower = text.lower()
        return sum(1 for term in technical_terms if term in text_lower)


if __name__ == "__main__":
    # Base agent architecture test
    print("BaseAgent architecture loaded successfully!")
    print("Available agent types:", [t.value for t in AgentType])
    print("Available response types:", [t.value for t in ResponseType])
    print("✅ Base agent architecture test completed!")
