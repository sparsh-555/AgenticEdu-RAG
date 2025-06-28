"""
Orchestrator Agent - Central Coordinator for Multi-Agent Educational System

This module implements the central orchestrator that coordinates the entire multi-agent
workflow using LangGraph. It serves as the primary entry point for student queries
and manages the routing between specialized agents based on SRL phase classification.

Key Responsibilities:
1. Query Reception and Initial Processing
2. SRL Phase Classification and Routing Decisions
3. Agent Coordination and Workflow Management
4. Response Aggregation and Quality Assurance
5. Educational Metadata Collection and Analysis
6. Error Handling and Graceful Degradation

LangGraph Integration:
This orchestrator uses LangGraph's StateGraph to define a sophisticated workflow
that adapts to different query types and educational contexts. The workflow
implements the following pattern:

Query → Classification → Agent Routing → Response Generation → Quality Check → Output

Architectural Patterns Demonstrated:
- Orchestrator Pattern: Central coordination of distributed agents
- State Machine: LangGraph-based workflow management
- Strategy Pattern: Dynamic agent selection based on classification
- Observer Pattern: Comprehensive logging and monitoring
- Template Method: Consistent processing workflow with customizable steps

This implementation showcases how to build production-ready multi-agent systems
that can scale and adapt to complex educational requirements.
"""

from typing import Dict, List, Optional, Any, Union, Literal
from dataclasses import dataclass
import time
import uuid
from enum import Enum

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated

from .base_agent import BaseAgent, AgentType, AgentInput, AgentResponse, ResponseType
from .implementation_agent import ImplementationAgent
from .debugging_agent import DebuggingAgent
from classification.srl_classifier import (
    get_srl_classifier, SRLClassifier, ClassificationContext, SRLPhase
)
from utils.logging_utils import get_logger, LogContext, EventType, create_context
from utils.api_utils import get_openai_client
from config.settings import get_settings


class WorkflowState(TypedDict):
    """
    State schema for the LangGraph workflow.
    
    This state is passed between nodes in the workflow and maintains
    all context needed for educational query processing.
    """
    # Input data
    query: str
    code_snippet: Optional[str]
    error_message: Optional[str]
    session_id: str
    user_id: Optional[str]
    student_level: Optional[str]
    conversation_history: Optional[List[Dict[str, Any]]]
    
    # Processing state
    classification_result: Optional[Dict[str, Any]]
    selected_agent: Optional[str]
    agent_response: Optional[Dict[str, Any]]
    
    # Metadata and tracking
    workflow_stage: str
    processing_start_time: float
    log_context: Optional[Dict[str, Any]]
    error_occurred: bool
    error_message_internal: Optional[str]
    
    # Educational context
    srl_phase: Optional[str]
    educational_metadata: Optional[Dict[str, Any]]
    
    # Final output
    final_response: Optional[Dict[str, Any]]
    performance_metrics: Optional[Dict[str, Any]]


class WorkflowStage(Enum):
    """Stages in the orchestrator workflow for tracking and monitoring."""
    INITIALIZATION = "initialization"
    CLASSIFICATION = "classification"
    AGENT_SELECTION = "agent_selection"
    AGENT_PROCESSING = "agent_processing"
    QUALITY_ASSURANCE = "quality_assurance"
    RESPONSE_FINALIZATION = "response_finalization"
    COMPLETED = "completed"
    ERROR_HANDLING = "error_handling"


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator behavior."""
    enable_quality_checks: bool = True
    max_processing_time_ms: float = 30000  # 30 seconds
    require_minimum_confidence: float = 0.5
    enable_fallback_routing: bool = True
    collect_educational_metrics: bool = True
    enable_conversation_memory: bool = True


class OrchestratorResponse(BaseModel):
    """Comprehensive response from the orchestrator with full metadata."""
    content: str = Field(..., description="Final response content for the student")
    agent_used: str = Field(..., description="Agent that processed the query")
    srl_phase: str = Field(..., description="Detected SRL phase")
    classification_confidence: float = Field(..., description="Classification confidence score")
    
    # Educational metadata
    educational_metadata: Dict[str, Any] = Field(default_factory=dict)
    concepts_covered: List[str] = Field(default_factory=list)
    learning_objectives: List[str] = Field(default_factory=list)
    suggested_follow_up: List[str] = Field(default_factory=list)
    
    # Performance metrics
    total_processing_time_ms: float = Field(..., description="Total processing time")
    classification_time_ms: float = Field(default=0.0)
    agent_processing_time_ms: float = Field(default=0.0)
    
    # System metadata
    session_id: str = Field(..., description="Session identifier")
    workflow_id: str = Field(..., description="Unique workflow execution ID")
    quality_checks_passed: bool = Field(default=True)
    fallback_used: bool = Field(default=False)
    
    # Conversation context
    conversation_turn: int = Field(default=1)
    maintains_context: bool = Field(default=False)


class OrchestratorAgent:
    """
    Central orchestrator for the multi-agent educational system.
    
    This agent coordinates the entire workflow from query reception to final response,
    using LangGraph to manage a sophisticated state machine that adapts to different
    educational contexts and learning phases.
    
    The orchestrator implements several advanced patterns:
    - Dynamic agent routing based on SRL classification
    - Quality assurance and confidence validation
    - Educational metadata collection and analysis
    - Graceful error handling and fallback mechanisms
    - Performance monitoring and optimization
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Initialize the orchestrator with configuration and dependencies.
        
        Args:
            config: Configuration for orchestrator behavior
        """
        self.config = config or OrchestratorConfig()
        self.settings = get_settings()
        self.logger = get_logger()
        self.openai_client = get_openai_client()
        self.srl_classifier = get_srl_classifier()
        
        # Initialize specialized agents
        self.implementation_agent = ImplementationAgent()
        self.debugging_agent = DebuggingAgent()
        
        # Performance tracking
        self.total_queries_processed = 0
        self.average_processing_time = 0.0
        self.classification_accuracy_samples = []
        
        # Conversation memory (simple in-memory storage)
        self.conversation_memory = {} if self.config.enable_conversation_memory else None
        
        # Build the LangGraph workflow
        self.workflow = self._build_langgraph_workflow()
        
        self.logger.log_event(
            EventType.SYSTEM_START,
            "Orchestrator agent initialized",
            extra_data={
                "config": {
                    "quality_checks": self.config.enable_quality_checks,
                    "max_processing_time": self.config.max_processing_time_ms,
                    "conversation_memory": self.config.enable_conversation_memory
                }
            }
        )
    
    def _build_langgraph_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow for multi-agent orchestration.
        
        This creates a sophisticated state machine that handles the complete
        educational query processing pipeline with proper error handling,
        quality assurance, and performance monitoring.
        
        Returns:
            Compiled LangGraph workflow
        """
        # Create the state graph
        workflow = StateGraph(WorkflowState)
        
        # Add workflow nodes
        workflow.add_node("initialize", self._initialize_processing)
        workflow.add_node("classify_query", self._classify_query_node)
        workflow.add_node("route_to_agent", self._route_to_agent_node)
        workflow.add_node("process_with_implementation", self._process_with_implementation_node)
        workflow.add_node("process_with_debugging", self._process_with_debugging_node)
        workflow.add_node("quality_assurance", self._quality_assurance_node)
        workflow.add_node("finalize_response", self._finalize_response_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Define the workflow edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "classify_query")
        workflow.add_conditional_edges(
            "classify_query",
            self._classification_router,
            {
                "implementation": "route_to_agent",
                "debugging": "route_to_agent", 
                "error": "handle_error"
            }
        )
        workflow.add_conditional_edges(
            "route_to_agent",
            self._agent_router,
            {
                "implementation": "process_with_implementation",
                "debugging": "process_with_debugging",
                "error": "handle_error"
            }
        )
        workflow.add_edge("process_with_implementation", "quality_assurance")
        workflow.add_edge("process_with_debugging", "quality_assurance")
        workflow.add_conditional_edges(
            "quality_assurance",
            self._quality_router,
            {
                "passed": "finalize_response",
                "failed": "handle_error",
                "retry": "route_to_agent"
            }
        )
        workflow.add_edge("finalize_response", END)
        workflow.add_edge("handle_error", END)
        
        # Compile the workflow
        return workflow.compile()
    
    def process_query(self, 
                     query: str,
                     code_snippet: Optional[str] = None,
                     error_message: Optional[str] = None,
                     session_id: Optional[str] = None,
                     user_id: Optional[str] = None,
                     student_level: Optional[str] = None) -> OrchestratorResponse:
        """
        Process a student query through the complete multi-agent workflow.
        
        This is the main entry point for the educational system, handling the
        entire pipeline from classification to specialized agent processing.
        
        Args:
            query: Student's programming question
            code_snippet: Optional code context
            error_message: Optional error message
            session_id: Session identifier for conversation tracking
            user_id: User identifier (optional)
            student_level: Student proficiency level
            
        Returns:
            Comprehensive orchestrator response with educational metadata
            
        Raises:
            RuntimeError: If workflow execution fails completely
        """
        start_time = time.time()
        session_id = session_id or str(uuid.uuid4())
        workflow_id = str(uuid.uuid4())
        log_context = create_context(session_id=session_id, user_id=user_id)
        
        try:
            # Load conversation history if available
            conversation_history = self._load_conversation_history(session_id)
            
            # Initialize workflow state
            initial_state: WorkflowState = {
                "query": query,
                "code_snippet": code_snippet,
                "error_message": error_message,
                "session_id": session_id,
                "user_id": user_id,
                "student_level": student_level,
                "conversation_history": conversation_history,
                "classification_result": None,
                "selected_agent": None,
                "agent_response": None,
                "workflow_stage": WorkflowStage.INITIALIZATION.value,
                "processing_start_time": start_time,
                "log_context": log_context.__dict__,
                "error_occurred": False,
                "error_message_internal": None,
                "srl_phase": None,
                "educational_metadata": None,
                "final_response": None,
                "performance_metrics": None
            }
            
            # Execute the workflow
            self.logger.log_event(
                EventType.QUERY_RECEIVED,
                f"Processing query through orchestrator workflow",
                context=log_context,
                extra_data={
                    "workflow_id": workflow_id,
                    "query_length": len(query),
                    "has_code": code_snippet is not None,
                    "has_error": error_message is not None
                }
            )
            
            # Run the LangGraph workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Build the response from final state
            response = self._build_orchestrator_response(final_state, workflow_id, start_time)
            
            # Update conversation memory
            self._update_conversation_memory(session_id, query, response)
            
            # Update performance tracking
            self._update_performance_metrics(response)
            
            # Log completion
            self.logger.log_event(
                EventType.QUERY_COMPLETED,
                f"Query processing completed successfully",
                context=log_context,
                extra_data={
                    "workflow_id": workflow_id,
                    "total_time_ms": response.total_processing_time_ms,
                    "agent_used": response.agent_used,
                    "srl_phase": response.srl_phase,
                    "quality_passed": response.quality_checks_passed
                }
            )
            
            return response
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Orchestrator workflow failed: {str(e)}",
                context=log_context,
                level="ERROR",
                extra_data={"workflow_id": workflow_id}
            )
            
            # Return error response
            return self._create_error_response(query, session_id, workflow_id, start_time, str(e))
    
    # LangGraph Node Implementations
    
    def _initialize_processing(self, state: WorkflowState) -> WorkflowState:
        """Initialize the processing workflow with validation and setup."""
        state["workflow_stage"] = WorkflowStage.INITIALIZATION.value
        
        # Validate input
        if not state["query"] or not state["query"].strip():
            state["error_occurred"] = True
            state["error_message_internal"] = "Empty query provided"
            return state
        
        # Check processing time limits
        if time.time() - state["processing_start_time"] > self.config.max_processing_time_ms / 1000:
            state["error_occurred"] = True
            state["error_message_internal"] = "Processing timeout during initialization"
            return state
        
        self.logger.log_event(
            EventType.SYSTEM_START,
            "Workflow initialization completed",
            extra_data={"query_length": len(state["query"])}
        )
        
        return state
    
    def _classify_query_node(self, state: WorkflowState) -> WorkflowState:
        """Classify the query into SRL phases using the classification system."""
        state["workflow_stage"] = WorkflowStage.CLASSIFICATION.value
        classification_start = time.time()
        
        try:
            # Create classification context
            classification_context = ClassificationContext(
                query=state["query"],
                code_snippet=state["code_snippet"],
                error_message=state["error_message"],
                conversation_history=state["conversation_history"],
                student_level=state["student_level"]
            )
            
            # Perform classification
            classification_result = self.srl_classifier.classify_query(classification_context)
            
            # Store results in state
            state["classification_result"] = {
                "classification": classification_result.classification.value,
                "confidence": classification_result.confidence,
                "reasoning": classification_result.reasoning,
                "indicators": classification_result.indicators,
                "strategy": classification_result.classification_strategy,
                "processing_time_ms": (time.time() - classification_start) * 1000
            }
            state["srl_phase"] = classification_result.classification.value
            
            # Check if confidence meets requirements
            if classification_result.confidence < self.config.require_minimum_confidence:
                if not self.config.enable_fallback_routing:
                    state["error_occurred"] = True
                    state["error_message_internal"] = f"Classification confidence too low: {classification_result.confidence}"
                    return state
            
            self.logger.log_event(
                EventType.SRL_CLASSIFICATION,
                f"Query classified as {classification_result.classification.value}",
                extra_data={
                    "confidence": classification_result.confidence,
                    "strategy": classification_result.classification_strategy
                }
            )
            
        except Exception as e:
            state["error_occurred"] = True
            state["error_message_internal"] = f"Classification failed: {str(e)}"
        
        return state
    
    def _route_to_agent_node(self, state: WorkflowState) -> WorkflowState:
        """Route the query to the appropriate specialized agent."""
        state["workflow_stage"] = WorkflowStage.AGENT_SELECTION.value
        
        if not state["classification_result"]:
            state["error_occurred"] = True
            state["error_message_internal"] = "No classification result available for routing"
            return state
        
        classification = state["classification_result"]["classification"]
        
        if classification == SRLPhase.FORETHOUGHT.value:
            state["selected_agent"] = "implementation"
        elif classification == SRLPhase.PERFORMANCE.value:
            state["selected_agent"] = "debugging"
        else:
            # Fallback to implementation agent for unknown classifications
            state["selected_agent"] = "implementation"
            self.logger.log_event(
                EventType.AGENT_INVOKED,
                f"Unknown classification, defaulting to implementation agent",
                level="WARNING"
            )
        
        self.logger.log_event(
            EventType.QUERY_ROUTED,
            f"Query routed to {state['selected_agent']} agent",
            extra_data={"classification": classification}
        )
        
        return state
    
    def _process_with_implementation_node(self, state: WorkflowState) -> WorkflowState:
        """Process the query using the Implementation Agent."""
        return self._process_with_agent(state, self.implementation_agent, "implementation")
    
    def _process_with_debugging_node(self, state: WorkflowState) -> WorkflowState:
        """Process the query using the Debugging Agent."""
        return self._process_with_agent(state, self.debugging_agent, "debugging")
    
    def _process_with_agent(self, 
                           state: WorkflowState, 
                           agent: BaseAgent, 
                           agent_name: str) -> WorkflowState:
        """Generic agent processing method."""
        state["workflow_stage"] = WorkflowStage.AGENT_PROCESSING.value
        agent_start = time.time()
        
        try:
            # Create agent input
            agent_input = AgentInput(
                query=state["query"],
                code_snippet=state["code_snippet"],
                error_message=state["error_message"],
                previous_interactions=state["conversation_history"],
                srl_phase=state["srl_phase"],
                student_level=state["student_level"]
            )
            
            # Process with the agent
            agent_response = agent.process_query(agent_input)
            
            # Store agent response in state
            state["agent_response"] = {
                "content": agent_response.content,
                "response_type": agent_response.response_type.value,
                "agent_type": agent_response.agent_type.value,
                "confidence": agent_response.confidence,
                "educational_metadata": agent_response.educational_metadata,
                "suggested_follow_up": agent_response.suggested_follow_up,
                "processing_time_ms": (time.time() - agent_start) * 1000
            }
            
            self.logger.log_agent_interaction(
                agent_type=agent_name,
                action="response_generated",
                context=None
            )
            
        except Exception as e:
            state["error_occurred"] = True
            state["error_message_internal"] = f"Agent processing failed: {str(e)}"
            
            self.logger.log_event(
                EventType.AGENT_ERROR,
                f"{agent_name} agent processing failed: {str(e)}",
                level="ERROR"
            )
        
        return state
    
    def _quality_assurance_node(self, state: WorkflowState) -> WorkflowState:
        """Perform quality assurance checks on the agent response."""
        state["workflow_stage"] = WorkflowStage.QUALITY_ASSURANCE.value
        
        if not self.config.enable_quality_checks:
            return state
        
        if not state["agent_response"]:
            state["error_occurred"] = True
            state["error_message_internal"] = "No agent response available for quality check"
            return state
        
        try:
            quality_checks = self._perform_quality_checks(state)
            state["quality_checks"] = quality_checks
            
            # If quality checks fail and we haven't retried yet
            if not quality_checks["overall_passed"] and not state.get("quality_retry_attempted"):
                state["quality_retry_attempted"] = True
                # Could implement retry logic here
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Quality assurance failed: {str(e)}",
                level="WARNING"
            )
            # Continue processing even if QA fails
            state["quality_checks"] = {"overall_passed": True, "error": str(e)}
        
        return state
    
    def _finalize_response_node(self, state: WorkflowState) -> WorkflowState:
        """Finalize the response with metadata and performance metrics."""
        state["workflow_stage"] = WorkflowStage.RESPONSE_FINALIZATION.value
        
        try:
            # Collect performance metrics
            total_time = (time.time() - state["processing_start_time"]) * 1000
            
            performance_metrics = {
                "total_processing_time_ms": total_time,
                "classification_time_ms": state.get("classification_result", {}).get("processing_time_ms", 0),
                "agent_processing_time_ms": state.get("agent_response", {}).get("processing_time_ms", 0),
                "quality_check_passed": state.get("quality_checks", {}).get("overall_passed", True)
            }
            
            state["performance_metrics"] = performance_metrics
            state["workflow_stage"] = WorkflowStage.COMPLETED.value
            
        except Exception as e:
            state["error_occurred"] = True
            state["error_message_internal"] = f"Response finalization failed: {str(e)}"
        
        return state
    
    def _handle_error_node(self, state: WorkflowState) -> WorkflowState:
        """Handle errors with appropriate fallback responses."""
        state["workflow_stage"] = WorkflowStage.ERROR_HANDLING.value
        
        error_message = state.get("error_message_internal", "Unknown error occurred")
        
        # Create fallback response
        fallback_response = {
            "content": self._generate_fallback_response(state["query"], error_message),
            "response_type": ResponseType.GUIDANCE.value,
            "agent_type": "orchestrator_fallback",
            "confidence": 0.3,
            "educational_metadata": {"fallback_used": True, "error": error_message},
            "suggested_follow_up": [
                "Could you rephrase your question with more specific details?",
                "If you have code, please share it for better assistance.",
                "What specific part of the problem are you struggling with?"
            ],
            "processing_time_ms": 0
        }
        
        state["agent_response"] = fallback_response
        state["selected_agent"] = "fallback"
        
        return state
    
    # Router Functions for LangGraph Conditional Edges
    
    def _classification_router(self, state: WorkflowState) -> Literal["implementation", "debugging", "error"]:
        """Route based on classification results."""
        if state["error_occurred"]:
            return "error"
        
        classification = state.get("classification_result", {}).get("classification")
        if classification == SRLPhase.FORETHOUGHT.value:
            return "implementation"
        elif classification == SRLPhase.PERFORMANCE.value:
            return "debugging"
        else:
            return "error"
    
    def _agent_router(self, state: WorkflowState) -> Literal["implementation", "debugging", "error"]:
        """Route to specific agent based on selection."""
        if state["error_occurred"]:
            return "error"
        
        selected_agent = state.get("selected_agent")
        if selected_agent == "implementation":
            return "implementation"
        elif selected_agent == "debugging":
            return "debugging"
        else:
            return "error"
    
    def _quality_router(self, state: WorkflowState) -> Literal["passed", "failed", "retry"]:
        """Route based on quality assurance results."""
        if state["error_occurred"]:
            return "failed"
        
        quality_checks = state.get("quality_checks", {})
        if quality_checks.get("overall_passed", True):
            return "passed"
        elif quality_checks.get("should_retry", False):
            return "retry"
        else:
            return "failed"
    
    # Helper Methods
    
    def _perform_quality_checks(self, state: WorkflowState) -> Dict[str, Any]:
        """Perform comprehensive quality checks on the agent response."""
        agent_response = state["agent_response"]
        checks = {
            "content_length_ok": len(agent_response["content"]) > 50,
            "confidence_adequate": agent_response["confidence"] > 0.3,
            "educational_metadata_present": bool(agent_response.get("educational_metadata")),
            "response_type_valid": agent_response.get("response_type") in [rt.value for rt in ResponseType],
            "processing_time_reasonable": agent_response.get("processing_time_ms", 0) < 25000
        }
        
        checks["overall_passed"] = all(checks.values())
        
        return checks
    
    def _generate_fallback_response(self, query: str, error: str) -> str:
        """Generate a helpful fallback response when the system encounters errors."""
        return f"""I apologize, but I encountered an issue processing your programming question. 

Here's what I can suggest:

1. **Rephrase your question**: Try to be more specific about what you're trying to accomplish or what's not working.

2. **Provide context**: If you have code that's not working, please share it along with any error messages.

3. **Break down the problem**: If your question is complex, try asking about one specific aspect at a time.

4. **Specify your goal**: Let me know whether you're:
   - Planning how to implement something new
   - Debugging code that's not working correctly
   - Understanding a concept or error message

I'm here to help with your programming learning journey. Feel free to try asking your question again with additional details!

*System note: {error}*"""
    
    def _load_conversation_history(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Load conversation history for the session."""
        if not self.conversation_memory:
            return None
        
        return self.conversation_memory.get(session_id, [])
    
    def _update_conversation_memory(self, 
                                   session_id: str, 
                                   query: str, 
                                   response: OrchestratorResponse):
        """Update conversation memory with the latest interaction."""
        if not self.conversation_memory:
            return
        
        if session_id not in self.conversation_memory:
            self.conversation_memory[session_id] = []
        
        # Add the latest interaction
        interaction = {
            "query": query,
            "response": response.content,
            "agent_used": response.agent_used,
            "srl_phase": response.srl_phase,
            "timestamp": time.time()
        }
        
        self.conversation_memory[session_id].append(interaction)
        
        # Keep only recent interactions (memory management)
        if len(self.conversation_memory[session_id]) > 10:
            self.conversation_memory[session_id] = self.conversation_memory[session_id][-10:]
    
    def _build_orchestrator_response(self, 
                                   final_state: WorkflowState, 
                                   workflow_id: str,
                                   start_time: float) -> OrchestratorResponse:
        """Build the final orchestrator response from workflow state."""
        agent_response = final_state.get("agent_response", {})
        classification_result = final_state.get("classification_result", {})
        performance_metrics = final_state.get("performance_metrics", {})
        
        # Determine conversation turn
        conversation_turn = 1
        if final_state.get("conversation_history"):
            conversation_turn = len(final_state["conversation_history"]) + 1
        
        return OrchestratorResponse(
            content=agent_response.get("content", "I apologize, but I was unable to process your request."),
            agent_used=final_state.get("selected_agent", "unknown"),
            srl_phase=final_state.get("srl_phase", "unknown"),
            classification_confidence=classification_result.get("confidence", 0.0),
            educational_metadata=agent_response.get("educational_metadata", {}),
            concepts_covered=agent_response.get("educational_metadata", {}).get("concepts_covered", []),
            learning_objectives=agent_response.get("educational_metadata", {}).get("learning_objectives", []),
            suggested_follow_up=agent_response.get("suggested_follow_up", []),
            total_processing_time_ms=performance_metrics.get("total_processing_time_ms", (time.time() - start_time) * 1000),
            classification_time_ms=performance_metrics.get("classification_time_ms", 0.0),
            agent_processing_time_ms=performance_metrics.get("agent_processing_time_ms", 0.0),
            session_id=final_state["session_id"],
            workflow_id=workflow_id,
            quality_checks_passed=performance_metrics.get("quality_check_passed", True),
            fallback_used=final_state.get("selected_agent") == "fallback",
            conversation_turn=conversation_turn,
            maintains_context=bool(final_state.get("conversation_history"))
        )
    
    def _create_error_response(self, 
                             query: str, 
                             session_id: str, 
                             workflow_id: str,
                             start_time: float, 
                             error: str) -> OrchestratorResponse:
        """Create an error response when workflow fails completely."""
        return OrchestratorResponse(
            content=self._generate_fallback_response(query, error),
            agent_used="error_handler",
            srl_phase="unknown",
            classification_confidence=0.0,
            educational_metadata={"system_error": True, "error_message": error},
            total_processing_time_ms=(time.time() - start_time) * 1000,
            session_id=session_id,
            workflow_id=workflow_id,
            quality_checks_passed=False,
            fallback_used=True
        )
    
    def _update_performance_metrics(self, response: OrchestratorResponse):
        """Update internal performance tracking metrics."""
        self.total_queries_processed += 1
        
        # Update running average of processing time
        current_avg = self.average_processing_time
        new_time = response.total_processing_time_ms
        self.average_processing_time = (
            (current_avg * (self.total_queries_processed - 1) + new_time) 
            / self.total_queries_processed
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics for the orchestrator."""
        return {
            "total_queries_processed": self.total_queries_processed,
            "average_processing_time_ms": self.average_processing_time,
            "conversation_sessions_active": len(self.conversation_memory) if self.conversation_memory else 0,
            "config": {
                "quality_checks_enabled": self.config.enable_quality_checks,
                "max_processing_time_ms": self.config.max_processing_time_ms,
                "minimum_confidence_required": self.config.require_minimum_confidence,
                "fallback_routing_enabled": self.config.enable_fallback_routing
            },
            "agent_stats": {
                "implementation_agent": self.implementation_agent.get_performance_stats(),
                "debugging_agent": self.debugging_agent.get_performance_stats()
            },
            "classifier_stats": self.srl_classifier.get_performance_stats()
        }


if __name__ == "__main__":
    # Orchestrator test
    try:
        orchestrator = OrchestratorAgent()
        
        # Test cases
        test_queries = [
            {
                "query": "How do I implement a binary search algorithm?",
                "expected_agent": "implementation"
            },
            {
                "query": "My code is giving me an IndexError",
                "code_snippet": "arr = [1,2,3]\nprint(arr[5])",
                "expected_agent": "debugging"
            }
        ]
        
        for i, test in enumerate(test_queries):
            print(f"\nTest {i+1}: {test['query'][:50]}...")
            
            response = orchestrator.process_query(
                query=test["query"],
                code_snippet=test.get("code_snippet"),
                session_id=f"test_session_{i}"
            )
            
            print(f"Agent used: {response.agent_used}")
            print(f"SRL Phase: {response.srl_phase}")
            print(f"Confidence: {response.classification_confidence:.3f}")
            print(f"Processing time: {response.total_processing_time_ms:.1f}ms")
            print(f"Quality passed: {response.quality_checks_passed}")
        
        # Performance stats
        print(f"\nPerformance Stats:")
        stats = orchestrator.get_performance_stats()
        print(f"Total queries: {stats['total_queries_processed']}")
        print(f"Average time: {stats['average_processing_time_ms']:.1f}ms")
        
        print("✅ Orchestrator test completed successfully!")
        
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
