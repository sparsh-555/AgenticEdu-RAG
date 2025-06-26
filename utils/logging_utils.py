"""
Comprehensive Logging System for Agentic Edu-RAG System

This module implements a sophisticated logging infrastructure designed specifically
for educational AI research and production systems. It provides structured logging,
performance monitoring, educational event tracking, and comprehensive analytics.

Key Features:
1. Multi-level Logging: Debug, Info, Warning, Error with contextual information
2. Educational Event Tracking: Specialized logging for SRL research
3. Performance Monitoring: Request timing, resource usage, and optimization metrics
4. Correlation Tracking: Request tracing across multi-agent workflows
5. Research Analytics: Data collection for educational effectiveness analysis
6. Security Compliance: Safe handling of sensitive educational data

Design Principles:
- Structured Logging: Machine-readable logs for analysis
- Privacy-First: No storage of sensitive student information
- Performance-Optimized: Minimal impact on system responsiveness
- Research-Oriented: Support for educational research requirements
- Production-Ready: Robust error handling and log rotation
"""

import json
import time
import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from contextlib import contextmanager
import logging
import logging.handlers
from functools import wraps

from loguru import logger as loguru_logger
from pydantic import BaseModel, Field

from ..config.settings import get_settings


class EventType(Enum):
    """Standardized event types for consistent logging across the system."""
    
    # System lifecycle events
    SYSTEM_START = "system_start"
    SYSTEM_SHUTDOWN = "system_shutdown"
    COMPONENT_INIT = "component_init"
    
    # Query processing events
    QUERY_RECEIVED = "query_received"
    QUERY_ROUTED = "query_routed"
    QUERY_COMPLETED = "query_completed"
    QUERY_FAILED = "query_failed"
    
    # SRL classification events
    SRL_CLASSIFICATION = "srl_classification"
    CLASSIFICATION_CONFIDENCE = "classification_confidence"
    
    # Agent interaction events
    AGENT_INVOKED = "agent_invoked"
    AGENT_RESPONSE = "agent_response"
    AGENT_ERROR = "agent_error"
    
    # Knowledge retrieval events
    KNOWLEDGE_RETRIEVED = "knowledge_retrieved"
    RAG_QUERY = "rag_query"
    CONTEXT_ENHANCED = "context_enhanced"
    
    # LLM interaction events
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    LLM_ERROR = "llm_error"
    
    # Educational events
    LEARNING_INTERACTION = "learning_interaction"
    STUDENT_PROGRESS = "student_progress"
    EDUCATIONAL_OUTCOME = "educational_outcome"
    
    # Performance events
    PERFORMANCE_METRIC = "performance_metric"
    RESOURCE_USAGE = "resource_usage"
    OPTIMIZATION_APPLIED = "optimization_applied"
    
    # Error events
    ERROR_OCCURRED = "error_occurred"
    WARNING_ISSUED = "warning_issued"
    RECOVERY_ATTEMPTED = "recovery_attempted"


class LogLevel(Enum):
    """Log levels with educational system context."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogContext:
    """
    Correlation context for tracking requests across the multi-agent system.
    
    This context travels with requests through the entire workflow, enabling
    comprehensive tracing and analysis of educational interactions.
    """
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    workflow_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    # Educational context
    student_level: Optional[str] = None
    learning_phase: Optional[str] = None
    domain: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Performance tracking data for optimization and monitoring."""
    operation_name: str
    start_time: float
    end_time: float
    duration_ms: float = field(init=False)
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    # Educational specific metrics
    query_complexity: Optional[str] = None
    agent_type: Optional[str] = None
    tokens_processed: Optional[int] = None
    
    def __post_init__(self):
        self.duration_ms = (self.end_time - self.start_time) * 1000


class EducationalEvent(BaseModel):
    """Specialized event model for educational research data."""
    event_type: str = Field(..., description="Type of educational event")
    timestamp: float = Field(default_factory=time.time)
    
    # Educational metadata
    srl_phase: Optional[str] = Field(default=None, description="Self-regulated learning phase")
    agent_type: Optional[str] = Field(default=None, description="Agent that handled interaction")
    student_level: Optional[str] = Field(default=None, description="Student proficiency level")
    query_complexity: Optional[str] = Field(default=None, description="Query complexity assessment")
    
    # Interaction data
    classification_confidence: Optional[float] = Field(default=None, description="SRL classification confidence")
    response_quality: Optional[float] = Field(default=None, description="Response quality score")
    educational_effectiveness: Optional[float] = Field(default=None, description="Educational effectiveness score")
    
    # Performance data
    response_time_ms: Optional[float] = Field(default=None, description="Response generation time")
    tokens_used: Optional[int] = Field(default=None, description="LLM tokens consumed")
    
    # Privacy-safe identifiers
    session_hash: Optional[str] = Field(default=None, description="Hashed session identifier")
    
    class Config:
        extra = "allow"  # Allow additional fields for research flexibility


class EducationalLogger:
    """
    Comprehensive logging system optimized for educational AI research.
    
    This logger provides structured, privacy-compliant logging with specialized
    support for educational research metrics, multi-agent correlation, and
    performance optimization. It's designed to support both development and
    production environments while maintaining research data quality.
    """
    
    def __init__(self):
        """Initialize the educational logging system."""
        self.settings = get_settings()
        
        # Setup file paths
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize loguru with custom configuration
        self._setup_loguru()
        
        # Educational events storage
        self.educational_events: List[EducationalEvent] = []
        self.events_lock = threading.Lock()
        
        # Performance tracking
        self.performance_metrics: List[PerformanceMetrics] = []
        self.active_timers: Dict[str, float] = {}
        
        # Session management
        self.active_sessions = set()
        
        # Analytics counters
        self.event_counters = {event_type.value: 0 for event_type in EventType}
        
        loguru_logger.info("Educational logging system initialized", 
                          extra={"log_level": self.settings.system.log_level})
    
    def _setup_loguru(self):
        """Configure loguru with educational system requirements."""
        # Remove default handler
        loguru_logger.remove()
        
        # Console handler with colored output for development
        loguru_logger.add(
            sink=lambda msg: print(msg, end=""),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            level=self.settings.system.log_level,
            colorize=True
        )
        
        # Structured JSON file handler for production analysis
        loguru_logger.add(
            sink=self.log_dir / "structured.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            rotation="100 MB",
            retention="30 days",
            compression="gzip",
            serialize=True,  # JSON format
            enqueue=True     # Thread-safe
        )
        
        # Educational events file handler
        loguru_logger.add(
            sink=self.log_dir / "educational_events.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="INFO",
            rotation="50 MB",
            retention="90 days",  # Keep educational data longer for research
            compression="gzip",
            filter=lambda record: "educational_event" in record["extra"],
            enqueue=True
        )
        
        # Error file handler
        loguru_logger.add(
            sink=self.log_dir / "errors.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            level="WARNING",
            rotation="10 MB",
            retention="180 days",
            compression="gzip",
            enqueue=True
        )
    
    def log_event(self, 
                  event_type: EventType,
                  message: str,
                  context: Optional[LogContext] = None,
                  level: str = "INFO",
                  extra_data: Optional[Dict[str, Any]] = None):
        """
        Log a system event with full context and correlation tracking.
        
        Args:
            event_type: Type of event being logged
            message: Human-readable event description
            context: Correlation context for request tracking
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            extra_data: Additional structured data for analysis
        """
        context = context or LogContext()
        extra_data = extra_data or {}
        
        # Build structured log entry
        log_data = {
            "event_type": event_type.value,
            "correlation_id": context.correlation_id,
            "session_id": context.session_id,
            "user_id": context.user_id,
            "workflow_id": context.workflow_id,
            "timestamp": time.time(),
            "student_level": context.student_level,
            "learning_phase": context.learning_phase,
            "domain": context.domain,
            **extra_data
        }
        
        # Update counters
        self.event_counters[event_type.value] += 1
        
        # Log to loguru with structured data
        getattr(loguru_logger, level.lower())(
            f"[{event_type.value}] {message}",
            extra=log_data
        )
        
        # Track educational events separately for research
        if self._is_educational_event(event_type):
            self._track_educational_event(event_type, message, context, extra_data)
    
    def log_educational_event(self, event: EducationalEvent):
        """
        Log a specialized educational event for research analysis.
        
        Args:
            event: Educational event with research metadata
        """
        with self.events_lock:
            self.educational_events.append(event)
        
        # Also log to structured system
        loguru_logger.info(
            f"Educational Event: {event.event_type}",
            extra={"educational_event": True, **event.dict()}
        )
    
    def log_query_processing(self,
                           query: str,
                           classification: str,
                           confidence: float,
                           context: Optional[LogContext] = None):
        """
        Log query processing with educational research metadata.
        
        Args:
            query: Student query (truncated for privacy)
            classification: SRL phase classification
            confidence: Classification confidence score
            context: Correlation context
        """
        # Create privacy-safe query representation
        query_preview = query[:100] + "..." if len(query) > 100 else query
        query_hash = self._hash_sensitive_data(query)
        
        educational_event = EducationalEvent(
            event_type="query_processing",
            srl_phase=classification,
            classification_confidence=confidence,
            query_complexity=self._assess_query_complexity(query),
            session_hash=self._hash_sensitive_data(context.session_id) if context and context.session_id else None
        )
        
        self.log_educational_event(educational_event)
        
        self.log_event(
            EventType.QUERY_COMPLETED,
            f"Query processed: {classification} (confidence: {confidence:.3f})",
            context=context,
            extra_data={
                "query_preview": query_preview,
                "query_hash": query_hash,
                "classification": classification,
                "confidence": confidence,
                "query_length": len(query)
            }
        )
    
    def log_agent_interaction(self,
                            agent_type: str,
                            action: str,
                            context: Optional[LogContext] = None,
                            performance_data: Optional[PerformanceMetrics] = None):
        """
        Log agent interactions with performance tracking.
        
        Args:
            agent_type: Type of agent (implementation, debugging, orchestrator)
            action: Action performed by agent
            context: Correlation context
            performance_data: Performance metrics for optimization
        """
        log_data = {
            "agent_type": agent_type,
            "action": action
        }
        
        if performance_data:
            log_data.update({
                "duration_ms": performance_data.duration_ms,
                "memory_usage_mb": performance_data.memory_usage_mb,
                "tokens_processed": performance_data.tokens_processed
            })
            
            # Store performance metrics for analysis
            self.performance_metrics.append(performance_data)
        
        self.log_event(
            EventType.AGENT_INVOKED,
            f"Agent {agent_type} performed {action}",
            context=context,
            extra_data=log_data
        )
    
    def log_rag_operation(self,
                         operation: str,
                         query: str,
                         results_count: int,
                         context: Optional[LogContext] = None):
        """
        Log RAG (Retrieval-Augmented Generation) operations.
        
        Args:
            operation: Type of RAG operation (query, retrieval, enhancement)
            query: Query used for retrieval (truncated for privacy)
            results_count: Number of results retrieved
            context: Correlation context
        """
        query_preview = query[:50] + "..." if len(query) > 50 else query
        
        self.log_event(
            EventType.KNOWLEDGE_RETRIEVED,
            f"RAG {operation}: {results_count} results for query",
            context=context,
            extra_data={
                "operation": operation,
                "query_preview": query_preview,
                "results_count": results_count,
                "query_length": len(query)
            }
        )
    
    @contextmanager
    def performance_timer(self, operation_name: str, context: Optional[LogContext] = None):
        """
        Context manager for timing operations with automatic logging.
        
        Args:
            operation_name: Name of operation being timed
            context: Correlation context
            
        Usage:
            with logger.performance_timer("classification", context):
                result = classifier.classify(query)
        """
        start_time = time.time()
        correlation_id = context.correlation_id if context else str(uuid.uuid4())
        
        # Track active timer
        timer_key = f"{operation_name}_{correlation_id}"
        self.active_timers[timer_key] = start_time
        
        try:
            yield
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # Remove from active timers
            self.active_timers.pop(timer_key, None)
            
            # Log performance
            self.log_event(
                EventType.PERFORMANCE_METRIC,
                f"Operation {operation_name} completed in {duration_ms:.1f}ms",
                context=context,
                extra_data={
                    "operation": operation_name,
                    "duration_ms": duration_ms,
                    "start_time": start_time,
                    "end_time": end_time
                }
            )
            
            # Store performance metrics
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time
            )
            self.performance_metrics.append(metrics)
    
    def _is_educational_event(self, event_type: EventType) -> bool:
        """Check if event type is educational for research tracking."""
        educational_events = {
            EventType.SRL_CLASSIFICATION,
            EventType.LEARNING_INTERACTION,
            EventType.STUDENT_PROGRESS,
            EventType.EDUCATIONAL_OUTCOME,
            EventType.QUERY_COMPLETED
        }
        return event_type in educational_events
    
    def _track_educational_event(self,
                               event_type: EventType,
                               message: str,
                               context: LogContext,
                               extra_data: Dict[str, Any]):
        """Track educational events for research analysis."""
        educational_event = EducationalEvent(
            event_type=event_type.value,
            srl_phase=context.learning_phase,
            student_level=context.student_level,
            session_hash=self._hash_sensitive_data(context.session_id) if context.session_id else None,
            **{k: v for k, v in extra_data.items() if k in EducationalEvent.__fields__}
        )
        
        self.log_educational_event(educational_event)
    
    def _hash_sensitive_data(self, data: Optional[str]) -> Optional[str]:
        """Create privacy-safe hash of sensitive data."""
        if not data:
            return None
        
        import hashlib
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _assess_query_complexity(self, query: str) -> str:
        """Simple heuristic to assess query complexity for research."""
        length = len(query)
        word_count = len(query.split())
        
        if length > 300 or word_count > 50:
            return "high"
        elif length > 150 or word_count > 25:
            return "medium"
        else:
            return "low"
    
    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """
        Get analytics for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session analytics
        """
        session_hash = self._hash_sensitive_data(session_id)
        
        # Filter events for this session
        session_events = [
            event for event in self.educational_events
            if event.session_hash == session_hash
        ]
        
        if not session_events:
            return {"session_id": session_id, "events": 0}
        
        # Calculate analytics
        total_events = len(session_events)
        avg_confidence = sum(
            event.classification_confidence 
            for event in session_events 
            if event.classification_confidence
        ) / max(1, sum(1 for event in session_events if event.classification_confidence))
        
        srl_phases = [event.srl_phase for event in session_events if event.srl_phase]
        phase_distribution = {
            phase: srl_phases.count(phase) for phase in set(srl_phases)
        }
        
        return {
            "session_id": session_id,
            "total_events": total_events,
            "average_confidence": avg_confidence,
            "srl_phase_distribution": phase_distribution,
            "duration_minutes": (
                (session_events[-1].timestamp - session_events[0].timestamp) / 60
                if len(session_events) > 1 else 0
            )
        }
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive system analytics.
        
        Returns:
            Dictionary with system-wide analytics
        """
        # Event distribution
        total_events = sum(self.event_counters.values())
        
        # Performance analytics
        if self.performance_metrics:
            avg_response_time = sum(m.duration_ms for m in self.performance_metrics) / len(self.performance_metrics)
            max_response_time = max(m.duration_ms for m in self.performance_metrics)
        else:
            avg_response_time = max_response_time = 0
        
        # Educational analytics
        educational_events_count = len(self.educational_events)
        
        if self.educational_events:
            # SRL phase distribution
            srl_phases = [e.srl_phase for e in self.educational_events if e.srl_phase]
            srl_distribution = {
                phase: srl_phases.count(phase) for phase in set(srl_phases)
            }
            
            # Average confidence
            confidences = [e.classification_confidence for e in self.educational_events if e.classification_confidence]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        else:
            srl_distribution = {}
            avg_confidence = 0
        
        return {
            "total_events": total_events,
            "event_distribution": dict(self.event_counters),
            "educational_events": educational_events_count,
            "active_sessions": len(self.active_sessions),
            "performance": {
                "avg_response_time_ms": avg_response_time,
                "max_response_time_ms": max_response_time,
                "total_operations": len(self.performance_metrics)
            },
            "educational_analytics": {
                "srl_phase_distribution": srl_distribution,
                "average_classification_confidence": avg_confidence
            }
        }
    
    def export_research_data(self, 
                           output_path: Path,
                           privacy_level: str = "high") -> Dict[str, Any]:
        """
        Export educational data for research analysis.
        
        Args:
            output_path: Path to save research data
            privacy_level: Privacy level (high, medium, low)
            
        Returns:
            Summary of exported data
        """
        research_data = {
            "metadata": {
                "export_timestamp": time.time(),
                "privacy_level": privacy_level,
                "total_events": len(self.educational_events),
                "date_range": {
                    "start": min(e.timestamp for e in self.educational_events) if self.educational_events else None,
                    "end": max(e.timestamp for e in self.educational_events) if self.educational_events else None
                }
            },
            "events": []
        }
        
        # Process events based on privacy level
        for event in self.educational_events:
            event_data = event.dict()
            
            if privacy_level == "high":
                # Remove any potentially identifying information
                event_data.pop("session_hash", None)
                event_data["timestamp"] = int(event_data["timestamp"])  # Round to seconds
            elif privacy_level == "medium":
                # Keep session hashes but round timestamps
                event_data["timestamp"] = int(event_data["timestamp"])
            # Low privacy keeps all data
            
            research_data["events"].append(event_data)
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(research_data, f, indent=2)
        
        self.log_event(
            EventType.SYSTEM_START,
            f"Research data exported: {len(self.educational_events)} events",
            extra_data={
                "output_path": str(output_path),
                "privacy_level": privacy_level,
                "events_count": len(self.educational_events)
            }
        )
        
        return research_data["metadata"]


# Global logger instance
_logger: Optional[EducationalLogger] = None


def get_logger(reload: bool = False) -> EducationalLogger:
    """
    Get global educational logger instance (singleton pattern).
    
    Args:
        reload: Force creation of new logger instance
        
    Returns:
        EducationalLogger instance
    """
    global _logger
    if _logger is None or reload:
        _logger = EducationalLogger()
    return _logger


def create_context(session_id: Optional[str] = None,
                  user_id: Optional[str] = None,
                  workflow_id: Optional[str] = None,
                  student_level: Optional[str] = None,
                  learning_phase: Optional[str] = None,
                  domain: Optional[str] = None) -> LogContext:
    """
    Create a new log context for request correlation.
    
    Args:
        session_id: Session identifier
        user_id: User identifier
        workflow_id: Workflow identifier
        student_level: Student proficiency level
        learning_phase: Current learning phase
        domain: Programming domain
        
    Returns:
        LogContext instance
    """
    return LogContext(
        session_id=session_id,
        user_id=user_id,
        workflow_id=workflow_id,
        student_level=student_level,
        learning_phase=learning_phase,
        domain=domain
    )


# Decorators for automatic logging
def log_function_call(event_type: EventType = EventType.SYSTEM_START):
    """
    Decorator to automatically log function calls with performance timing.
    
    Args:
        event_type: Type of event to log
        
    Usage:
        @log_function_call(EventType.AGENT_INVOKED)
        def process_query(self, query):
            return result
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            function_name = f"{func.__module__}.{func.__qualname__}"
            
            with logger.performance_timer(function_name):
                logger.log_event(
                    event_type,
                    f"Function {function_name} called",
                    extra_data={
                        "function": function_name,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs)
                    }
                )
                
                result = func(*args, **kwargs)
                
                logger.log_event(
                    EventType.SYSTEM_START,
                    f"Function {function_name} completed",
                    extra_data={"function": function_name}
                )
                
                return result
        return wrapper
    return decorator


if __name__ == "__main__":
    # Logging system test
    try:
        logger = get_logger()
        
        # Test basic logging
        context = create_context(
            session_id="test_session_123",
            student_level="intermediate"
        )
        
        logger.log_event(
            EventType.SYSTEM_START,
            "Testing logging system",
            context=context,
            extra_data={"test": True}
        )
        
        # Test performance timing
        with logger.performance_timer("test_operation", context):
            time.sleep(0.1)  # Simulate work
        
        # Test educational event
        educational_event = EducationalEvent(
            event_type="test_interaction",
            srl_phase="forethought",
            student_level="intermediate",
            classification_confidence=0.85
        )
        logger.log_educational_event(educational_event)
        
        # Test analytics
        analytics = logger.get_system_analytics()
        print(f"System analytics: {analytics}")
        
        print("✅ Logging system test completed successfully!")
        
    except Exception as e:
        print(f"❌ Logging system test failed: {e}")
