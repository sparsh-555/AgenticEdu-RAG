"""
Main Application Entry Point for Agentic Edu-RAG System

This module serves as the primary entry point for the multi-agent educational RAG system.
It demonstrates the complete workflow from query reception through SRL classification
to specialized agent processing and response generation.

Key Features:
1. System Initialization: Set up all components and load educational content
2. Interactive Demo: Command-line interface for testing the system
3. Complete Workflow: End-to-end processing from query to educational response
4. Performance Monitoring: Comprehensive metrics collection and reporting
5. Error Handling: Robust error management with graceful degradation

Usage Examples:
- python main.py --demo                    # Run interactive demo
- python main.py --load-content           # Load educational content
- python main.py --query "How do I implement binary search?"  # Single query
- python main.py --benchmark              # Run performance benchmarks
- python main.py --stats                  # Show system statistics

This implementation demonstrates enterprise-grade system integration patterns
and provides a complete example of how to build sophisticated educational AI systems.
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import all system components
from agents.orchestrator import OrchestratorAgent, OrchestratorConfig
from classification.srl_classifier import get_srl_classifier, ClassificationContext, SRLPhase
from rag.knowledge_base import get_knowledge_base, RetrievalContext, RetrievalStrategy
from rag.content_loader import EducationalContentLoader, ContentLoadingConfig
from utils.logging_utils import get_logger, create_context, EventType
from utils.api_utils import get_openai_client
from config.settings import get_settings


class AgenticEduRAGSystem:
    """
    Complete Agentic Educational RAG System.
    
    This class integrates all system components and provides a unified interface
    for educational query processing using Self-Regulated Learning principles
    and multi-agent architecture.
    
    The system demonstrates:
    - Multi-agent orchestration with LangGraph
    - SRL-based query classification and routing
    - Educational content retrieval and augmentation
    - Comprehensive performance monitoring and evaluation
    """
    
    def __init__(self, 
                 auto_initialize: bool = True,
                 load_content: bool = True,
                 enable_demo_mode: bool = False):
        """
        Initialize the complete Agentic Edu-RAG system.
        
        Args:
            auto_initialize: Whether to automatically initialize all components
            load_content: Whether to load educational content on startup
            enable_demo_mode: Whether to enable demo-specific features
        """
        self.logger = get_logger()
        self.settings = get_settings()
        
        # System components (initialized later)
        self.orchestrator = None
        self.knowledge_base = None
        self.classifier = None
        self.openai_client = None
        
        # System state
        self.is_initialized = False
        self.content_loaded = False
        self.demo_mode = enable_demo_mode
        
        # Performance tracking
        self.system_stats = {
            "startup_time_ms": 0,
            "queries_processed": 0,
            "total_processing_time_ms": 0,
            "initialization_time": None,
            "content_loading_time": None
        }
        
        self.logger.log_event(
            EventType.SYSTEM_START,
            "Agentic Edu-RAG System created",
            extra_data={
                "auto_initialize": auto_initialize,
                "load_content": load_content,
                "demo_mode": enable_demo_mode
            }
        )
        
        if auto_initialize:
            self.initialize_system()
            
        if load_content and self.is_initialized:
            self.load_educational_content()
    
    def initialize_system(self) -> Dict[str, Any]:
        """
        Initialize all system components in the correct order.
        
        Returns:
            Dictionary with initialization results and timing
        """
        start_time = time.time()
        
        try:
            self.logger.log_event(
                EventType.SYSTEM_START,
                "Starting system initialization"
            )
            
            # Step 1: Initialize core utilities
            self.logger.log_event(EventType.SYSTEM_START, "Initializing API client")
            self.openai_client = get_openai_client()
            
            # Step 2: Initialize knowledge base
            self.logger.log_event(EventType.SYSTEM_START, "Initializing knowledge base")
            self.knowledge_base = get_knowledge_base()
            
            # Step 3: Initialize SRL classifier
            self.logger.log_event(EventType.SYSTEM_START, "Initializing SRL classifier")
            self.classifier = get_srl_classifier()
            
            # Step 4: Initialize orchestrator with multi-agent workflow
            self.logger.log_event(EventType.SYSTEM_START, "Initializing orchestrator")
            orchestrator_config = OrchestratorConfig(
                enable_quality_checks=True,
                enable_conversation_memory=True,
                collect_educational_metrics=True
            )
            self.orchestrator = OrchestratorAgent(config=orchestrator_config)
            
            # System initialization complete
            initialization_time = (time.time() - start_time) * 1000
            self.system_stats["startup_time_ms"] = initialization_time
            self.system_stats["initialization_time"] = time.time()
            self.is_initialized = True
            
            self.logger.log_event(
                EventType.SYSTEM_START,
                f"System initialization completed in {initialization_time:.1f}ms",
                extra_data={"initialization_time_ms": initialization_time}
            )
            
            return {
                "status": "success",
                "initialization_time_ms": initialization_time,
                "components_initialized": [
                    "openai_client",
                    "knowledge_base", 
                    "srl_classifier",
                    "orchestrator_agent"
                ]
            }
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"System initialization failed: {str(e)}",
                level="ERROR"
            )
            return {
                "status": "error",
                "error": str(e),
                "initialization_time_ms": (time.time() - start_time) * 1000
            }
    
    def load_educational_content(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load educational content into the knowledge base.
        
        Args:
            force_reload: Whether to reload content even if already loaded
            
        Returns:
            Dictionary with loading results
        """
        if not self.is_initialized:
            return {"status": "error", "error": "System not initialized"}
        
        start_time = time.time()
        
        try:
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                "Starting educational content loading",
                extra_data={"force_reload": force_reload}
            )
            
            # Load content using knowledge base
            loading_results = self.knowledge_base.load_all_content(
                force_reload=force_reload
            )
            
            loading_time = (time.time() - start_time) * 1000
            self.system_stats["content_loading_time"] = loading_time
            
            if loading_results["status"] == "success":
                self.content_loaded = True
                
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                f"Content loading completed: {loading_results.get('chunks_stored', 0)} chunks loaded",
                extra_data={
                    "loading_time_ms": loading_time,
                    "results": loading_results
                }
            )
            
            return loading_results
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Content loading failed: {str(e)}",
                level="ERROR"
            )
            return {"status": "error", "error": str(e)}
    
    def process_query(self, 
                     query: str,
                     code_snippet: Optional[str] = None,
                     error_message: Optional[str] = None,
                     student_level: Optional[str] = "intermediate",
                     session_id: Optional[str] = None,
                     user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a student query through the complete multi-agent workflow.
        
        Args:
            query: Student's programming question
            code_snippet: Optional code context
            error_message: Optional error message
            student_level: Student proficiency level
            session_id: Session identifier for conversation tracking
            user_id: User identifier
            
        Returns:
            Comprehensive response with educational content and metadata
        """
        if not self.is_initialized:
            return {
                "status": "error", 
                "error": "System not initialized. Please run initialize_system() first."
            }
        
        start_time = time.time()
        
        try:
            # Process query through orchestrator
            orchestrator_response = self.orchestrator.process_query(
                query=query,
                code_snippet=code_snippet,
                error_message=error_message,
                student_level=student_level,
                session_id=session_id,
                user_id=user_id
            )
            
            # Update system statistics
            processing_time = (time.time() - start_time) * 1000
            self.system_stats["queries_processed"] += 1
            self.system_stats["total_processing_time_ms"] += processing_time
            
            # Create comprehensive response
            response = {
                "status": "success",
                "content": orchestrator_response.content,
                "agent_used": orchestrator_response.agent_used,
                "srl_phase": orchestrator_response.srl_phase,
                "classification_confidence": orchestrator_response.classification_confidence,
                "educational_metadata": orchestrator_response.educational_metadata,
                "suggested_follow_up": orchestrator_response.suggested_follow_up,
                "performance_metrics": {
                    "total_processing_time_ms": orchestrator_response.total_processing_time_ms,
                    "classification_time_ms": orchestrator_response.classification_time_ms,
                    "agent_processing_time_ms": orchestrator_response.agent_processing_time_ms
                },
                "system_metadata": {
                    "session_id": orchestrator_response.session_id,
                    "workflow_id": orchestrator_response.workflow_id,
                    "quality_checks_passed": orchestrator_response.quality_checks_passed,
                    "conversation_turn": orchestrator_response.conversation_turn
                }
            }
            
            self.logger.log_event(
                EventType.QUERY_COMPLETED,
                f"Query processed successfully in {processing_time:.1f}ms",
                extra_data={
                    "processing_time_ms": processing_time,
                    "agent_used": orchestrator_response.agent_used,
                    "srl_phase": orchestrator_response.srl_phase
                }
            )
            
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Query processing failed: {str(e)}",
                level="ERROR",
                extra_data={"processing_time_ms": processing_time}
            )
            
            return {
                "status": "error",
                "error": str(e),
                "processing_time_ms": processing_time
            }
    
    def run_interactive_demo(self):
        """
        Run an interactive command-line demo of the system.
        
        This provides a user-friendly interface for testing the complete
        multi-agent workflow with various types of programming questions.
        """
        print("\n" + "="*80)
        print("🎓 AGENTIC EDU-RAG SYSTEM - INTERACTIVE DEMO")
        print("="*80)
        print("\nWelcome to the Self-Regulated Learning Multi-Agent Educational System!")
        print("This system uses specialized agents to provide contextual programming help.")
        print("\nAgent Specializations:")
        print("  🎯 Implementation Agent: Planning, strategy, algorithm design (Forethought Phase)")
        print("  🔧 Debugging Agent: Error analysis, troubleshooting (Performance Phase)")
        print("  🤖 Orchestrator: Intelligent routing based on SRL classification")
        
        if not self.is_initialized:
            print("\n⚠️  System not initialized. Initializing now...")
            init_result = self.initialize_system()
            if init_result["status"] != "success":
                print(f"❌ Initialization failed: {init_result['error']}")
                return
            print("✅ System initialized successfully!")
        
        if not self.content_loaded:
            print("\n📚 Loading educational content...")
            load_result = self.load_educational_content()
            if load_result["status"] == "success":
                print(f"✅ Loaded {load_result.get('chunks_stored', 0)} content chunks")
            else:
                print(f"⚠️  Content loading failed: {load_result.get('error', 'Unknown error')}")
                print("Continuing with demo (some features may be limited)")
        
        print("\n" + "-"*80)
        print("DEMO COMMANDS:")
        print("  Type your programming question and press Enter")
        print("  Commands: 'stats' (show statistics), 'examples' (show examples), 'quit' (exit)")
        print("-"*80)
        
        session_id = f"demo_session_{int(time.time())}"
        
        while True:
            try:
                print("\n💭 Your programming question:")
                user_input = input(">>> ").strip()
                
                if not user_input:
                    continue
                    
                # Handle special commands
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'stats':
                    self._show_system_stats()
                    continue
                elif user_input.lower() == 'examples':
                    self._show_example_queries()
                    continue
                
                # Process the query
                print("\n🔄 Processing your query...")
                response = self.process_query(
                    query=user_input,
                    session_id=session_id,
                    student_level="intermediate"
                )
                
                if response["status"] == "success":
                    print(f"\n🤖 Agent Used: {response['agent_used'].title()} Agent")
                    print(f"📊 SRL Phase: {response['srl_phase']}")
                    print(f"🎯 Confidence: {response['classification_confidence']:.2f}")
                    print(f"⏱️  Processing Time: {response['performance_metrics']['total_processing_time_ms']:.1f}ms")
                    
                    print("\n📝 Response:")
                    print("-" * 60)
                    print(response["content"])
                    print("-" * 60)
                    
                    if response.get("suggested_follow_up"):
                        print("\n💡 Suggested follow-up questions:")
                        for i, suggestion in enumerate(response["suggested_follow_up"][:3], 1):
                            print(f"  {i}. {suggestion}")
                else:
                    print(f"\n❌ Error: {response['error']}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Demo interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}")
        
        print("\n" + "="*80)
        print("DEMO SESSION SUMMARY")
        print("="*80)
        self._show_system_stats()
        print("\n🎓 Thank you for trying the Agentic Edu-RAG System!")
        print("For more information, see the documentation or run with --help")
    
    def _show_system_stats(self):
        """Display comprehensive system statistics."""
        print("\n📊 SYSTEM STATISTICS")
        print("-" * 40)
        
        # Basic stats
        print(f"Queries Processed: {self.system_stats['queries_processed']}")
        if self.system_stats['queries_processed'] > 0:
            avg_time = self.system_stats['total_processing_time_ms'] / self.system_stats['queries_processed']
            print(f"Average Processing Time: {avg_time:.1f}ms")
        
        print(f"System Initialized: {'✅' if self.is_initialized else '❌'}")
        print(f"Content Loaded: {'✅' if self.content_loaded else '❌'}")
        
        # Component stats
        if self.orchestrator:
            orchestrator_stats = self.orchestrator.get_performance_stats()
            print(f"Orchestrator Queries: {orchestrator_stats['total_queries_processed']}")
        
        if self.classifier:
            classifier_stats = self.classifier.get_performance_stats()
            print(f"Classification Accuracy: Available after evaluation")
        
        if self.knowledge_base:
            kb_stats = self.knowledge_base.get_knowledge_base_stats()
            print(f"Knowledge Base Chunks: {kb_stats.total_chunks}")
            print(f"Cache Hit Rate: {kb_stats.cache_hit_rate:.2%}")
    
    def _show_example_queries(self):
        """Show example queries for different agent types."""
        print("\n💡 EXAMPLE QUERIES")
        print("-" * 40)
        
        print("\n🎯 Implementation Agent Examples (Forethought Phase):")
        implementation_examples = [
            "How do I implement a binary search algorithm?",
            "What's the best approach for solving this sorting problem?", 
            "How should I design a class hierarchy for this problem?",
            "What data structure should I use for fast lookups?",
            "How can I plan the architecture for this application?"
        ]
        for i, example in enumerate(implementation_examples, 1):
            print(f"  {i}. {example}")
        
        print("\n🔧 Debugging Agent Examples (Performance Phase):")
        debugging_examples = [
            "I'm getting an IndexError in my list code",
            "My binary search function returns the wrong result",
            "Why is my code throwing a TypeError?",
            "This recursive function causes a stack overflow",
            "My loop seems to run infinitely"
        ]
        for i, example in enumerate(debugging_examples, 1):
            print(f"  {i}. {example}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and statistics."""
        status = {
            "system_initialized": self.is_initialized,
            "content_loaded": self.content_loaded,
            "demo_mode": self.demo_mode,
            "performance_stats": self.system_stats.copy()
        }
        
        if self.orchestrator:
            status["orchestrator_stats"] = self.orchestrator.get_performance_stats()
        
        if self.classifier:
            status["classifier_stats"] = self.classifier.get_performance_stats()
        
        if self.knowledge_base:
            status["knowledge_base_stats"] = self.knowledge_base.get_knowledge_base_stats()
        
        return status


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Agentic Edu-RAG: Multi-Agent Educational System for Programming Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                                    Run interactive demo
  python main.py --query "How do I implement binary search?"  Process single query
  python main.py --load-content                           Load educational content
  python main.py --stats                                  Show system statistics
  
For more information, visit: https://github.com/your-repo/agentic-edu-rag
        """
    )
    
    parser.add_argument("--demo", action="store_true", 
                       help="Run interactive demo mode")
    parser.add_argument("--query", type=str, 
                       help="Process a single query and exit")
    parser.add_argument("--code", type=str, 
                       help="Code snippet to include with query")
    parser.add_argument("--error", type=str, 
                       help="Error message to include with query")
    parser.add_argument("--level", type=str, choices=["beginner", "intermediate", "advanced"],
                       default="intermediate", help="Student difficulty level")
    parser.add_argument("--load-content", action="store_true",
                       help="Load educational content and exit")
    parser.add_argument("--stats", action="store_true",
                       help="Show system statistics and exit")
    parser.add_argument("--force-reload", action="store_true",
                       help="Force reload of content even if already loaded")
    parser.add_argument("--no-init", action="store_true",
                       help="Don't auto-initialize the system")
    
    args = parser.parse_args()
    
    # Create system instance
    auto_init = not args.no_init
    system = AgenticEduRAGSystem(
        auto_initialize=auto_init,
        load_content=False,  # Load manually based on args
        enable_demo_mode=args.demo
    )
    
    try:
        # Handle different command modes
        if args.load_content:
            print("📚 Loading educational content...")
            result = system.load_educational_content(force_reload=args.force_reload)
            if result["status"] == "success":
                print(f"✅ Successfully loaded {result.get('chunks_stored', 0)} content chunks")
            else:
                print(f"❌ Content loading failed: {result.get('error', 'Unknown error')}")
            return
        
        if args.stats:
            status = system.get_system_status()
            print("\n📊 SYSTEM STATUS")
            print("="*50)
            print(json.dumps(status, indent=2, default=str))
            return
        
        if args.query:
            if not system.is_initialized:
                print("⚠️  System not initialized. Initializing...")
                init_result = system.initialize_system()
                if init_result["status"] != "success":
                    print(f"❌ Initialization failed: {init_result['error']}")
                    return
            
            print(f"🔄 Processing query: {args.query}")
            response = system.process_query(
                query=args.query,
                code_snippet=args.code,
                error_message=args.error,
                student_level=args.level
            )
            
            if response["status"] == "success":
                print(f"\n🤖 Agent: {response['agent_used']}")
                print(f"📊 SRL Phase: {response['srl_phase']}")
                print(f"🎯 Confidence: {response['classification_confidence']:.2f}")
                print("\n📝 Response:")
                print("-" * 60)
                print(response["content"])
                print("-" * 60)
            else:
                print(f"❌ Error: {response['error']}")
            return
        
        if args.demo:
            system.run_interactive_demo()
            return
        
        # Default: show help
        parser.print_help()
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
