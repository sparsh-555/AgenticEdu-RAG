"""
Main Application Entry Point for Unified Agentic Edu-RAG System

This module serves as the primary entry point for the unified multi-agent educational RAG system.
It demonstrates the complete workflow from query reception through SRL classification
to specialized agent processing and response generation using the new unified architecture.

Key Features:
1. Unified System Initialization: Set up all components with single collection architecture
2. PDF Content Loading: Load educational content from PDF books
3. Interactive Demo: Command-line interface for testing the unified system
4. Complete Workflow: End-to-end processing with unified knowledge base
5. Performance Monitoring: Comprehensive metrics collection and reporting
6. Error Handling: Robust error management with graceful degradation

Usage Examples:
- python main.py --demo                    # Run interactive demo
- python main.py --load-pdfs               # Load PDF content into unified knowledge base
- python main.py --query "How do I implement binary search?"  # Single query
- python main.py --benchmark               # Run performance benchmarks
- python main.py --stats                   # Show unified system statistics

This implementation demonstrates the new unified architecture patterns
and provides a complete example of how to build scalable educational AI systems.
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add the current directory to Python path to enable absolute imports
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import unified system components
from agents.orchestrator import OrchestratorAgent, OrchestratorConfig
from classification.srl_classifier import get_srl_classifier, ClassificationContext, SRLPhase
from rag.knowledge_base import (
    get_knowledge_base, initialize_knowledge_base, retrieve_for_agent, 
    get_knowledge_base_stats, UnifiedRetrievalRequest, UnifiedEducationalKnowledgeBase
)
from rag.content_loader import get_content_loader, UnifiedContentLoadingConfig, ChunkingStrategy
from rag.vector_store import get_vector_store, AgentSpecialization, ContentType
from utils.logging_utils import get_logger, create_context, EventType
from utils.api_utils import get_openai_client
from config.settings import get_settings


class UnifiedAgenticEduRAGSystem:
    """
    Complete Unified Agentic Educational RAG System.
    
    This class integrates all unified system components and provides a streamlined interface
    for educational query processing using Self-Regulated Learning principles
    and the new unified multi-agent architecture.
    
    The unified system demonstrates:
    - Multi-agent orchestration with unified knowledge base
    - SRL-based query classification and intelligent routing
    - PDF-based educational content loading and retrieval
    - Unified metadata-driven content filtering
    - Comprehensive performance monitoring and evaluation
    """
    
    def __init__(self, 
                 auto_initialize: bool = True,
                 load_pdfs: bool = False,
                 enable_demo_mode: bool = False):
        """
        Initialize the unified Agentic Edu-RAG system.
        
        Args:
            auto_initialize: Whether to automatically initialize all components
            load_pdfs: Whether to load PDF content on startup
            enable_demo_mode: Whether to enable demo-specific features
        """
        self.logger = get_logger()
        self.settings = get_settings()
        
        # Unified system components (initialized later)
        self.orchestrator = None
        self.knowledge_base = None
        self.classifier = None
        self.openai_client = None
        self.vector_store = None
        self.content_loader = None
        
        # System state
        self.is_initialized = False
        self.pdfs_loaded = False
        self.demo_mode = enable_demo_mode
        
        # Performance tracking
        self.system_stats = {
            "startup_time_ms": 0,
            "queries_processed": 0,
            "total_processing_time_ms": 0,
            "initialization_time": None,
            "pdf_loading_time": None,
            "unified_architecture": True  # Flag to indicate unified architecture
        }
        
        self.logger.log_event(
            EventType.SYSTEM_START,
            "Unified Agentic Edu-RAG System created",
            extra_data={
                "auto_initialize": auto_initialize,
                "load_pdfs": load_pdfs,
                "demo_mode": enable_demo_mode,
                "architecture": "unified"
            }
        )
        
        if auto_initialize:
            self.initialize_unified_system()
            
        if load_pdfs and self.is_initialized:
            self.load_pdf_content()
    
    def initialize_unified_system(self) -> Dict[str, Any]:
        """
        Initialize all unified system components in the correct order.
        
        Returns:
            Dictionary with initialization results and timing
        """
        start_time = time.time()
        
        try:
            self.logger.log_event(
                EventType.SYSTEM_START,
                "Starting unified system initialization"
            )
            
            # Step 1: Initialize core utilities
            self.logger.log_event(EventType.SYSTEM_START, "Initializing API client")
            self.openai_client = get_openai_client()
            
            # Step 2: Initialize unified vector store
            self.logger.log_event(EventType.SYSTEM_START, "Initializing unified vector store")
            self.vector_store = get_vector_store()
            
            # Step 3: Initialize unified content loader
            self.logger.log_event(EventType.SYSTEM_START, "Initializing unified content loader")
            loader_config = UnifiedContentLoadingConfig(
                chunking_strategy=ChunkingStrategy.PDF_AWARE,
                max_chunk_size=1000,
                min_chunk_size=200
            )
            self.content_loader = get_content_loader(loader_config)
            
            # Step 4: Initialize unified knowledge base
            self.logger.log_event(EventType.SYSTEM_START, "Initializing unified knowledge base")
            self.knowledge_base = get_knowledge_base()
            
            # Step 5: Initialize SRL classifier
            self.logger.log_event(EventType.SYSTEM_START, "Initializing SRL classifier")
            self.classifier = get_srl_classifier()
            
            # Step 6: Initialize orchestrator with unified workflow
            self.logger.log_event(EventType.SYSTEM_START, "Initializing unified orchestrator")
            orchestrator_config = OrchestratorConfig(
                enable_quality_checks=True,
                enable_conversation_memory=True,
                collect_educational_metrics=True
            )
            self.orchestrator = OrchestratorAgent(config=orchestrator_config)
            
            # Unified system initialization complete
            initialization_time = (time.time() - start_time) * 1000
            self.system_stats["startup_time_ms"] = initialization_time
            self.system_stats["initialization_time"] = time.time()
            self.is_initialized = True
            
            self.logger.log_event(
                EventType.SYSTEM_START,
                f"Unified system initialization completed in {initialization_time:.1f}ms",
                extra_data={"initialization_time_ms": initialization_time}
            )
            
            return {
                "status": "success",
                "initialization_time_ms": initialization_time,
                "components_initialized": [
                    "openai_client",
                    "unified_vector_store",
                    "unified_content_loader", 
                    "unified_knowledge_base",
                    "srl_classifier",
                    "unified_orchestrator"
                ],
                "architecture": "unified"
            }
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Unified system initialization failed: {str(e)}",
                level="ERROR"
            )
            return {
                "status": "error",
                "error": str(e),
                "initialization_time_ms": (time.time() - start_time) * 1000
            }
    
    def load_pdf_content(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load educational content from PDF books into the unified knowledge base.
        
        Args:
            force_reload: Whether to reload content even if already loaded
            
        Returns:
            Dictionary with PDF loading results
        """
        if not self.is_initialized:
            return {"status": "error", "error": "Unified system not initialized"}
        
        start_time = time.time()
        
        try:
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                "Starting PDF content loading into unified knowledge base",
                extra_data={"force_reload": force_reload}
            )
            
            # Load PDF content using unified knowledge base
            loading_results = initialize_knowledge_base(force_reload=force_reload)
            
            loading_time = (time.time() - start_time) * 1000
            self.system_stats["pdf_loading_time"] = loading_time
            
            if loading_results["status"] == "success":
                self.pdfs_loaded = True
                
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                f"PDF loading completed: {loading_results.get('content_statistics', {}).get('total_pieces', 0)} content pieces loaded",
                extra_data={
                    "loading_time_ms": loading_time,
                    "results": loading_results
                }
            )
            
            return loading_results
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"PDF content loading failed: {str(e)}",
                level="ERROR"
            )
            return {"status": "error", "error": str(e)}
    
    def process_query(self, 
                     query: str,
                     code_snippet: Optional[str] = None,
                     error_message: Optional[str] = None,
                     student_level: Optional[str] = "intermediate",
                     session_id: Optional[str] = None,
                     user_id: Optional[str] = None,
                     agent_preference: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a student query through the unified multi-agent workflow.
        
        Args:
            query: Student's programming question
            code_snippet: Optional code context
            error_message: Optional error message
            student_level: Student proficiency level
            session_id: Session identifier for conversation tracking
            user_id: User identifier
            agent_preference: Optional preferred agent specialization
            
        Returns:
            Comprehensive response with educational content and unified metadata
        """
        if not self.is_initialized:
            return {
                "status": "error", 
                "error": "Unified system not initialized. Please run initialize_unified_system() first."
            }
        
        start_time = time.time()
        
        try:
            # Process query through unified orchestrator
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
            
            # Create comprehensive unified response
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
                    "conversation_turn": orchestrator_response.conversation_turn,
                    "architecture": "unified"
                },
                "unified_metadata": {
                    "knowledge_base_used": "unified_collection",
                    "pdf_content_available": self.pdfs_loaded,
                    "metadata_filtering_applied": True
                }
            }
            
            self.logger.log_event(
                EventType.QUERY_COMPLETED,
                f"Query processed through unified system in {processing_time:.1f}ms",
                extra_data={
                    "processing_time_ms": processing_time,
                    "agent_used": orchestrator_response.agent_used,
                    "srl_phase": orchestrator_response.srl_phase,
                    "architecture": "unified"
                }
            )
            
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Unified query processing failed: {str(e)}",
                level="ERROR",
                extra_data={"processing_time_ms": processing_time}
            )
            
            return {
                "status": "error",
                "error": str(e),
                "processing_time_ms": processing_time,
                "architecture": "unified"
            }
    
    def test_unified_retrieval(self, 
                              query: str,
                              agent_type: str = "implementation",
                              srl_phase: str = "forethought") -> Dict[str, Any]:
        """
        Test the unified retrieval system directly.
        
        Args:
            query: Test query
            agent_type: Agent specialization to test
            srl_phase: SRL phase for context
            
        Returns:
            Direct retrieval results from unified system
        """
        if not self.is_initialized:
            return {"status": "error", "error": "Unified system not initialized"}
        
        try:
            # Test unified retrieval
            response = retrieve_for_agent(
                query=query,
                agent_type=agent_type,
                srl_phase=srl_phase,
                context={
                    "student_level": "intermediate",
                    "prefer_code_examples": True,
                    "max_results": 5
                }
            )
            
            return {
                "status": "success",
                "results_count": len(response.results),
                "average_relevance": response.average_relevance_score,
                "average_metadata_match": response.average_metadata_match_score,
                "content_types_found": response.content_types_found,
                "concepts_covered": response.concepts_covered,
                "retrieval_strategy": response.retrieval_strategy,
                "served_from_cache": response.served_from_cache,
                "retrieval_time_ms": response.retrieval_time_ms
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_interactive_demo(self):
        """
        Run an interactive command-line demo of the unified system.
        
        This provides a user-friendly interface for testing the complete
        unified multi-agent workflow with various types of programming questions.
        """
        print("\n" + "="*80)
        print("🎓 UNIFIED AGENTIC EDU-RAG SYSTEM - INTERACTIVE DEMO")
        print("="*80)
        print("\nWelcome to the Unified Self-Regulated Learning Multi-Agent Educational System!")
        print("This system uses a unified knowledge base with intelligent metadata filtering.")
        print("\nUnified Architecture Features:")
        print("  📚 Single Collection: All content in one unified ChromaDB collection")
        print("  🔍 Smart Filtering: Metadata-based content retrieval")
        print("  📖 PDF Content: Load educational books directly from PDF files")
        print("  🎯 Agent Specialization: Intelligent routing with unified storage")
        print("\nAgent Specializations:")
        print("  🎯 Implementation Agent: Planning, strategy, algorithm design (Forethought Phase)")
        print("  🔧 Debugging Agent: Error analysis, troubleshooting (Performance Phase)")
        print("  🤖 Orchestrator: Intelligent routing with unified knowledge base")
        
        if not self.is_initialized:
            print("\n⚠️  Unified system not initialized. Initializing now...")
            init_result = self.initialize_unified_system()
            if init_result["status"] != "success":
                print(f"❌ Initialization failed: {init_result['error']}")
                return
            print("✅ Unified system initialized successfully!")
        
        if not self.pdfs_loaded:
            print("\n📚 Loading PDF educational content into unified knowledge base...")
            load_result = self.load_pdf_content()
            if load_result["status"] == "success":
                content_stats = load_result.get('content_statistics', {})
                print(f"✅ Loaded {content_stats.get('total_pieces', 0)} content pieces from PDFs")
                print(f"   By type: {content_stats.get('by_type', {})}")
                print(f"   By agent: {content_stats.get('by_agent', {})}")
            else:
                print(f"⚠️  PDF content loading: {load_result.get('error', 'No PDFs found')}")
                print("Place your PDF books in data/pdfs/ directory for full functionality")
        
        # Check unified system status
        unified_stats = get_knowledge_base_stats()
        print(f"\n📊 Unified Knowledge Base Status:")
        print(f"   Total content pieces: {unified_stats.total_content_pieces}")
        print(f"   Cache hit rate: {unified_stats.cache_hit_rate:.1%}")
        
        print("\n" + "-"*80)
        print("UNIFIED DEMO COMMANDS:")
        print("  Type your programming question and press Enter")
        print("  Commands: 'stats' (system statistics), 'test' (test retrieval), 'examples', 'quit'")
        print("-"*80)
        
        session_id = f"unified_demo_{int(time.time())}"
        
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
                    self._show_unified_system_stats()
                    continue
                elif user_input.lower() == 'test':
                    self._run_unified_retrieval_test()
                    continue
                elif user_input.lower() == 'examples':
                    self._show_unified_example_queries()
                    continue
                
                # Process the query through unified system
                print("\n🔄 Processing through unified system...")
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
                    print(f"🏗️  Architecture: {response['unified_metadata']['knowledge_base_used']}")
                    
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
        print("UNIFIED DEMO SESSION SUMMARY")
        print("="*80)
        self._show_unified_system_stats()
        print("\n🎓 Thank you for trying the Unified Agentic Edu-RAG System!")
        print("The unified architecture provides better performance and simpler maintenance.")
    
    def _show_unified_system_stats(self):
        """Display comprehensive unified system statistics."""
        print("\n📊 UNIFIED SYSTEM STATISTICS")
        print("-" * 50)
        
        # Basic stats
        print(f"Architecture: Unified Collection")
        print(f"Queries Processed: {self.system_stats['queries_processed']}")
        if self.system_stats['queries_processed'] > 0:
            avg_time = self.system_stats['total_processing_time_ms'] / self.system_stats['queries_processed']
            print(f"Average Processing Time: {avg_time:.1f}ms")
        
        print(f"System Initialized: {'✅' if self.is_initialized else '❌'}")
        print(f"PDFs Loaded: {'✅' if self.pdfs_loaded else '❌'}")
        
        # Unified knowledge base stats
        if self.knowledge_base:
            kb_stats = get_knowledge_base_stats()
            print(f"\n📚 Unified Knowledge Base:")
            print(f"   Total Content Pieces: {kb_stats.total_content_pieces}")
            print(f"   By Content Type: {kb_stats.content_by_type}")
            print(f"   By Agent Specialization: {kb_stats.content_by_agent}")
            print(f"   By Difficulty Level: {kb_stats.content_by_difficulty}")
            print(f"   Cache Hit Rate: {kb_stats.cache_hit_rate:.2%}")
            print(f"   Average Query Time: {kb_stats.average_query_time_ms:.1f}ms")
        
        # Component stats
        if self.orchestrator:
            orchestrator_stats = self.orchestrator.get_performance_stats()
            print(f"\n🤖 Orchestrator:")
            print(f"   Total Queries: {orchestrator_stats['total_queries_processed']}")
        
        if self.vector_store:
            vs_stats = self.vector_store.get_vector_store_stats()
            print(f"\n📦 Unified Vector Store:")
            print(f"   Total Documents: {vs_stats.total_documents}")
            print(f"   Total Queries: {vs_stats.total_queries}")
            print(f"   Average Query Time: {vs_stats.average_query_time_ms:.1f}ms")
    
    def _run_unified_retrieval_test(self):
        """Run a test of the unified retrieval system."""
        print("\n🔬 UNIFIED RETRIEVAL TEST")
        print("-" * 40)
        
        test_queries = [
            ("How to implement binary search?", "implementation", "forethought"),
            ("Fix this segmentation fault error", "debugging", "performance"),
            ("Explain sorting algorithms", "implementation", "forethought")
        ]
        
        for query, agent_type, srl_phase in test_queries:
            print(f"\nTesting: {query}")
            print(f"Agent: {agent_type}, Phase: {srl_phase}")
            
            result = self.test_unified_retrieval(query, agent_type, srl_phase)
            
            if result["status"] == "success":
                print(f"  ✅ Results: {result['results_count']}")
                print(f"  📊 Relevance: {result['average_relevance']:.3f}")
                print(f"  🎯 Metadata Match: {result['average_metadata_match']:.3f}")
                print(f"  ⏱️  Time: {result['retrieval_time_ms']:.1f}ms")
                print(f"  📚 Types: {result['content_types_found']}")
            else:
                print(f"  ❌ Error: {result['error']}")
    
    def _show_unified_example_queries(self):
        """Show example queries for the unified system."""
        print("\n💡 UNIFIED SYSTEM EXAMPLE QUERIES")
        print("-" * 50)
        
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
        
        print("\n🏗️ Unified System Benefits:")
        print("  • Single collection with intelligent metadata filtering")
        print("  • PDF-based content loading from educational books")  
        print("  • Better performance with unified caching")
        print("  • Simplified architecture and maintenance")
        print("  • Enhanced content discovery across agent types")
    
    def get_unified_system_status(self) -> Dict[str, Any]:
        """Get comprehensive unified system status and statistics."""
        status = {
            "architecture": "unified",
            "system_initialized": self.is_initialized,
            "pdfs_loaded": self.pdfs_loaded,
            "demo_mode": self.demo_mode,
            "performance_stats": self.system_stats.copy()
        }
        
        if self.orchestrator:
            status["orchestrator_stats"] = self.orchestrator.get_performance_stats()
        
        if self.classifier:
            status["classifier_stats"] = self.classifier.get_performance_stats()
        
        if self.knowledge_base:
            status["unified_knowledge_base_stats"] = get_knowledge_base_stats().dict()
        
        if self.vector_store:
            status["unified_vector_store_stats"] = self.vector_store.get_vector_store_stats().dict()
        
        return status


def main():
    """Main entry point with unified system command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Unified Agentic Edu-RAG: Multi-Agent Educational System with Unified Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                                    Run interactive demo with unified system
  python main.py --query "How do I implement binary search?"  Process single query through unified system
  python main.py --load-pdfs                              Load PDF content into unified knowledge base
  python main.py --stats                                  Show unified system statistics
  python main.py --test-retrieval                         Test unified retrieval system
  
Unified Architecture Features:
  • Single ChromaDB collection with metadata filtering
  • PDF-based educational content loading
  • Intelligent agent routing with unified storage
  • Enhanced performance and simplified maintenance
  
For more information, visit: https://github.com/your-repo/agentic-edu-rag
        """
    )
    
    parser.add_argument("--demo", action="store_true", 
                       help="Run interactive demo mode with unified system")
    parser.add_argument("--query", type=str, 
                       help="Process a single query through unified system and exit")
    parser.add_argument("--code", type=str, 
                       help="Code snippet to include with query")
    parser.add_argument("--error", type=str, 
                       help="Error message to include with query")
    parser.add_argument("--level", type=str, choices=["beginner", "intermediate", "advanced"],
                       default="intermediate", help="Student difficulty level")
    parser.add_argument("--agent", type=str, choices=["implementation", "debugging"],
                       help="Preferred agent specialization")
    parser.add_argument("--load-pdfs", action="store_true",
                       help="Load PDF content into unified knowledge base and exit")
    parser.add_argument("--stats", action="store_true",
                       help="Show unified system statistics and exit")
    parser.add_argument("--test-retrieval", action="store_true",
                       help="Test unified retrieval system and exit")
    parser.add_argument("--force-reload", action="store_true",
                       help="Force reload of PDF content even if already loaded")
    parser.add_argument("--no-init", action="store_true",
                       help="Don't auto-initialize the unified system")
    
    args = parser.parse_args()
    
    # Create unified system instance
    auto_init = not args.no_init
    system = UnifiedAgenticEduRAGSystem(
        auto_initialize=auto_init,
        load_pdfs=False,  # Load manually based on args
        enable_demo_mode=args.demo
    )
    
    try:
        # Handle different command modes
        if args.load_pdfs:
            print("📚 Loading PDF content into unified knowledge base...")
            result = system.load_pdf_content(force_reload=args.force_reload)
            if result["status"] == "success":
                content_stats = result.get('content_statistics', {})
                print(f"✅ Successfully loaded {content_stats.get('total_pieces', 0)} content pieces")
                print(f"   Files processed: {result.get('loading_details', {}).get('files_processed', 0)}")
                print(f"   By content type: {content_stats.get('by_type', {})}")
                print(f"   By agent specialization: {content_stats.get('by_agent', {})}")
            else:
                print(f"❌ PDF loading failed: {result.get('error', 'Unknown error')}")
                print("Make sure to place your PDF books in the data/pdfs/ directory")
            return
        
        if args.stats:
            status = system.get_unified_system_status()
            print("\n📊 UNIFIED SYSTEM STATUS")
            print("="*60)
            print(json.dumps(status, indent=2, default=str))
            return
        
        if args.test_retrieval:
            print("🔬 Testing unified retrieval system...")
            test_result = system.test_unified_retrieval(
                "How to implement quicksort algorithm?",
                "implementation", 
                "forethought"
            )
            print(f"Test result: {json.dumps(test_result, indent=2)}")
            return
        
        if args.query:
            if not system.is_initialized:
                print("⚠️  Unified system not initialized. Initializing...")
                init_result = system.initialize_unified_system()
                if init_result["status"] != "success":
                    print(f"❌ Initialization failed: {init_result['error']}")
                    return
            
            print(f"🔄 Processing query through unified system: {args.query}")
            response = system.process_query(
                query=args.query,
                code_snippet=args.code,
                error_message=args.error,
                student_level=args.level,
                agent_preference=args.agent
            )
            
            if response["status"] == "success":
                print(f"\n🤖 Agent: {response['agent_used']}")
                print(f"📊 SRL Phase: {response['srl_phase']}")
                print(f"🎯 Confidence: {response['classification_confidence']:.2f}")
                print(f"🏗️  Architecture: {response['unified_metadata']['knowledge_base_used']}")
                print("\n📝 Response:")
                print("-" * 60)
                print(response["content"])
                print("-" * 60)
                
                if response.get("suggested_follow_up"):
                    print("\n💡 Suggested follow-up:")
                    for suggestion in response["suggested_follow_up"][:3]:
                        print(f"  • {suggestion}")
            else:
                print(f"❌ Error: {response['error']}")
            return
        
        if args.demo:
            system.run_interactive_demo()
            return
        
        # Default: show help and unified system info
        parser.print_help()
        print("\n" + "="*60)
        print("🏗️  UNIFIED ARCHITECTURE INFORMATION")
        print("="*60)
        print("This system uses a unified architecture with:")
        print("  📚 Single ChromaDB collection for all content")
        print("  🔍 Intelligent metadata-based filtering")
        print("  📖 PDF-based educational content loading")
        print("  🎯 Smart agent specialization routing")
        print("  📊 Enhanced performance monitoring")
        print("\nTo get started:")
        print("  1. Place your PDF books in data/pdfs/")
        print("  2. Run: python main.py --load-pdfs")
        print("  3. Run: python main.py --demo")
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
