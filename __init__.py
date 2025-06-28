"""
Agentic Edu-RAG System

A research implementation of a multi-agent RAG system for Self-Regulated Learning in CS education.
This system demonstrates how specialized agents can effectively route programming help requests 
based on SRL phases.

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AgenticEdu-RAG Research Team"
__description__ = "Multi-agent RAG system for Self-Regulated Learning in CS education"

# Make key components available at package level
from .config.settings import get_settings
from .agents.orchestrator import OrchestratorAgent
from .classification.srl_classifier import get_srl_classifier
from .rag.knowledge_base import get_knowledge_base

__all__ = [
    "get_settings",
    "OrchestratorAgent", 
    "get_srl_classifier",
    "get_knowledge_base"
]
