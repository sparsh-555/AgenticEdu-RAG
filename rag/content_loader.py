"""
Educational Content Loader for RAG System

This module handles loading, processing, and preparing educational content for
the vector store. It supports various content formats, intelligent chunking
strategies, and automated metadata extraction optimized for programming education.

Key Features:
1. Multi-Format Support: Text files, markdown, code examples, and structured content
2. Intelligent Chunking: Context-aware chunking that preserves educational coherence
3. Metadata Extraction: Automatic detection of programming concepts and difficulty levels
4. Agent-Specific Organization: Content classification for specialized agent collections
5. Incremental Loading: Support for updating and adding new content efficiently
6. Quality Validation: Content quality checks and educational appropriateness scoring

Content Processing Pipeline:
1. File Discovery: Scan educational content directories
2. Content Parsing: Extract and clean text content
3. Concept Detection: Identify programming concepts and topics
4. Chunking Strategy: Create coherent chunks for vector storage
5. Metadata Generation: Create rich metadata for educational research
6. Quality Validation: Ensure content meets educational standards
"""

import os
import re
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple, Set, Iterator
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import threading

from pydantic import BaseModel, Field

from .vector_store import (
    ContentMetadata, ContentType, AgentSpecialization, 
    EducationalVectorStore, get_vector_store
)
from ..config.settings import get_settings
from ..utils.logging_utils import get_logger, LogContext, EventType, create_context


class ContentFormat(Enum):
    """Supported content formats for processing."""
    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"
    JSON = "json"
    MIXED = "mixed"


class ChunkingStrategy(Enum):
    """Different strategies for content chunking."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    PARAGRAPH = "paragraph"
    CODE_AWARE = "code_aware"
    EDUCATIONAL_UNIT = "educational_unit"


@dataclass
class ContentChunk:
    """A chunk of educational content ready for vector storage."""
    chunk_id: str
    content: str
    metadata: ContentMetadata
    
    # Chunk-specific metadata
    chunk_index: int = 0
    total_chunks: int = 1
    chunk_type: str = "content"  # content, code_example, explanation
    
    # Quality metrics
    readability_score: float = 0.5
    educational_value: float = 0.5
    concept_density: float = 0.5


@dataclass
class ContentLoadingConfig:
    """Configuration for content loading process."""
    max_chunk_size: int = 1000
    min_chunk_size: int = 200
    chunk_overlap: int = 100
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    
    # Quality filters
    min_readability_score: float = 0.3
    min_educational_value: float = 0.3
    
    # Processing options
    extract_code_examples: bool = True
    detect_difficulty_automatically: bool = True
    preserve_formatting: bool = True
    
    # Performance options
    batch_size: int = 50
    parallel_processing: bool = True


class ContentProcessor:
    """Processes raw content into educational chunks."""
    
    def __init__(self, config: ContentLoadingConfig):
        """Initialize content processor with configuration."""
        self.config = config
        self.logger = get_logger()
        
        # Programming concept patterns
        self.concept_patterns = self._initialize_concept_patterns()
        
        # Code detection patterns
        self.code_patterns = self._initialize_code_patterns()
        
        # Difficulty indicators
        self.difficulty_indicators = self._initialize_difficulty_indicators()
        
        # Quality assessment patterns
        self.quality_patterns = self._initialize_quality_patterns()
    
    def process_content(self, 
                       content: str, 
                       source_file: str,
                       base_metadata: ContentMetadata) -> List[ContentChunk]:
        """
        Process content into educational chunks.
        
        Args:
            content: Raw content text
            source_file: Source file path
            base_metadata: Base metadata for the content
            
        Returns:
            List of processed content chunks
        """
        try:
            # Clean and preprocess content
            cleaned_content = self._clean_content(content)
            
            # Detect content format
            content_format = self._detect_content_format(cleaned_content)
            
            # Extract programming concepts
            concepts = self._extract_programming_concepts(cleaned_content)
            
            # Detect difficulty level if not set
            if self.config.detect_difficulty_automatically:
                difficulty = self._detect_difficulty_level(cleaned_content)
                base_metadata.difficulty_level = difficulty
            
            # Update metadata with extracted information
            base_metadata.programming_concepts = concepts
            base_metadata.content_length = len(cleaned_content)
            base_metadata.has_code_examples = self._has_code_examples(cleaned_content)
            base_metadata.source_file = source_file
            
            # Create chunks based on strategy
            chunks = self._create_chunks(cleaned_content, base_metadata, content_format)
            
            # Validate and score chunks
            validated_chunks = []
            for chunk in chunks:
                if self._validate_chunk_quality(chunk):
                    validated_chunks.append(chunk)
            
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                f"Processed content: {len(validated_chunks)} chunks from {source_file}",
                extra_data={
                    "source_file": source_file,
                    "original_length": len(content),
                    "chunks_created": len(chunks),
                    "chunks_validated": len(validated_chunks),
                    "concepts_detected": len(concepts)
                }
            )
            
            return validated_chunks
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Content processing failed for {source_file}: {str(e)}",
                level="ERROR"
            )
            return []
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content text."""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        
        # Remove common file artifacts
        content = re.sub(r'^---.*?---\s*', '', content, flags=re.DOTALL)  # YAML frontmatter
        content = content.strip()
        
        return content
    
    def _detect_content_format(self, content: str) -> ContentFormat:
        """Detect the format of content."""
        content_lower = content.lower()
        
        # Check for code indicators
        code_indicators = ['def ', 'class ', 'function', '```', 'import ', 'from ']
        if any(indicator in content_lower for indicator in code_indicators):
            return ContentFormat.CODE
        
        # Check for markdown indicators
        markdown_indicators = ['# ', '## ', '- ', '* ', '![', '](']
        if any(indicator in content for indicator in markdown_indicators):
            return ContentFormat.MARKDOWN
        
        # Check for JSON
        try:
            json.loads(content)
            return ContentFormat.JSON
        except:
            pass
        
        # Check for mixed content
        if '```' in content or 'example:' in content_lower:
            return ContentFormat.MIXED
        
        return ContentFormat.TEXT
    
    def _extract_programming_concepts(self, content: str) -> List[str]:
        """Extract programming concepts from content."""
        concepts = set()
        content_lower = content.lower()
        
        # Check for concepts in our patterns
        for concept, patterns in self.concept_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                concepts.add(concept)
        
        return list(concepts)
    
    def _detect_difficulty_level(self, content: str) -> str:
        """Automatically detect difficulty level of content."""
        content_lower = content.lower()
        difficulty_score = 0
        
        # Check for difficulty indicators
        for level, indicators in self.difficulty_indicators.items():
            level_score = sum(1 for indicator in indicators if indicator in content_lower)
            
            if level == "beginner":
                difficulty_score -= level_score
            elif level == "advanced":
                difficulty_score += level_score * 2
            elif level == "intermediate":
                difficulty_score += level_score
        
        # Determine difficulty based on score
        if difficulty_score <= -2:
            return "beginner"
        elif difficulty_score >= 3:
            return "advanced"
        else:
            return "intermediate"
    
    def _has_code_examples(self, content: str) -> bool:
        """Check if content contains code examples."""
        code_indicators = [
            '```', 'def ', 'class ', 'function', 'import ',
            'from ', '{', '}', 'for (', 'while (', 'if ('
        ]
        
        return any(indicator in content for indicator in code_indicators)
    
    def _create_chunks(self, 
                      content: str, 
                      base_metadata: ContentMetadata,
                      content_format: ContentFormat) -> List[ContentChunk]:
        """Create chunks based on configured strategy."""
        if self.config.chunking_strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunking(content, base_metadata)
        elif self.config.chunking_strategy == ChunkingStrategy.CODE_AWARE:
            return self._code_aware_chunking(content, base_metadata)
        elif self.config.chunking_strategy == ChunkingStrategy.EDUCATIONAL_UNIT:
            return self._educational_unit_chunking(content, base_metadata)
        elif self.config.chunking_strategy == ChunkingStrategy.PARAGRAPH:
            return self._paragraph_chunking(content, base_metadata)
        else:
            return self._fixed_size_chunking(content, base_metadata)
    
    def _semantic_chunking(self, content: str, base_metadata: ContentMetadata) -> List[ContentChunk]:
        """Create semantically coherent chunks."""
        chunks = []
        
        # Split by major sections (headers, double newlines, etc.)
        sections = re.split(r'\n\s*\n|#{1,3}\s+', content)
        sections = [s.strip() for s in sections if s.strip()]
        
        current_chunk = ""
        chunk_index = 0
        
        for section in sections:
            # Check if adding this section would exceed max size
            if len(current_chunk) + len(section) > self.config.max_chunk_size and current_chunk:
                # Create chunk from current content
                chunk = self._create_chunk(
                    current_chunk, base_metadata, chunk_index, len(sections)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + section
                chunk_index += 1
            else:
                current_chunk += "\n\n" + section if current_chunk else section
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk(
                current_chunk, base_metadata, chunk_index, len(sections)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _code_aware_chunking(self, content: str, base_metadata: ContentMetadata) -> List[ContentChunk]:
        """Create chunks that respect code boundaries."""
        chunks = []
        
        # Split content preserving code blocks
        parts = re.split(r'(```.*?```)', content, flags=re.DOTALL)
        
        current_chunk = ""
        chunk_index = 0
        
        for part in parts:
            is_code_block = part.startswith('```')
            
            if is_code_block:
                # Code blocks should stay together when possible
                if len(current_chunk) + len(part) > self.config.max_chunk_size and current_chunk:
                    # Create chunk without code block
                    chunk = self._create_chunk(
                        current_chunk, base_metadata, chunk_index, len(parts)
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk with code block
                    current_chunk = part
                    chunk_index += 1
                else:
                    current_chunk += part
            else:
                # Regular text can be split normally
                if len(current_chunk) + len(part) > self.config.max_chunk_size and current_chunk:
                    chunk = self._create_chunk(
                        current_chunk, base_metadata, chunk_index, len(parts)
                    )
                    chunks.append(chunk)
                    
                    current_chunk = part
                    chunk_index += 1
                else:
                    current_chunk += part
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk(
                current_chunk, base_metadata, chunk_index, len(parts)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _educational_unit_chunking(self, content: str, base_metadata: ContentMetadata) -> List[ContentChunk]:
        """Create chunks based on educational units (concepts, examples, exercises)."""
        chunks = []
        
        # Split by educational markers
        unit_markers = [
            r'Example\s*\d*:?',
            r'Exercise\s*\d*:?',
            r'Concept\s*\d*:?',
            r'Step\s*\d+:?',
            r'Note:?',
            r'Important:?'
        ]
        
        pattern = '(' + '|'.join(unit_markers) + ')'
        parts = re.split(pattern, content, flags=re.IGNORECASE)
        
        current_chunk = ""
        chunk_index = 0
        
        for i, part in enumerate(parts):
            if len(current_chunk) + len(part) > self.config.max_chunk_size and current_chunk:
                chunk = self._create_chunk(
                    current_chunk, base_metadata, chunk_index, len(parts)
                )
                chunks.append(chunk)
                
                current_chunk = part
                chunk_index += 1
            else:
                current_chunk += part
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk(
                current_chunk, base_metadata, chunk_index, len(parts)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _paragraph_chunking(self, content: str, base_metadata: ContentMetadata) -> List[ContentChunk]:
        """Create chunks based on paragraph boundaries."""
        paragraphs = content.split('\n\n')
        chunks = []
        
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > self.config.max_chunk_size and current_chunk:
                chunk = self._create_chunk(
                    current_chunk, base_metadata, chunk_index, len(paragraphs)
                )
                chunks.append(chunk)
                
                current_chunk = paragraph
                chunk_index += 1
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk(
                current_chunk, base_metadata, chunk_index, len(paragraphs)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _fixed_size_chunking(self, content: str, base_metadata: ContentMetadata) -> List[ContentChunk]:
        """Create fixed-size chunks with overlap."""
        chunks = []
        chunk_index = 0
        
        start = 0
        while start < len(content):
            end = start + self.config.max_chunk_size
            
            # Try to break at word boundary
            if end < len(content):
                # Look for last space within reasonable distance
                space_pos = content.rfind(' ', start, end)
                if space_pos > start + self.config.max_chunk_size * 0.8:
                    end = space_pos
            
            chunk_content = content[start:end].strip()
            
            if len(chunk_content) >= self.config.min_chunk_size:
                chunk = self._create_chunk(
                    chunk_content, base_metadata, chunk_index, 
                    (len(content) // self.config.max_chunk_size) + 1
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + self.config.max_chunk_size - self.config.chunk_overlap, end)
        
        return chunks
    
    def _create_chunk(self, 
                     content: str, 
                     base_metadata: ContentMetadata,
                     chunk_index: int,
                     total_chunks: int) -> ContentChunk:
        """Create a content chunk with metadata."""
        # Generate unique chunk ID
        chunk_id = f"{base_metadata.content_id}_chunk_{chunk_index}"
        
        # Create chunk metadata (copy base metadata)
        chunk_metadata = ContentMetadata(
            content_id=chunk_id,
            content_type=base_metadata.content_type,
            agent_specialization=base_metadata.agent_specialization,
            programming_concepts=base_metadata.programming_concepts.copy(),
            difficulty_level=base_metadata.difficulty_level,
            programming_language=base_metadata.programming_language,
            topic_tags=base_metadata.topic_tags.copy(),
            content_length=len(content),
            has_code_examples=self._has_code_examples(content),
            has_error_examples=self._has_error_examples(content),
            source_file=base_metadata.source_file,
            created_timestamp=base_metadata.created_timestamp,
            updated_timestamp=time.time()
        )
        
        # Assess chunk quality
        readability = self._assess_readability(content)
        educational_value = self._assess_educational_value(content)
        concept_density = self._assess_concept_density(content, chunk_metadata.programming_concepts)
        
        return ContentChunk(
            chunk_id=chunk_id,
            content=content,
            metadata=chunk_metadata,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            chunk_type=self._determine_chunk_type(content),
            readability_score=readability,
            educational_value=educational_value,
            concept_density=concept_density
        )
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text for chunk continuity."""
        if len(text) < self.config.chunk_overlap:
            return text
        
        # Try to get last complete sentence
        sentences = re.split(r'[.!?]+', text)
        overlap_text = ""
        
        for sentence in reversed(sentences):
            sentence = sentence.strip()
            if sentence and len(overlap_text) + len(sentence) < self.config.chunk_overlap:
                overlap_text = sentence + ". " + overlap_text
            else:
                break
        
        return overlap_text.strip()
    
    def _has_error_examples(self, content: str) -> bool:
        """Check if content contains error examples."""
        error_indicators = [
            'error:', 'exception:', 'traceback:', 'syntaxerror',
            'typeerror', 'indexerror', 'nameerror', 'valueerror'
        ]
        
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in error_indicators)
    
    def _determine_chunk_type(self, content: str) -> str:
        """Determine the type of chunk."""
        content_lower = content.lower()
        
        if self._has_code_examples(content):
            return "code_example"
        elif any(word in content_lower for word in ['explain', 'concept', 'understand']):
            return "explanation"
        elif any(word in content_lower for word in ['exercise', 'practice', 'try']):
            return "exercise"
        else:
            return "content"
    
    def _assess_readability(self, content: str) -> float:
        """Assess readability of content."""
        # Simple readability assessment
        sentences = re.split(r'[.!?]+', content)
        words = content.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Shorter sentences are generally more readable
        readability = max(0.0, 1.0 - (avg_sentence_length - 10) / 20)
        
        return min(1.0, readability)
    
    def _assess_educational_value(self, content: str) -> float:
        """Assess educational value of content."""
        value_score = 0.0
        content_lower = content.lower()
        
        # Educational indicators
        educational_words = [
            'learn', 'understand', 'concept', 'example', 'practice',
            'explain', 'demonstrate', 'show', 'teach', 'guide'
        ]
        
        for word in educational_words:
            if word in content_lower:
                value_score += 0.1
        
        # Code examples boost educational value
        if self._has_code_examples(content):
            value_score += 0.2
        
        # Error examples also valuable for learning
        if self._has_error_examples(content):
            value_score += 0.15
        
        return min(1.0, value_score)
    
    def _assess_concept_density(self, content: str, concepts: List[str]) -> float:
        """Assess density of programming concepts in content."""
        if not concepts:
            return 0.0
        
        words = content.split()
        if not words:
            return 0.0
        
        # Calculate concept density
        concept_density = len(concepts) / max(len(words) / 100, 1)  # Concepts per 100 words
        
        return min(1.0, concept_density)
    
    def _validate_chunk_quality(self, chunk: ContentChunk) -> bool:
        """Validate chunk meets quality thresholds."""
        if chunk.readability_score < self.config.min_readability_score:
            return False
        
        if chunk.educational_value < self.config.min_educational_value:
            return False
        
        if len(chunk.content) < self.config.min_chunk_size:
            return False
        
        return True
    
    def _initialize_concept_patterns(self) -> Dict[str, List[str]]:
        """Initialize programming concept detection patterns."""
        return {
            "algorithms": [
                "algorithm", "sorting", "searching", "traversal", "recursion",
                "iteration", "complexity", "big o", "efficiency"
            ],
            "data_structures": [
                "array", "list", "stack", "queue", "tree", "graph", "hash",
                "linked list", "binary tree", "heap", "data structure"
            ],
            "object_oriented": [
                "class", "object", "inheritance", "polymorphism", "encapsulation",
                "abstraction", "constructor", "method", "attribute"
            ],
            "control_flow": [
                "if", "else", "while", "for", "loop", "condition", "branch",
                "control flow", "iteration"
            ],
            "functions": [
                "function", "method", "parameter", "argument", "return",
                "scope", "local", "global"
            ],
            "error_handling": [
                "error", "exception", "try", "catch", "finally", "throw",
                "debugging", "troubleshooting"
            ],
            "testing": [
                "test", "testing", "unit test", "assertion", "mock",
                "validation", "verification"
            ]
        }
    
    def _initialize_code_patterns(self) -> List[str]:
        """Initialize code detection patterns."""
        return [
            r'def\s+\w+\s*\(',
            r'class\s+\w+\s*\(',
            r'function\s+\w+\s*\(',
            r'import\s+\w+',
            r'from\s+\w+\s+import',
            r'```[\w]*\n',
            r'^\s*#.*$',
            r'{\s*$',
            r'}\s*$'
        ]
    
    def _initialize_difficulty_indicators(self) -> Dict[str, List[str]]:
        """Initialize difficulty level indicators."""
        return {
            "beginner": [
                "basic", "simple", "introduction", "getting started",
                "first", "begin", "easy", "fundamental"
            ],
            "intermediate": [
                "intermediate", "moderate", "practice", "apply",
                "combine", "build", "extend"
            ],
            "advanced": [
                "advanced", "complex", "optimization", "performance",
                "scalability", "architecture", "design pattern",
                "algorithm analysis", "complexity theory"
            ]
        }
    
    def _initialize_quality_patterns(self) -> Dict[str, List[str]]:
        """Initialize quality assessment patterns."""
        return {
            "high_quality": [
                "example", "demonstrates", "explains", "shows how",
                "step by step", "clearly", "understand"
            ],
            "low_quality": [
                "todo", "fixme", "placeholder", "incomplete",
                "broken", "not working"
            ]
        }


class EducationalContentLoader:
    """
    Main content loader for educational RAG system.
    
    This class orchestrates the loading, processing, and storage of educational
    content from various sources into the vector store, with support for
    different content types and agent specializations.
    """
    
    def __init__(self, config: Optional[ContentLoadingConfig] = None):
        """Initialize the content loader."""
        self.config = config or ContentLoadingConfig()
        self.logger = get_logger()
        self.vector_store = get_vector_store()
        self.processor = ContentProcessor(self.config)
        
        # Content directories
        self.content_dirs = self._initialize_content_directories()
        
        # Loading statistics
        self.loading_stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "chunks_stored": 0,
            "errors_encountered": 0,
            "processing_time_ms": 0.0
        }
        
        self.logger.log_event(
            EventType.COMPONENT_INIT,
            "Educational content loader initialized",
            extra_data={"chunking_strategy": self.config.chunking_strategy.value}
        )
    
    def load_all_content(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load all educational content from configured directories.
        
        Args:
            force_reload: Whether to reload content even if already processed
            
        Returns:
            Dictionary with loading results and statistics
        """
        start_time = time.time()
        
        try:
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                "Starting full content loading process",
                extra_data={"force_reload": force_reload}
            )
            
            # Reset statistics
            self.loading_stats = {
                "files_processed": 0,
                "chunks_created": 0,
                "chunks_stored": 0,
                "errors_encountered": 0,
                "processing_time_ms": 0.0
            }
            
            # Process each content directory
            for specialization, directory_path in self.content_dirs.items():
                if directory_path.exists():
                    self._load_directory_content(directory_path, specialization, force_reload)
                else:
                    self.logger.log_event(
                        EventType.WARNING_ISSUED,
                        f"Content directory not found: {directory_path}",
                        level="WARNING"
                    )
            
            # Calculate total processing time
            processing_time = (time.time() - start_time) * 1000
            self.loading_stats["processing_time_ms"] = processing_time
            
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                "Content loading completed",
                extra_data=self.loading_stats
            )
            
            return {
                "status": "success",
                **self.loading_stats
            }
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Content loading failed: {str(e)}",
                level="ERROR"
            )
            
            return {
                "status": "error",
                "error": str(e),
                **self.loading_stats
            }
    
    def _initialize_content_directories(self) -> Dict[str, Path]:
        """Initialize content directory mappings."""
        base_dir = Path("data/educational_content")
        
        return {
            "implementation": base_dir / "implementation_guides",
            "debugging": base_dir / "debugging_resources", 
            "shared": base_dir / "general_programming"
        }
    
    def _load_directory_content(self, 
                               directory: Path,
                               specialization: str,
                               force_reload: bool):
        """Load content from a specific directory."""
        try:
            # Find all text files in directory
            text_files = list(directory.glob("*.txt"))
            md_files = list(directory.glob("*.md"))
            all_files = text_files + md_files
            
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                f"Loading {len(all_files)} files from {specialization} directory",
                extra_data={"directory": str(directory), "file_count": len(all_files)}
            )
            
            for file_path in all_files:
                try:
                    self._load_file_content(file_path, specialization)
                    self.loading_stats["files_processed"] += 1
                except Exception as e:
                    self.loading_stats["errors_encountered"] += 1
                    self.logger.log_event(
                        EventType.ERROR_OCCURRED,
                        f"Failed to load file {file_path}: {str(e)}",
                        level="ERROR"
                    )
                    
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Failed to load directory {directory}: {str(e)}",
                level="ERROR"
            )
    
    def _load_file_content(self, file_path: Path, specialization: str):
        """Load and process content from a single file."""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return
            
            # Determine content type and agent specialization
            content_type = self._determine_content_type(file_path.name, content)
            agent_spec = self._map_specialization(specialization)
            
            # Create base metadata
            content_id = self._generate_content_id(file_path)
            base_metadata = ContentMetadata(
                content_id=content_id,
                content_type=content_type,
                agent_specialization=agent_spec,
                source_file=str(file_path)
            )
            
            # Process content into chunks
            chunks = self.processor.process_content(
                content, str(file_path), base_metadata
            )
            
            self.loading_stats["chunks_created"] += len(chunks)
            
            # Store chunks in vector store
            stored_count = 0
            for chunk in chunks:
                if self.vector_store.add_content(chunk.content, chunk.metadata):
                    stored_count += 1
            
            self.loading_stats["chunks_stored"] += stored_count
            
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                f"Loaded file: {file_path.name} -> {stored_count}/{len(chunks)} chunks stored",
                extra_data={
                    "file_path": str(file_path),
                    "content_type": content_type.value,
                    "chunks_created": len(chunks),
                    "chunks_stored": stored_count
                }
            )
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Failed to process file {file_path}: {str(e)}",
                level="ERROR"
            )
            raise
    
    def _determine_content_type(self, filename: str, content: str) -> ContentType:
        """Determine content type based on filename and content."""
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        # Check filename patterns
        if "debug" in filename_lower or "error" in filename_lower:
            return ContentType.DEBUGGING_RESOURCE
        elif "implement" in filename_lower or "approach" in filename_lower:
            return ContentType.IMPLEMENTATION_GUIDE
        elif "example" in filename_lower:
            return ContentType.CODE_EXAMPLE
        elif "practice" in filename_lower or "exercise" in filename_lower:
            return ContentType.EXERCISE
        elif "best" in filename_lower and "practice" in filename_lower:
            return ContentType.BEST_PRACTICE
        
        # Check content patterns
        if any(word in content_lower for word in ["error", "debug", "fix", "troubleshoot"]):
            return ContentType.DEBUGGING_RESOURCE
        elif any(word in content_lower for word in ["implement", "approach", "design", "strategy"]):
            return ContentType.IMPLEMENTATION_GUIDE
        elif "```" in content or "example:" in content_lower:
            return ContentType.CODE_EXAMPLE
        elif any(word in content_lower for word in ["concept", "understand", "explain"]):
            return ContentType.CONCEPT_EXPLANATION
        
        return ContentType.GENERAL
    
    def _map_specialization(self, specialization: str) -> AgentSpecialization:
        """Map directory specialization to agent specialization."""
        mapping = {
            "implementation": AgentSpecialization.IMPLEMENTATION,
            "debugging": AgentSpecialization.DEBUGGING,
            "shared": AgentSpecialization.SHARED
        }
        
        return mapping.get(specialization, AgentSpecialization.SHARED)
    
    def _generate_content_id(self, file_path: Path) -> str:
        """Generate unique content ID for file."""
        # Use file path hash for consistent IDs
        path_str = str(file_path.absolute())
        hash_object = hashlib.md5(path_str.encode())
        return f"content_{hash_object.hexdigest()[:12]}"
    
    def get_loading_stats(self) -> Dict[str, Any]:
        """Get content loading statistics."""
        return self.loading_stats.copy()


# Global content loader instance
_content_loader: Optional[EducationalContentLoader] = None


def get_content_loader(config: Optional[ContentLoadingConfig] = None, 
                      reload: bool = False) -> EducationalContentLoader:
    """
    Get global content loader instance (singleton pattern).
    
    Args:
        config: Content loading configuration
        reload: Force creation of new loader instance
        
    Returns:
        EducationalContentLoader instance
    """
    global _content_loader
    if _content_loader is None or reload:
        _content_loader = EducationalContentLoader(config)
    return _content_loader


if __name__ == "__main__":
    # Content loader test
    try:
        # Create test content directory structure
        test_dir = Path("test_content")
        test_dir.mkdir(exist_ok=True)
        
        # Create sample content
        sample_content = """
        # Binary Search Algorithm
        
        Binary search is an efficient algorithm for finding an item from a sorted list.
        It works by repeatedly dividing the search interval in half.
        
        ## Implementation
        
        Here's a simple implementation:
        
        ```python
        def binary_search(arr, target):
            left, right = 0, len(arr) - 1
            
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return -1
        ```
        
        This algorithm has O(log n) time complexity, making it very efficient for large datasets.
        """
        
        test_file = test_dir / "binary_search.md"
        with open(test_file, 'w') as f:
            f.write(sample_content)
        
        # Test content processor
        config = ContentLoadingConfig(
            max_chunk_size=500,
            chunking_strategy=ChunkingStrategy.SEMANTIC
        )
        
        processor = ContentProcessor(config)
        
        base_metadata = ContentMetadata(
            content_id="test_content",
            content_type=ContentType.IMPLEMENTATION_GUIDE,
            agent_specialization=AgentSpecialization.IMPLEMENTATION
        )
        
        chunks = processor.process_content(sample_content, str(test_file), base_metadata)
        
        print(f"Content processor test: {len(chunks)} chunks created")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i}: {len(chunk.content)} chars, "
                  f"quality: {chunk.educational_value:.2f}")
        
        # Clean up
        test_file.unlink()
        test_dir.rmdir()
        
        print("✅ Content loader test completed successfully!")
        
    except Exception as e:
        print(f"❌ Content loader test failed: {e}")
        import traceback
        traceback.print_exc()
