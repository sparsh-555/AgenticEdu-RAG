"""
Unified Educational Content Loader for RAG System

This module handles loading, processing, and preparing educational content for
the unified vector store. It supports PDF processing, intelligent chunking
strategies, and automated metadata extraction optimized for programming education.

Key Features:
1. PDF Support: Process PDF books and documents into educational content
2. Unified Loading: All content goes into single collection with rich metadata
3. Intelligent Chunking: Context-aware chunking that preserves educational coherence
4. Metadata Generation: Comprehensive metadata for content filtering and retrieval
5. Quality Validation: Content quality checks and educational appropriateness scoring
6. Incremental Loading: Support for updating and adding new content efficiently

Content Processing Pipeline:
1. PDF Processing: Extract and clean text from PDF documents
2. Content Analysis: Identify programming concepts, difficulty, and content types
3. Intelligent Chunking: Create coherent chunks for vector storage
4. Metadata Enrichment: Generate comprehensive metadata for filtering
5. Quality Validation: Ensure content meets educational standards
6. Unified Storage: Store all content in single collection with metadata
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

import PyPDF2
import fitz  # PyMuPDF for better PDF processing
from pydantic import BaseModel, Field

from .vector_store import (
    ContentMetadata, ContentType, AgentSpecialization, 
    UnifiedEducationalVectorStore, get_vector_store
)
from config.settings import get_settings
from utils.logging_utils import get_logger, LogContext, EventType, create_context


class ContentFormat(Enum):
    """Supported content formats for processing."""
    PDF = "pdf"
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
    PDF_AWARE = "pdf_aware"


@dataclass
class ContentChunk:
    """A chunk of educational content ready for unified vector storage."""
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
class UnifiedContentLoadingConfig:
    """Configuration for unified content loading process."""
    max_chunk_size: int = 1000
    min_chunk_size: int = 200
    chunk_overlap: int = 100
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.PDF_AWARE
    
    # Quality filters
    min_readability_score: float = 0.3
    min_educational_value: float = 0.3
    
    # Processing options
    extract_code_examples: bool = True
    detect_difficulty_automatically: bool = True
    preserve_formatting: bool = True
    
    # PDF processing options
    pdf_extract_images: bool = False
    pdf_extract_tables: bool = True
    pdf_clean_text: bool = True
    
    # Performance options
    batch_size: int = 50
    parallel_processing: bool = True


class UnifiedContentProcessor:
    """Processes raw content into educational chunks for unified storage."""
    
    def __init__(self, config: UnifiedContentLoadingConfig):
        """Initialize unified content processor with configuration."""
        self.config = config
        self.logger = get_logger()
        
        # Programming concept patterns
        self.concept_patterns = self._initialize_concept_patterns()
        
        # Code detection patterns
        self.code_patterns = self._initialize_code_patterns()
        
        # Difficulty indicators
        self.difficulty_indicators = self._initialize_difficulty_indicators()
        
        # Content type detection patterns
        self.content_type_patterns = self._initialize_content_type_patterns()
        
        # Agent specialization mapping patterns
        self.agent_specialization_patterns = self._initialize_agent_patterns()
    
    def process_pdf_content(self, 
                           pdf_path: Path, 
                           base_agent_specialization: AgentSpecialization = AgentSpecialization.SHARED) -> List[ContentChunk]:
        """
        Process PDF content into educational chunks.
        
        Args:
            pdf_path: Path to PDF file
            base_agent_specialization: Default agent specialization
            
        Returns:
            List of processed content chunks
        """
        try:
            # Extract text from PDF
            pdf_text = self._extract_pdf_text(pdf_path)
            
            if not pdf_text.strip():
                self.logger.log_event(
                    EventType.WARNING_ISSUED,
                    f"No text extracted from PDF: {pdf_path}",
                    level="WARNING"
                )
                return []
            
            # Process extracted content
            return self.process_content(
                content=pdf_text,
                source_file=str(pdf_path),
                base_agent_specialization=base_agent_specialization
            )
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"PDF processing failed for {pdf_path}: {str(e)}",
                level="ERROR"
            )
            return []
    
    def process_content(self, 
                       content: str, 
                       source_file: str,
                       base_agent_specialization: AgentSpecialization = AgentSpecialization.SHARED) -> List[ContentChunk]:
        """
        Process content into educational chunks for unified storage.
        
        Args:
            content: Raw content text
            source_file: Source file path
            base_agent_specialization: Base agent specialization
            
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
            
            # Detect difficulty level
            difficulty = self._detect_difficulty_level(cleaned_content)
            
            # Determine content type and agent specialization for each section
            content_sections = self._split_into_sections(cleaned_content)
            
            all_chunks = []
            
            for section_index, section_content in enumerate(content_sections):
                # Determine specific content type and agent specialization for this section
                content_type = self._determine_content_type(section_content)
                agent_specialization = self._determine_agent_specialization(section_content, base_agent_specialization)
                
                # Create base metadata for this section
                base_metadata = ContentMetadata(
                    content_id=f"{self._generate_content_id(source_file)}_section_{section_index}",
                    content_type=content_type,
                    agent_specialization=agent_specialization,
                    programming_concepts=self._extract_programming_concepts(section_content),
                    difficulty_level=difficulty,
                    content_length=len(section_content),
                    has_code_examples=self._has_code_examples(section_content),
                    has_error_examples=self._has_error_examples(section_content),
                    source_file=source_file
                )
                
                # Create chunks for this section
                section_chunks = self._create_chunks(section_content, base_metadata, content_format)
                
                # Validate and score chunks
                for chunk in section_chunks:
                    if self._validate_chunk_quality(chunk):
                        all_chunks.append(chunk)
            
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                f"Processed unified content: {len(all_chunks)} chunks from {source_file}",
                extra_data={
                    "source_file": source_file,
                    "original_length": len(content),
                    "sections_processed": len(content_sections),
                    "chunks_created": len(all_chunks),
                    "concepts_detected": len(concepts)
                }
            )
            
            return all_chunks
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Unified content processing failed for {source_file}: {str(e)}",
                level="ERROR"
            )
            return []
    
    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF using PyMuPDF for better quality."""
        try:
            # Use PyMuPDF (fitz) for better text extraction
            doc = fitz.open(str(pdf_path))
            text_content = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                if self.config.pdf_clean_text:
                    page_text = self._clean_pdf_text(page_text)
                
                text_content += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
            
            doc.close()
            return text_content
            
        except Exception as e:
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_content = ""
                    
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        
                        if self.config.pdf_clean_text:
                            page_text = self._clean_pdf_text(page_text)
                        
                        text_content += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
                    
                    return text_content
                    
            except Exception as fallback_error:
                self.logger.log_event(
                    EventType.ERROR_OCCURRED,
                    f"Both PDF extraction methods failed for {pdf_path}: {str(fallback_error)}",
                    level="ERROR"
                )
                return ""
    
    def _clean_pdf_text(self, text: str) -> str:
        """Clean extracted PDF text."""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove page headers/footers patterns
        text = re.sub(r'^.*?Page \d+.*?\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\s*\n', '', text, flags=re.MULTILINE)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add spaces between words
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words
        
        return text.strip()
    
    def _split_into_sections(self, content: str) -> List[str]:
        """Split content into logical sections for processing."""
        # Split by headers, major sections, or topics
        section_patterns = [
            r'\n\s*Chapter\s+\d+.*?\n',
            r'\n\s*Section\s+\d+.*?\n',
            r'\n\s*##+\s+.*?\n',  # Markdown headers
            r'\n\s*\d+\.\s+[A-Z].*?\n',  # Numbered sections
            r'\n\s*[A-Z][A-Z\s]{10,}\n',  # All caps headers
        ]
        
        # Try to split by patterns
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            if len(matches) > 1:
                sections = []
                last_end = 0
                
                for match in matches:
                    if last_end < match.start():
                        section = content[last_end:match.start()].strip()
                        if section:
                            sections.append(section)
                    last_end = match.start()
                
                # Add final section
                if last_end < len(content):
                    final_section = content[last_end:].strip()
                    if final_section:
                        sections.append(final_section)
                
                if len(sections) > 1:
                    return sections
        
        # Fallback: split by double newlines for paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # Combine small paragraphs into larger sections
        sections = []
        current_section = ""
        
        for paragraph in paragraphs:
            if len(current_section) + len(paragraph) > self.config.max_chunk_size * 2:
                if current_section:
                    sections.append(current_section)
                current_section = paragraph
            else:
                current_section += "\n\n" + paragraph if current_section else paragraph
        
        if current_section:
            sections.append(current_section)
        
        return sections if sections else [content]
    
    def _determine_content_type(self, content: str) -> ContentType:
        """Determine content type based on content analysis."""
        content_lower = content.lower()
        
        # Check content type patterns
        for content_type, patterns in self.content_type_patterns.items():
            score = sum(1 for pattern in patterns if pattern in content_lower)
            if score >= 2:  # Require multiple indicators
                return ContentType(content_type)
        
        # Check for code examples
        if self._has_code_examples(content):
            return ContentType.CODE_EXAMPLE
        
        # Check for error/debugging content
        if self._has_error_examples(content):
            return ContentType.DEBUGGING_RESOURCE
        
        # Default based on educational indicators
        if any(word in content_lower for word in ["concept", "understand", "explain", "theory"]):
            return ContentType.CONCEPT_EXPLANATION
        elif any(word in content_lower for word in ["implement", "approach", "strategy", "design"]):
            return ContentType.IMPLEMENTATION_GUIDE
        elif any(word in content_lower for word in ["best", "practice", "recommendation", "guideline"]):
            return ContentType.BEST_PRACTICE
        
        return ContentType.GENERAL
    
    def _determine_agent_specialization(self, content: str, base_specialization: AgentSpecialization) -> AgentSpecialization:
        """Determine agent specialization based on content analysis."""
        content_lower = content.lower()
        
        # Check for debugging/performance indicators
        debugging_indicators = [
            "debug", "error", "fix", "troubleshoot", "problem", "issue",
            "exception", "bug", "trace", "stack trace", "performance"
        ]
        debugging_score = sum(1 for indicator in debugging_indicators if indicator in content_lower)
        
        # Check for implementation/forethought indicators
        implementation_indicators = [
            "implement", "design", "plan", "approach", "strategy", "architecture",
            "algorithm", "method", "solution", "build", "create"
        ]
        implementation_score = sum(1 for indicator in implementation_indicators if indicator in content_lower)
        
        # Determine specialization based on scores
        if debugging_score > implementation_score and debugging_score >= 2:
            return AgentSpecialization.DEBUGGING
        elif implementation_score > debugging_score and implementation_score >= 2:
            return AgentSpecialization.IMPLEMENTATION
        else:
            return base_specialization  # Use base specialization as default
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content text for unified processing."""
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
        
        # Check for PDF indicators
        if "--- page" in content_lower:
            return ContentFormat.PDF
        
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
        """Extract programming concepts from content for unified metadata."""
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
    
    def _has_error_examples(self, content: str) -> bool:
        """Check if content contains error examples."""
        error_indicators = [
            'error:', 'exception:', 'traceback:', 'syntaxerror',
            'typeerror', 'indexerror', 'nameerror', 'valueerror'
        ]
        
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in error_indicators)
    
    def _create_chunks(self, 
                      content: str, 
                      base_metadata: ContentMetadata,
                      content_format: ContentFormat) -> List[ContentChunk]:
        """Create chunks based on configured strategy for unified storage."""
        if self.config.chunking_strategy == ChunkingStrategy.PDF_AWARE:
            return self._pdf_aware_chunking(content, base_metadata)
        elif self.config.chunking_strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunking(content, base_metadata)
        elif self.config.chunking_strategy == ChunkingStrategy.CODE_AWARE:
            return self._code_aware_chunking(content, base_metadata)
        elif self.config.chunking_strategy == ChunkingStrategy.EDUCATIONAL_UNIT:
            return self._educational_unit_chunking(content, base_metadata)
        elif self.config.chunking_strategy == ChunkingStrategy.PARAGRAPH:
            return self._paragraph_chunking(content, base_metadata)
        else:
            return self._fixed_size_chunking(content, base_metadata)
    
    def _pdf_aware_chunking(self, content: str, base_metadata: ContentMetadata) -> List[ContentChunk]:
        """Create chunks that respect PDF structure and page boundaries."""
        chunks = []
        
        # Split by page boundaries first
        pages = re.split(r'--- Page \d+ ---', content)
        pages = [p.strip() for p in pages if p.strip()]
        
        current_chunk = ""
        chunk_index = 0
        
        for page in pages:
            # For each page, split by paragraphs or sections
            paragraphs = page.split('\n\n')
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # Check if adding this paragraph would exceed max size
                if len(current_chunk) + len(paragraph) > self.config.max_chunk_size and current_chunk:
                    # Create chunk from current content
                    chunk = self._create_chunk_with_metadata(
                        current_chunk, base_metadata, chunk_index, len(pages)
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + "\n\n" + paragraph
                    chunk_index += 1
                else:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk_with_metadata(
                current_chunk, base_metadata, chunk_index, len(pages)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _semantic_chunking(self, content: str, base_metadata: ContentMetadata) -> List[ContentChunk]:
        """Create semantically coherent chunks for unified storage."""
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
                chunk = self._create_chunk_with_metadata(
                    current_chunk, base_metadata, chunk_index, len(sections)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + "\n\n" + section
                chunk_index += 1
            else:
                current_chunk += "\n\n" + section if current_chunk else section
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk_with_metadata(
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
                    chunk = self._create_chunk_with_metadata(
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
                    chunk = self._create_chunk_with_metadata(
                        current_chunk, base_metadata, chunk_index, len(parts)
                    )
                    chunks.append(chunk)
                    
                    current_chunk = part
                    chunk_index += 1
                else:
                    current_chunk += part
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk_with_metadata(
                current_chunk, base_metadata, chunk_index, len(parts)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _educational_unit_chunking(self, content: str, base_metadata: ContentMetadata) -> List[ContentChunk]:
        """Create chunks based on educational units."""
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
                chunk = self._create_chunk_with_metadata(
                    current_chunk, base_metadata, chunk_index, len(parts)
                )
                chunks.append(chunk)
                
                current_chunk = part
                chunk_index += 1
            else:
                current_chunk += part
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk_with_metadata(
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
                chunk = self._create_chunk_with_metadata(
                    current_chunk, base_metadata, chunk_index, len(paragraphs)
                )
                chunks.append(chunk)
                
                current_chunk = paragraph
                chunk_index += 1
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk_with_metadata(
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
                chunk = self._create_chunk_with_metadata(
                    chunk_content, base_metadata, chunk_index, 
                    (len(content) // self.config.max_chunk_size) + 1
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + self.config.max_chunk_size - self.config.chunk_overlap, end)
        
        return chunks
    
    def _create_chunk_with_metadata(self, 
                                   content: str, 
                                   base_metadata: ContentMetadata,
                                   chunk_index: int,
                                   total_chunks: int) -> ContentChunk:
        """Create a content chunk with comprehensive metadata for unified storage."""
        # Generate unique chunk ID
        chunk_id = f"{base_metadata.content_id}_chunk_{chunk_index}"
        
        # Create chunk metadata (copy base metadata and enhance)
        chunk_metadata = ContentMetadata(
            content_id=chunk_id,
            content_type=base_metadata.content_type,
            agent_specialization=base_metadata.agent_specialization,
            programming_concepts=self._extract_programming_concepts(content),  # Re-extract for chunk
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
    
    def _generate_content_id(self, file_path: str) -> str:
        """Generate unique content ID for file."""
        # Use file path hash for consistent IDs
        hash_object = hashlib.md5(file_path.encode())
        return f"unified_content_{hash_object.hexdigest()[:12]}"
    
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
    
    def _initialize_content_type_patterns(self) -> Dict[str, List[str]]:
        """Initialize content type detection patterns."""
        return {
            "implementation_guide": [
                "implement", "implementation", "approach", "strategy", "method",
                "build", "create", "design", "plan"
            ],
            "debugging_resource": [
                "debug", "debugging", "error", "fix", "troubleshoot", "problem",
                "issue", "trace", "exception"
            ],
            "concept_explanation": [
                "concept", "theory", "principle", "understand", "explain",
                "definition", "meaning", "what is"
            ],
            "code_example": [
                "example", "sample", "code", "snippet", "demonstration",
                "illustration", "show"
            ],
            "best_practice": [
                "best practice", "recommendation", "guideline", "convention",
                "standard", "should", "recommended"
            ],
            "exercise": [
                "exercise", "practice", "assignment", "problem", "challenge",
                "task", "try", "do"
            ]
        }
    
    def _initialize_agent_patterns(self) -> Dict[str, List[str]]:
        """Initialize agent specialization detection patterns."""
        return {
            "implementation": [
                "implement", "design", "plan", "approach", "strategy",
                "architecture", "algorithm", "method", "solution", "build"
            ],
            "debugging": [
                "debug", "error", "fix", "troubleshoot", "problem", "issue",
                "exception", "bug", "trace", "performance", "optimize"
            ]
        }


class UnifiedEducationalContentLoader:
    """
    Unified content loader for educational RAG system.
    
    This class loads educational content from PDFs and other sources into
    a unified vector store with comprehensive metadata for intelligent
    content filtering and retrieval.
    """
    
    def __init__(self, config: Optional[UnifiedContentLoadingConfig] = None):
        """Initialize the unified content loader."""
        self.config = config or UnifiedContentLoadingConfig()
        self.logger = get_logger()
        self.vector_store = get_vector_store()
        self.processor = UnifiedContentProcessor(self.config)
        
        # Content source directory
        self.content_dir = Path("data/pdfs")
        
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
            "Unified educational content loader initialized",
            extra_data={"chunking_strategy": self.config.chunking_strategy.value}
        )
    
    def load_pdf_books(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load educational content from PDF books in the data/pdfs directory.
        
        Args:
            force_reload: Whether to reload content even if already processed
            
        Returns:
            Dictionary with loading results and statistics
        """
        start_time = time.time()
        
        try:
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                "Starting unified PDF content loading",
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
            
            # Ensure PDF directory exists
            self.content_dir.mkdir(parents=True, exist_ok=True)
            
            # Find all PDF files
            pdf_files = list(self.content_dir.glob("*.pdf"))
            
            if not pdf_files:
                self.logger.log_event(
                    EventType.WARNING_ISSUED,
                    f"No PDF files found in {self.content_dir}",
                    level="WARNING"
                )
                return {
                    "status": "warning",
                    "message": f"No PDF files found in {self.content_dir}",
                    **self.loading_stats
                }
            
            # Process each PDF file
            for pdf_path in pdf_files:
                try:
                    self._load_pdf_file(pdf_path)
                    self.loading_stats["files_processed"] += 1
                except Exception as e:
                    self.loading_stats["errors_encountered"] += 1
                    self.logger.log_event(
                        EventType.ERROR_OCCURRED,
                        f"Failed to load PDF {pdf_path}: {str(e)}",
                        level="ERROR"
                    )
            
            # Calculate total processing time
            processing_time = (time.time() - start_time) * 1000
            self.loading_stats["processing_time_ms"] = processing_time
            
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                "Unified PDF content loading completed",
                extra_data=self.loading_stats
            )
            
            return {
                "status": "success",
                **self.loading_stats
            }
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Unified content loading failed: {str(e)}",
                level="ERROR"
            )
            
            return {
                "status": "error",
                "error": str(e),
                **self.loading_stats
            }
    
    def _load_pdf_file(self, pdf_path: Path):
        """Load and process content from a single PDF file."""
        try:
            # Determine base agent specialization from filename
            base_specialization = self._determine_base_specialization(pdf_path.name)
            
            # Process PDF content into chunks
            chunks = self.processor.process_pdf_content(pdf_path, base_specialization)
            
            self.loading_stats["chunks_created"] += len(chunks)
            
            # Store chunks in unified vector store
            stored_count = 0
            for chunk in chunks:
                if self.vector_store.add_content(chunk.content, chunk.metadata):
                    stored_count += 1
            
            self.loading_stats["chunks_stored"] += stored_count
            
            self.logger.log_event(
                EventType.KNOWLEDGE_RETRIEVED,
                f"Loaded PDF: {pdf_path.name} -> {stored_count}/{len(chunks)} chunks stored",
                extra_data={
                    "pdf_path": str(pdf_path),
                    "base_specialization": base_specialization.value,
                    "chunks_created": len(chunks),
                    "chunks_stored": stored_count
                }
            )
            
        except Exception as e:
            self.logger.log_event(
                EventType.ERROR_OCCURRED,
                f"Failed to process PDF {pdf_path}: {str(e)}",
                level="ERROR"
            )
            raise
    
    def _determine_base_specialization(self, filename: str) -> AgentSpecialization:
        """Determine base agent specialization from PDF filename."""
        filename_lower = filename.lower()
        
        # Check for debugging/troubleshooting indicators
        if any(word in filename_lower for word in ["debug", "troubleshoot", "error", "fix", "problem"]):
            return AgentSpecialization.DEBUGGING
        
        # Check for implementation/design indicators
        if any(word in filename_lower for word in ["implement", "design", "approach", "guide", "method"]):
            return AgentSpecialization.IMPLEMENTATION
        
        # Default to shared
        return AgentSpecialization.SHARED
    
    def get_loading_stats(self) -> Dict[str, Any]:
        """Get unified content loading statistics."""
        return self.loading_stats.copy()


# Global unified content loader instance
_unified_content_loader: Optional[UnifiedEducationalContentLoader] = None


def get_content_loader(config: Optional[UnifiedContentLoadingConfig] = None, 
                      reload: bool = False) -> UnifiedEducationalContentLoader:
    """
    Get global unified content loader instance (singleton pattern).
    
    Args:
        config: Content loading configuration
        reload: Force creation of new loader instance
        
    Returns:
        UnifiedEducationalContentLoader instance
    """
    global _unified_content_loader
    if _unified_content_loader is None or reload:
        _unified_content_loader = UnifiedEducationalContentLoader(config)
    return _unified_content_loader


if __name__ == "__main__":
    # Unified content loader test
    try:
        # Test with a sample PDF (if available)
        loader = get_content_loader()
        
        # Load PDF content
        result = loader.load_pdf_books()
        
        print(f"PDF loading result: {result['status']}")
        print(f"Files processed: {result.get('files_processed', 0)}")
        print(f"Chunks created: {result.get('chunks_created', 0)}")
        print(f"Chunks stored: {result.get('chunks_stored', 0)}")
        print(f"Processing time: {result.get('processing_time_ms', 0):.2f}ms")
        
        if result.get('errors_encountered', 0) > 0:
            print(f"Errors encountered: {result['errors_encountered']}")
        
        print("✅ Unified content loader test completed!")
        
    except Exception as e:
        print(f"❌ Unified content loader test failed: {e}")
        import traceback
        traceback.print_exc()
