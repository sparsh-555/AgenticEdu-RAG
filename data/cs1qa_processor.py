"""
CS1QA Dataset Processing & SRL Labeling Module

This module handles CS1QA dataset processing for evaluation purposes, including
SRL phase labeling based on query content analysis and preparation of evaluation
datasets with ground truth labels for implementation vs debugging classification.

Author: Agentic Edu-RAG System
"""

import json
import csv
import re
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np
from openai import AsyncOpenAI
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CS1QAQuery:
    """Data structure for CS1QA query entries."""
    query_id: str
    question: str
    code_snippet: Optional[str] = None
    error_message: Optional[str] = None
    tags: List[str] = None
    difficulty_level: Optional[str] = None
    original_context: Dict[str, Any] = None

@dataclass
class SRLLabel:
    """SRL phase label with metadata."""
    phase: str  # 'implementation' or 'debugging'
    confidence: float
    reasoning: str
    classification_features: Dict[str, Any]

@dataclass
class LabeledQuery:
    """CS1QA query with SRL labeling."""
    query: CS1QAQuery
    srl_label: SRLLabel
    ground_truth_label: Optional[str] = None
    validation_status: str = "pending"  # pending, validated, disputed

class CS1QAProcessor:
    """
    Processes CS1QA dataset for SRL-aware evaluation.
    
    This class handles loading, preprocessing, and labeling of CS1QA data
    to create evaluation datasets aligned with Self-Regulated Learning phases.
    """
    
    def __init__(self, 
                 data_dir: Path,
                 openai_api_key: str,
                 output_dir: Optional[Path] = None):
        """
        Initialize CS1QA processor.
        
        Args:
            data_dir: Path to CS1QA dataset directory
            openai_api_key: OpenAI API key for classification
            output_dir: Directory for processed data output
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "processed"
        self.raw_data_dir = self.data_dir / "raw"
        
        # Initialize OpenAI client for SRL classification
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        
        # Initialize data containers
        self.raw_queries: List[CS1QAQuery] = []
        self.labeled_queries: List[LabeledQuery] = []
        
        # Classification prompts for SRL phase detection
        self.classification_prompts = self._load_classification_prompts()
        
        # Validation metrics
        self.validation_metrics = {}
        
    def _load_classification_prompts(self) -> Dict[str, str]:
        """Load SRL classification prompt templates."""
        return {
            "system_prompt": """You are an expert in Self-Regulated Learning (SRL) theory and programming education. 
            Your task is to classify programming student queries into SRL phases based on their content and context.

            SRL Phases:
            1. IMPLEMENTATION (Forethought Phase): Queries about planning, strategy, "how to code", algorithm design, 
               approach selection, implementation guidance, and conceptual understanding.
            2. DEBUGGING (Performance Phase): Queries about error resolution, troubleshooting, monitoring execution, 
               fixing broken code, and performance issues.

            Consider these indicators:
            - Implementation queries often ask "how to", seek guidance on approach, request planning help
            - Debugging queries often include error messages, broken code, or troubleshooting requests
            - Context clues from code snippets and error messages are crucial
            - Student language patterns reveal learning phase intent""",
            
            "classification_prompt": """Classify this programming query into SRL phase:

Query: "{query}"
{code_context}
{error_context}

Respond with JSON:
{{
    "phase": "implementation" or "debugging",
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation",
    "features": {{
        "has_error_message": boolean,
        "has_broken_code": boolean,
        "seeks_implementation_guidance": boolean,
        "seeks_troubleshooting_help": boolean,
        "query_intent_keywords": ["list", "of", "keywords"],
        "complexity_indicators": ["list", "of", "indicators"]
    }}
}}"""
        }

    async def load_cs1qa_dataset(self) -> None:
        """
        Load CS1QA dataset from various file formats.
        
        Supports JSON, CSV, and TSV formats commonly used for CS1QA data.
        """
        logger.info(f"Loading CS1QA dataset from {self.raw_data_dir}")
        
        # Look for common CS1QA file patterns
        data_files = []
        for pattern in ["*.json", "*.jsonl", "*.csv", "*.tsv"]:
            data_files.extend(self.raw_data_dir.glob(pattern))
        
        if not data_files:
            raise FileNotFoundError(f"No CS1QA data files found in {self.raw_data_dir}")
        
        for file_path in data_files:
            logger.info(f"Processing {file_path}")
            
            if file_path.suffix == ".json":
                await self._load_json_data(file_path)
            elif file_path.suffix == ".jsonl":
                await self._load_jsonl_data(file_path)
            elif file_path.suffix in [".csv", ".tsv"]:
                await self._load_csv_data(file_path)
        
        logger.info(f"Loaded {len(self.raw_queries)} queries from CS1QA dataset")

    async def _load_json_data(self, file_path: Path) -> None:
        """Load data from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            for item in data:
                self._parse_query_item(item)
        elif isinstance(data, dict):
            if "queries" in data:
                for item in data["queries"]:
                    self._parse_query_item(item)
            else:
                self._parse_query_item(data)

    async def _load_jsonl_data(self, file_path: Path) -> None:
        """Load data from JSONL file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self._parse_query_item(item)

    async def _load_csv_data(self, file_path: Path) -> None:
        """Load data from CSV/TSV file."""
        delimiter = '\t' if file_path.suffix == '.tsv' else ','
        df = pd.read_csv(file_path, delimiter=delimiter)
        
        for _, row in df.iterrows():
            self._parse_query_item(row.to_dict())

    def _parse_query_item(self, item: Dict[str, Any]) -> None:
        """Parse individual query item into CS1QAQuery object."""
        try:
            # Extract common fields with flexible naming
            query_id = str(item.get('id', item.get('query_id', len(self.raw_queries))))
            question = item.get('question', item.get('query', item.get('text', '')))
            
            # Extract code snippet with flexible field names
            code_snippet = item.get('code', item.get('code_snippet', 
                                    item.get('program', item.get('source_code'))))
            
            # Extract error message
            error_message = item.get('error', item.get('error_message', 
                                    item.get('exception', item.get('error_output'))))
            
            # Extract tags/categories
            tags = item.get('tags', item.get('categories', item.get('labels', [])))
            if isinstance(tags, str):
                tags = [tag.strip() for tag in tags.split(',')]
            
            # Extract difficulty
            difficulty = item.get('difficulty', item.get('level', item.get('complexity')))
            
            query = CS1QAQuery(
                query_id=query_id,
                question=question,
                code_snippet=code_snippet,
                error_message=error_message,
                tags=tags or [],
                difficulty_level=difficulty,
                original_context=item
            )
            
            self.raw_queries.append(query)
            
        except Exception as e:
            logger.warning(f"Failed to parse query item: {e}")

    async def apply_srl_labeling(self, 
                                batch_size: int = 5,
                                max_concurrent: int = 3) -> None:
        """
        Apply SRL phase labeling to all queries using GPT-based classification.
        
        Args:
            batch_size: Number of queries to process in each batch
            max_concurrent: Maximum concurrent API requests
        """
        logger.info(f"Starting SRL labeling for {len(self.raw_queries)} queries")
        
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Process queries in batches
        for i in range(0, len(self.raw_queries), batch_size):
            batch = self.raw_queries[i:i + batch_size]
            
            # Create tasks for concurrent processing
            tasks = [
                self._classify_query_srl(query, semaphore) 
                for query in batch
            ]
            
            # Execute batch with error handling
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for query, result in zip(batch, results):
                    if isinstance(result, Exception):
                        logger.error(f"Classification failed for query {query.query_id}: {result}")
                        # Create default label for failed classification
                        srl_label = SRLLabel(
                            phase="unknown",
                            confidence=0.0,
                            reasoning="Classification failed due to API error",
                            classification_features={}
                        )
                    else:
                        srl_label = result
                    
                    labeled_query = LabeledQuery(
                        query=query,
                        srl_label=srl_label
                    )
                    self.labeled_queries.append(labeled_query)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(self.raw_queries) + batch_size - 1)//batch_size}")
                
                # Brief pause between batches to respect rate limits
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")

    async def _classify_query_srl(self, 
                                 query: CS1QAQuery, 
                                 semaphore: asyncio.Semaphore) -> SRLLabel:
        """
        Classify a single query into SRL phase using OpenAI API.
        
        Args:
            query: CS1QA query to classify
            semaphore: Rate limiting semaphore
            
        Returns:
            SRL label with classification results
        """
        async with semaphore:
            try:
                # Prepare context information
                code_context = f"Code snippet: {query.code_snippet}" if query.code_snippet else ""
                error_context = f"Error message: {query.error_message}" if query.error_message else ""
                
                # Format classification prompt
                prompt = self.classification_prompts["classification_prompt"].format(
                    query=query.question,
                    code_context=code_context,
                    error_context=error_context
                )
                
                # Make API request
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": self.classification_prompts["system_prompt"]},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                
                # Parse response with safe content extraction
                if not response.choices:
                    raise ValueError("No choices in OpenAI response")
                
                message = response.choices[0].message
                if hasattr(message, 'content') and message.content:
                    response_text = message.content
                    
                    # Strip markdown code blocks if present
                    response_text = response_text.strip()
                    if response_text.startswith('```json'):
                        response_text = response_text[7:]  # Remove ```json
                    if response_text.startswith('```'):
                        response_text = response_text[3:]   # Remove ``` 
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]  # Remove closing ```
                    response_text = response_text.strip()
                    
                else:
                    raise ValueError("OpenAI response missing content")
                classification_result = json.loads(response_text)
                
                # Create SRL label
                srl_label = SRLLabel(
                    phase=classification_result["phase"],
                    confidence=classification_result["confidence"],
                    reasoning=classification_result["reasoning"],
                    classification_features=classification_result.get("features", {})
                )
                
                return srl_label
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse classification response for query {query.query_id}: {e}")
                return self._create_fallback_label(query)
                
            except Exception as e:
                logger.error(f"API error for query {query.query_id}: {e}")
                return self._create_fallback_label(query)

    def _create_fallback_label(self, query: CS1QAQuery) -> SRLLabel:
        """Create fallback SRL label using rule-based classification."""
        
        # Simple heuristic-based classification
        question_lower = query.question.lower()
        
        # Debugging indicators
        debugging_keywords = [
            'error', 'exception', 'bug', 'fix', 'debug', 'wrong', 'not working',
            'traceback', 'crash', 'fail', 'broken', 'issue', 'problem'
        ]
        
        # Implementation indicators  
        implementation_keywords = [
            'how to', 'how do i', 'implement', 'create', 'write', 'code',
            'algorithm', 'approach', 'method', 'solution', 'design'
        ]
        
        # Check for error message presence
        has_error = query.error_message is not None
        
        # Count keyword matches
        debug_score = sum(1 for kw in debugging_keywords if kw in question_lower)
        impl_score = sum(1 for kw in implementation_keywords if kw in question_lower)
        
        # Make classification decision
        if has_error or debug_score > impl_score:
            phase = "debugging"
            confidence = 0.6 if has_error else 0.4
        else:
            phase = "implementation"
            confidence = 0.5
        
        return SRLLabel(
            phase=phase,
            confidence=confidence,
            reasoning="Fallback rule-based classification",
            classification_features={
                "has_error_message": has_error,
                "debug_keyword_count": debug_score,
                "implementation_keyword_count": impl_score
            }
        )

    def create_evaluation_datasets(self, 
                                  test_size: float = 0.2,
                                  validation_size: float = 0.1,
                                  random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """
        Create train/validation/test splits for evaluation.
        
        Args:
            test_size: Proportion for test set
            validation_size: Proportion for validation set  
            random_state: Random seed for reproducible splits
            
        Returns:
            Dictionary containing train, validation, and test DataFrames
        """
        logger.info("Creating evaluation dataset splits")
        
        # Convert labeled queries to DataFrame
        data_records = []
        for labeled_query in self.labeled_queries:
            record = {
                'query_id': labeled_query.query.query_id,
                'question': labeled_query.query.question,
                'code_snippet': labeled_query.query.code_snippet or '',
                'error_message': labeled_query.query.error_message or '',
                'tags': ','.join(labeled_query.query.tags),
                'difficulty_level': labeled_query.query.difficulty_level or '',
                'srl_phase': labeled_query.srl_label.phase,
                'confidence': labeled_query.srl_label.confidence,
                'reasoning': labeled_query.srl_label.reasoning,
                'has_error_message': bool(labeled_query.query.error_message),
                'has_code_snippet': bool(labeled_query.query.code_snippet),
                'question_length': len(labeled_query.query.question),
                'ground_truth_label': labeled_query.ground_truth_label
            }
            data_records.append(record)
        
        df = pd.DataFrame(data_records)
        
        # Filter out unknown classifications for evaluation
        df_filtered = df[df['srl_phase'].isin(['implementation', 'debugging'])].copy()
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df_filtered, 
            test_size=test_size, 
            stratify=df_filtered['srl_phase'],
            random_state=random_state
        )
        
        # Second split: separate validation from training
        val_relative_size = validation_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_relative_size,
            stratify=train_val_df['srl_phase'],
            random_state=random_state
        )
        
        datasets = {
            'train': train_df,
            'validation': val_df,
            'test': test_df,
            'full': df
        }
        
        # Log dataset statistics
        for split_name, split_df in datasets.items():
            if split_name != 'full':
                phase_counts = split_df['srl_phase'].value_counts()
                logger.info(f"{split_name.capitalize()} set: {len(split_df)} samples")
                for phase, count in phase_counts.items():
                    logger.info(f"  {phase}: {count} ({count/len(split_df)*100:.1f}%)")
        
        return datasets

    def validate_labeling_consistency(self, 
                                    sample_size: int = 100,
                                    inter_rater_validation: bool = True) -> Dict[str, float]:
        """
        Validate SRL labeling consistency and quality.
        
        Args:
            sample_size: Number of samples for validation
            inter_rater_validation: Whether to perform inter-rater reliability check
            
        Returns:
            Validation metrics dictionary
        """
        logger.info("Validating labeling consistency")
        
        # Sample queries for validation
        sample_queries = np.random.choice(
            self.labeled_queries, 
            size=min(sample_size, len(self.labeled_queries)), 
            replace=False
        )
        
        validation_results = {
            'total_samples': len(sample_queries),
            'high_confidence_ratio': 0.0,
            'phase_distribution': {},
            'feature_consistency': {}
        }
        
        # Analyze confidence distribution
        confidences = [lq.srl_label.confidence for lq in sample_queries]
        validation_results['mean_confidence'] = np.mean(confidences)
        validation_results['std_confidence'] = np.std(confidences) 
        validation_results['high_confidence_ratio'] = np.mean([c >= 0.7 for c in confidences])
        
        # Analyze phase distribution
        phases = [lq.srl_label.phase for lq in sample_queries]
        validation_results['phase_distribution'] = {
            phase: phases.count(phase) / len(phases) 
            for phase in set(phases)
        }
        
        # Feature consistency analysis
        error_with_debug = 0
        no_error_with_impl = 0
        
        for lq in sample_queries:
            has_error = bool(lq.query.error_message)
            is_debug = lq.srl_label.phase == 'debugging'
            
            if has_error and is_debug:
                error_with_debug += 1
            if not has_error and lq.srl_label.phase == 'implementation':
                no_error_with_impl += 1
        
        validation_results['feature_consistency'] = {
            'error_debug_alignment': error_with_debug / len([lq for lq in sample_queries if lq.query.error_message]),
            'no_error_impl_alignment': no_error_with_impl / len([lq for lq in sample_queries if not lq.query.error_message])
        }
        
        self.validation_metrics = validation_results
        logger.info(f"Validation completed. Mean confidence: {validation_results['mean_confidence']:.3f}")
        
        return validation_results

    def save_processed_data(self, output_format: str = 'both') -> None:
        """
        Save processed and labeled data to files.
        
        Args:
            output_format: 'json', 'csv', or 'both'
        """
        logger.info(f"Saving processed data to {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for export
        export_data = []
        for labeled_query in self.labeled_queries:
            export_record = {
                **asdict(labeled_query.query),
                'srl_label': asdict(labeled_query.srl_label),
                'ground_truth_label': labeled_query.ground_truth_label,
                'validation_status': labeled_query.validation_status,
                'processing_timestamp': datetime.now().isoformat()
            }
            export_data.append(export_record)
        
        if output_format in ['json', 'both']:
            # Save as JSON
            json_path = self.output_dir / 'cs1qa_srl_labeled.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved JSON data to {json_path}")
        
        if output_format in ['csv', 'both']:
            # Flatten for CSV export
            csv_records = []
            for record in export_data:
                flat_record = {
                    'query_id': record['query_id'],
                    'question': record['question'],
                    'code_snippet': record['code_snippet'] or '',
                    'error_message': record['error_message'] or '',
                    'tags': ','.join(record['tags']) if record['tags'] else '',
                    'difficulty_level': record['difficulty_level'] or '',
                    'srl_phase': record['srl_label']['phase'],
                    'srl_confidence': record['srl_label']['confidence'],
                    'srl_reasoning': record['srl_label']['reasoning'],
                    'ground_truth_label': record['ground_truth_label'] or '',
                    'validation_status': record['validation_status'],
                    'processing_timestamp': record['processing_timestamp']
                }
                csv_records.append(flat_record)
            
            csv_path = self.output_dir / 'cs1qa_srl_labeled.csv'
            df = pd.DataFrame(csv_records)
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV data to {csv_path}")
        
        # Save validation metrics
        if self.validation_metrics:
            metrics_path = self.output_dir / 'validation_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(self.validation_metrics, f, indent=2)
            logger.info(f"Saved validation metrics to {metrics_path}")

    def generate_processing_report(self) -> Dict[str, Any]:
        """Generate comprehensive processing report."""
        
        total_queries = len(self.raw_queries)
        labeled_queries = len(self.labeled_queries)
        successful_labels = len([lq for lq in self.labeled_queries if lq.srl_label.phase != 'unknown'])
        
        # Phase distribution
        phase_counts = {}
        confidence_stats = {}
        
        if self.labeled_queries:
            phases = [lq.srl_label.phase for lq in self.labeled_queries]
            confidences = [lq.srl_label.confidence for lq in self.labeled_queries if lq.srl_label.phase != 'unknown']
            
            for phase in set(phases):
                phase_counts[phase] = phases.count(phase)
            
            if confidences:
                confidence_stats = {
                    'mean': np.mean(confidences),
                    'std': np.std(confidences),
                    'min': np.min(confidences),
                    'max': np.max(confidences),
                    'median': np.median(confidences)
                }
        
        report = {
            'processing_summary': {
                'total_queries_loaded': total_queries,
                'queries_labeled': labeled_queries,
                'successful_classifications': successful_labels,
                'classification_success_rate': successful_labels / total_queries if total_queries > 0 else 0
            },
            'phase_distribution': phase_counts,
            'confidence_statistics': confidence_stats,
            'validation_metrics': self.validation_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        return report

# Example usage and testing
async def main():
    """Example usage of CS1QAProcessor."""
    
    # Initialize processor
    processor = CS1QAProcessor(
        data_dir=Path("data/cs1qa"),
        openai_api_key="your-api-key-here"  # Replace with actual API key
    )
    
    try:
        # Load CS1QA dataset
        await processor.load_cs1qa_dataset()
        
        # Apply SRL labeling
        await processor.apply_srl_labeling(batch_size=3, max_concurrent=2)
        
        # Validate labeling consistency
        validation_results = processor.validate_labeling_consistency()
        
        # Create evaluation datasets
        datasets = processor.create_evaluation_datasets()
        
        # Save processed data
        processor.save_processed_data()
        
        # Generate report
        report = processor.generate_processing_report()
        print("Processing completed successfully!")
        print(f"Processed {report['processing_summary']['total_queries_loaded']} queries")
        print(f"Classification success rate: {report['processing_summary']['classification_success_rate']:.2%}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
