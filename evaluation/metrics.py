"""
Comprehensive Evaluation Framework for Agentic Edu-RAG System

This module provides a sophisticated evaluation framework for assessing the performance
of the multi-agent educational RAG system. It includes metrics for classification
accuracy, response quality, educational effectiveness, and system performance.

Key Evaluation Areas:
1. SRL Classification Accuracy: Precision, recall, F1-score for routing decisions
2. Response Quality: Educational appropriateness, clarity, and helpfulness
3. Educational Effectiveness: Learning support and pedagogical quality
4. System Performance: Response time, throughput, and resource utilization
5. Agent Specialization: Effectiveness of specialized vs. general responses
6. User Experience: Satisfaction and learning outcomes

Evaluation Methodology:
- Quantitative Metrics: Statistical analysis of system performance
- Qualitative Assessment: Expert evaluation of educational quality
- Comparative Analysis: Multi-agent vs. single-agent performance
- Longitudinal Studies: Learning effectiveness over time
- A/B Testing: Optimization of system parameters
"""

import time
import json
import statistics
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from ..utils.logging_utils import get_logger, LogContext, EventType, create_context
from ..config.settings import get_settings


class EvaluationMetricType(Enum):
    """Types of evaluation metrics."""
    CLASSIFICATION_ACCURACY = "classification_accuracy"
    RESPONSE_QUALITY = "response_quality"
    EDUCATIONAL_EFFECTIVENESS = "educational_effectiveness"
    SYSTEM_PERFORMANCE = "system_performance"
    USER_EXPERIENCE = "user_experience"
    AGENT_SPECIALIZATION = "agent_specialization"


class ResponseQualityDimension(Enum):
    """Dimensions for response quality evaluation."""
    ACCURACY = "accuracy"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    HELPFULNESS = "helpfulness"
    EDUCATIONAL_VALUE = "educational_value"


@dataclass
class EvaluationSample:
    """A single evaluation sample with ground truth and predictions."""
    sample_id: str
    query: str
    ground_truth_srl_phase: str
    predicted_srl_phase: str
    classification_confidence: float
    
    # Context information
    agent_used: str
    response_content: str
    student_level: Optional[str] = None
    has_code_snippet: bool = False
    has_error_message: bool = False
    
    # Ground truth annotations
    expected_agent: Optional[str] = None
    response_quality_scores: Dict[str, float] = field(default_factory=dict)
    educational_effectiveness_score: Optional[float] = None
    
    # System performance
    response_time_ms: float = 0.0
    tokens_used: int = 0
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    evaluator_id: Optional[str] = None


class ClassificationMetrics(BaseModel):
    """Metrics for SRL classification evaluation."""
    accuracy: float = Field(..., description="Overall classification accuracy")
    precision: Dict[str, float] = Field(default_factory=dict, description="Precision per class")
    recall: Dict[str, float] = Field(default_factory=dict, description="Recall per class")
    f1_score: Dict[str, float] = Field(default_factory=dict, description="F1-score per class")
    
    # Detailed metrics
    confusion_matrix: List[List[int]] = Field(default_factory=list, description="Confusion matrix")
    classification_report: Dict[str, Any] = Field(default_factory=dict, description="Detailed report")
    
    # Confidence-based metrics
    average_confidence: float = Field(default=0.0, description="Average classification confidence")
    confidence_accuracy_correlation: float = Field(default=0.0, description="Confidence-accuracy correlation")
    
    # Educational context
    accuracy_by_student_level: Dict[str, float] = Field(default_factory=dict)
    accuracy_by_query_type: Dict[str, float] = Field(default_factory=dict)


class ResponseQualityMetrics(BaseModel):
    """Metrics for response quality evaluation."""
    overall_quality: float = Field(..., description="Overall response quality score")
    quality_dimensions: Dict[str, float] = Field(default_factory=dict, description="Quality by dimension")
    
    # Quality distribution
    excellent_responses_ratio: float = Field(default=0.0, description="Ratio of excellent responses")
    poor_responses_ratio: float = Field(default=0.0, description="Ratio of poor responses")
    
    # Agent comparison
    quality_by_agent: Dict[str, float] = Field(default_factory=dict, description="Quality by agent type")
    
    # Educational appropriateness
    educational_appropriateness: float = Field(default=0.0, description="Educational appropriateness score")
    complexity_alignment: float = Field(default=0.0, description="Complexity-level alignment")


class SystemPerformanceMetrics(BaseModel):
    """Metrics for system performance evaluation."""
    # Response time metrics
    average_response_time_ms: float = Field(..., description="Average response time")
    percentile_95_response_time_ms: float = Field(..., description="95th percentile response time")
    response_time_by_agent: Dict[str, float] = Field(default_factory=dict)
    
    # Throughput metrics
    queries_per_second: float = Field(default=0.0, description="Query processing throughput")
    concurrent_query_support: int = Field(default=1, description="Max concurrent queries supported")
    
    # Resource utilization
    average_tokens_per_query: float = Field(default=0.0, description="Average tokens consumed")
    cost_per_query_usd: float = Field(default=0.0, description="Average cost per query")
    
    # Reliability metrics
    success_rate: float = Field(default=1.0, description="Query success rate")
    error_rate: float = Field(default=0.0, description="Query error rate")


class EducationalEffectivenessMetrics(BaseModel):
    """Metrics for educational effectiveness evaluation."""
    # Learning support metrics
    conceptual_understanding_support: float = Field(default=0.0, description="Support for conceptual understanding")
    practical_application_support: float = Field(default=0.0, description="Support for practical application")
    problem_solving_guidance: float = Field(default=0.0, description="Quality of problem-solving guidance")
    
    # SRL alignment
    srl_phase_appropriateness: float = Field(default=0.0, description="SRL phase alignment")
    metacognitive_support: float = Field(default=0.0, description="Metacognitive skill support")
    
    # Pedagogical quality
    scaffolding_quality: float = Field(default=0.0, description="Quality of educational scaffolding")
    feedback_effectiveness: float = Field(default=0.0, description="Effectiveness of feedback")
    
    # Learning outcomes
    knowledge_transfer_support: float = Field(default=0.0, description="Support for knowledge transfer")
    skill_development_support: float = Field(default=0.0, description="Support for skill development")


@dataclass
class EvaluationConfig:
    """Configuration for evaluation process."""
    # Sample selection
    sample_size: Optional[int] = None
    stratified_sampling: bool = True
    random_seed: int = 42
    
    # Quality thresholds
    min_classification_accuracy: float = 0.80
    min_response_quality: float = 0.70
    min_educational_effectiveness: float = 0.65
    
    # Performance thresholds
    max_response_time_ms: float = 3000.0
    min_success_rate: float = 0.95
    
    # Evaluation methods
    include_human_evaluation: bool = True
    include_automated_metrics: bool = True
    cross_validate: bool = True
    
    # Output configuration
    generate_detailed_reports: bool = True
    save_individual_scores: bool = True
    export_for_analysis: bool = True


class EducationalRAGEvaluator:
    """
    Comprehensive evaluator for the educational RAG system.
    
    This evaluator provides multi-dimensional assessment of system performance,
    focusing on educational effectiveness, technical performance, and user experience.
    It supports both automated metrics and human evaluation protocols.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize the evaluator with configuration."""
        self.config = config or EvaluationConfig()
        self.logger = get_logger()
        self.settings = get_settings()
        
        # Evaluation data storage
        self.evaluation_samples: List[EvaluationSample] = []
        self.evaluation_results: Dict[str, Any] = {}
        
        # Human evaluators (for future extension)
        self.human_evaluators: List[str] = []
        
        self.logger.log_event(
            EventType.COMPONENT_INIT,
            "Educational RAG evaluator initialized",
            extra_data={
                "config": {
                    "min_classification_accuracy": self.config.min_classification_accuracy,
                    "min_response_quality": self.config.min_response_quality,
                    "human_evaluation": self.config.include_human_evaluation
                }
            }
        )
    
    def add_evaluation_sample(self, sample: EvaluationSample):
        """Add a sample for evaluation."""
        self.evaluation_samples.append(sample)
    
    def evaluate_classification_performance(self, 
                                          samples: Optional[List[EvaluationSample]] = None) -> ClassificationMetrics:
        """
        Evaluate SRL classification performance.
        
        Args:
            samples: Evaluation samples (uses all if None)
            
        Returns:
            Classification performance metrics
        """
        samples = samples or self.evaluation_samples
        
        if not samples:
            raise ValueError("No evaluation samples available")
        
        # Extract ground truth and predictions
        y_true = [sample.ground_truth_srl_phase for sample in samples]
        y_pred = [sample.predicted_srl_phase for sample in samples]
        confidences = [sample.classification_confidence for sample in samples]
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate per-class metrics
        labels = list(set(y_true + y_pred))
        precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        
        # Create per-class dictionaries
        precision_dict = {labels[i]: precision[i] for i in range(len(labels))}
        recall_dict = {labels[i]: recall[i] for i in range(len(labels))}
        f1_dict = {labels[i]: f1[i] for i in range(len(labels))}
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Classification report
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        
        # Confidence metrics
        avg_confidence = statistics.mean(confidences)
        
        # Correlation between confidence and accuracy
        correct_predictions = [1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))]
        confidence_accuracy_corr = np.corrcoef(confidences, correct_predictions)[0, 1] if len(confidences) > 1 else 0.0
        
        # Accuracy by student level
        accuracy_by_level = self._calculate_accuracy_by_attribute(samples, "student_level")
        
        # Accuracy by query type (based on presence of code/error)
        accuracy_by_type = self._calculate_accuracy_by_query_type(samples)
        
        metrics = ClassificationMetrics(
            accuracy=accuracy,
            precision=precision_dict,
            recall=recall_dict,
            f1_score=f1_dict,
            confusion_matrix=cm.tolist(),
            classification_report=report,
            average_confidence=avg_confidence,
            confidence_accuracy_correlation=confidence_accuracy_corr,
            accuracy_by_student_level=accuracy_by_level,
            accuracy_by_query_type=accuracy_by_type
        )
        
        self.logger.log_event(
            EventType.LEARNING_INTERACTION,
            f"Classification evaluation completed: {accuracy:.3f} accuracy",
            extra_data={
                "accuracy": accuracy,
                "sample_count": len(samples),
                "average_confidence": avg_confidence
            }
        )
        
        return metrics
    
    def evaluate_response_quality(self, 
                                samples: Optional[List[EvaluationSample]] = None) -> ResponseQualityMetrics:
        """
        Evaluate response quality across multiple dimensions.
        
        Args:
            samples: Evaluation samples (uses all if None)
            
        Returns:
            Response quality metrics
        """
        samples = samples or self.evaluation_samples
        
        if not samples:
            raise ValueError("No evaluation samples available")
        
        # Calculate quality scores across dimensions
        quality_scores = defaultdict(list)
        overall_scores = []
        agent_scores = defaultdict(list)
        
        for sample in samples:
            if sample.response_quality_scores:
                # Individual dimension scores
                for dimension, score in sample.response_quality_scores.items():
                    quality_scores[dimension].append(score)
                
                # Overall score (average of dimensions)
                overall_score = statistics.mean(sample.response_quality_scores.values())
                overall_scores.append(overall_score)
                agent_scores[sample.agent_used].append(overall_score)
        
        if not overall_scores:
            # If no quality scores provided, use automated assessment
            return self._automated_response_quality_assessment(samples)
        
        # Calculate averages
        overall_quality = statistics.mean(overall_scores)
        quality_dimensions = {
            dim: statistics.mean(scores) for dim, scores in quality_scores.items()
        }
        
        # Quality distribution
        excellent_threshold = 0.8
        poor_threshold = 0.4
        
        excellent_count = sum(1 for score in overall_scores if score >= excellent_threshold)
        poor_count = sum(1 for score in overall_scores if score <= poor_threshold)
        
        excellent_ratio = excellent_count / len(overall_scores)
        poor_ratio = poor_count / len(overall_scores)
        
        # Quality by agent
        quality_by_agent = {
            agent: statistics.mean(scores) for agent, scores in agent_scores.items()
        }
        
        # Educational appropriateness (simplified calculation)
        educational_appropriateness = self._calculate_educational_appropriateness(samples)
        complexity_alignment = self._calculate_complexity_alignment(samples)
        
        metrics = ResponseQualityMetrics(
            overall_quality=overall_quality,
            quality_dimensions=quality_dimensions,
            excellent_responses_ratio=excellent_ratio,
            poor_responses_ratio=poor_ratio,
            quality_by_agent=quality_by_agent,
            educational_appropriateness=educational_appropriateness,
            complexity_alignment=complexity_alignment
        )
        
        self.logger.log_event(
            EventType.LEARNING_INTERACTION,
            f"Response quality evaluation completed: {overall_quality:.3f} average quality",
            extra_data={
                "overall_quality": overall_quality,
                "excellent_ratio": excellent_ratio,
                "sample_count": len(samples)
            }
        )
        
        return metrics
    
    def evaluate_system_performance(self, 
                                  samples: Optional[List[EvaluationSample]] = None) -> SystemPerformanceMetrics:
        """
        Evaluate system performance metrics.
        
        Args:
            samples: Evaluation samples (uses all if None)
            
        Returns:
            System performance metrics
        """
        samples = samples or self.evaluation_samples
        
        if not samples:
            raise ValueError("No evaluation samples available")
        
        # Response time metrics
        response_times = [sample.response_time_ms for sample in samples if sample.response_time_ms > 0]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            percentile_95 = np.percentile(response_times, 95)
        else:
            avg_response_time = 0.0
            percentile_95 = 0.0
        
        # Response time by agent
        agent_response_times = defaultdict(list)
        for sample in samples:
            if sample.response_time_ms > 0:
                agent_response_times[sample.agent_used].append(sample.response_time_ms)
        
        response_time_by_agent = {
            agent: statistics.mean(times) for agent, times in agent_response_times.items()
        }
        
        # Token usage
        tokens_used = [sample.tokens_used for sample in samples if sample.tokens_used > 0]
        avg_tokens = statistics.mean(tokens_used) if tokens_used else 0.0
        
        # Cost estimation (simplified)
        cost_per_query = self._estimate_cost_per_query(avg_tokens)
        
        # Success rate (assume all samples in evaluation were successful)
        success_rate = 1.0  # Could be calculated from system logs
        error_rate = 0.0
        
        # Throughput (simplified calculation)
        if response_times:
            queries_per_second = 1000.0 / avg_response_time if avg_response_time > 0 else 0.0
        else:
            queries_per_second = 0.0
        
        metrics = SystemPerformanceMetrics(
            average_response_time_ms=avg_response_time,
            percentile_95_response_time_ms=percentile_95,
            response_time_by_agent=response_time_by_agent,
            queries_per_second=queries_per_second,
            average_tokens_per_query=avg_tokens,
            cost_per_query_usd=cost_per_query,
            success_rate=success_rate,
            error_rate=error_rate
        )
        
        self.logger.log_event(
            EventType.PERFORMANCE_METRIC,
            f"Performance evaluation completed: {avg_response_time:.1f}ms average response time",
            extra_data={
                "avg_response_time_ms": avg_response_time,
                "queries_per_second": queries_per_second,
                "sample_count": len(samples)
            }
        )
        
        return metrics
    
    def evaluate_educational_effectiveness(self, 
                                         samples: Optional[List[EvaluationSample]] = None) -> EducationalEffectivenessMetrics:
        """
        Evaluate educational effectiveness of responses.
        
        Args:
            samples: Evaluation samples (uses all if None)
            
        Returns:
            Educational effectiveness metrics
        """
        samples = samples or self.evaluation_samples
        
        if not samples:
            raise ValueError("No evaluation samples available")
        
        # This would typically involve human evaluation or sophisticated NLP analysis
        # For now, provide automated approximations
        
        # SRL phase appropriateness
        srl_appropriateness = self._calculate_srl_phase_appropriateness(samples)
        
        # Conceptual understanding support
        conceptual_support = self._assess_conceptual_understanding_support(samples)
        
        # Practical application support
        practical_support = self._assess_practical_application_support(samples)
        
        # Problem-solving guidance
        problem_solving_guidance = self._assess_problem_solving_guidance(samples)
        
        # Metacognitive support
        metacognitive_support = self._assess_metacognitive_support(samples)
        
        # Scaffolding quality
        scaffolding_quality = self._assess_scaffolding_quality(samples)
        
        metrics = EducationalEffectivenessMetrics(
            conceptual_understanding_support=conceptual_support,
            practical_application_support=practical_support,
            problem_solving_guidance=problem_solving_guidance,
            srl_phase_appropriateness=srl_appropriateness,
            metacognitive_support=metacognitive_support,
            scaffolding_quality=scaffolding_quality,
            feedback_effectiveness=0.75,  # Placeholder
            knowledge_transfer_support=0.70,  # Placeholder
            skill_development_support=0.72   # Placeholder
        )
        
        self.logger.log_event(
            EventType.EDUCATIONAL_OUTCOME,
            f"Educational effectiveness evaluation completed",
            extra_data={
                "srl_appropriateness": srl_appropriateness,
                "conceptual_support": conceptual_support,
                "sample_count": len(samples)
            }
        )
        
        return metrics
    
    def generate_comprehensive_report(self, 
                                    output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Comprehensive evaluation results
        """
        if not self.evaluation_samples:
            raise ValueError("No evaluation samples available for report generation")
        
        # Evaluate all dimensions
        classification_metrics = self.evaluate_classification_performance()
        quality_metrics = self.evaluate_response_quality()
        performance_metrics = self.evaluate_system_performance()
        effectiveness_metrics = self.evaluate_educational_effectiveness()
        
        # Compile comprehensive report
        report = {
            "evaluation_summary": {
                "total_samples": len(self.evaluation_samples),
                "evaluation_timestamp": time.time(),
                "evaluation_config": self.config.__dict__
            },
            "classification_performance": classification_metrics.dict(),
            "response_quality": quality_metrics.dict(),
            "system_performance": performance_metrics.dict(),
            "educational_effectiveness": effectiveness_metrics.dict(),
            "overall_assessment": self._generate_overall_assessment(
                classification_metrics, quality_metrics, performance_metrics, effectiveness_metrics
            ),
            "recommendations": self._generate_recommendations(
                classification_metrics, quality_metrics, performance_metrics, effectiveness_metrics
            )
        }
        
        # Save report if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.log_event(
                EventType.SYSTEM_START,
                f"Evaluation report saved to {output_path}",
                extra_data={"report_path": str(output_path)}
            )
        
        self.evaluation_results = report
        return report
    
    # Helper methods for automated assessment
    
    def _calculate_accuracy_by_attribute(self, samples: List[EvaluationSample], attribute: str) -> Dict[str, float]:
        """Calculate accuracy grouped by sample attribute."""
        grouped_samples = defaultdict(list)
        
        for sample in samples:
            attr_value = getattr(sample, attribute, None)
            if attr_value:
                grouped_samples[attr_value].append(sample)
        
        accuracy_by_attr = {}
        for attr_value, attr_samples in grouped_samples.items():
            correct = sum(1 for s in attr_samples 
                         if s.ground_truth_srl_phase == s.predicted_srl_phase)
            accuracy_by_attr[attr_value] = correct / len(attr_samples)
        
        return accuracy_by_attr
    
    def _calculate_accuracy_by_query_type(self, samples: List[EvaluationSample]) -> Dict[str, float]:
        """Calculate accuracy by query type."""
        query_types = {
            "code_query": [],
            "error_query": [],
            "conceptual_query": [],
            "general_query": []
        }
        
        for sample in samples:
            if sample.has_code_snippet:
                query_types["code_query"].append(sample)
            elif sample.has_error_message:
                query_types["error_query"].append(sample)
            elif any(word in sample.query.lower() for word in ["concept", "understand", "explain"]):
                query_types["conceptual_query"].append(sample)
            else:
                query_types["general_query"].append(sample)
        
        accuracy_by_type = {}
        for query_type, type_samples in query_types.items():
            if type_samples:
                correct = sum(1 for s in type_samples 
                             if s.ground_truth_srl_phase == s.predicted_srl_phase)
                accuracy_by_type[query_type] = correct / len(type_samples)
        
        return accuracy_by_type
    
    def _automated_response_quality_assessment(self, samples: List[EvaluationSample]) -> ResponseQualityMetrics:
        """Automated assessment of response quality."""
        # Simplified automated quality assessment
        quality_scores = []
        
        for sample in samples:
            score = self._assess_single_response_quality(sample.response_content, sample.query)
            quality_scores.append(score)
        
        overall_quality = statistics.mean(quality_scores)
        
        return ResponseQualityMetrics(
            overall_quality=overall_quality,
            quality_dimensions={"automated_assessment": overall_quality},
            excellent_responses_ratio=sum(1 for s in quality_scores if s >= 0.8) / len(quality_scores),
            poor_responses_ratio=sum(1 for s in quality_scores if s <= 0.4) / len(quality_scores),
            educational_appropriateness=overall_quality * 0.9,  # Approximation
            complexity_alignment=overall_quality * 0.85        # Approximation
        )
    
    def _assess_single_response_quality(self, response: str, query: str) -> float:
        """Assess quality of a single response."""
        score = 0.5  # Base score
        
        # Length appropriateness
        if 100 <= len(response) <= 2000:
            score += 0.1
        
        # Contains code examples (if query suggests need for code)
        if "code" in query.lower() or "implement" in query.lower():
            if "```" in response or "def " in response:
                score += 0.15
        
        # Educational language
        educational_terms = ["understand", "learn", "concept", "example", "step"]
        if any(term in response.lower() for term in educational_terms):
            score += 0.15
        
        # Structure (paragraphs, organization)
        if response.count('\n\n') >= 2:  # Multiple paragraphs
            score += 0.1
        
        # Specific to query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words & response_words)
        if overlap >= 3:
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_educational_appropriateness(self, samples: List[EvaluationSample]) -> float:
        """Calculate educational appropriateness score."""
        scores = []
        
        for sample in samples:
            score = 0.5  # Base score
            
            # SRL phase alignment
            if sample.ground_truth_srl_phase == sample.predicted_srl_phase:
                score += 0.3
            
            # Response contains educational elements
            educational_indicators = ["example", "step", "understand", "learn", "practice"]
            if any(indicator in sample.response_content.lower() for indicator in educational_indicators):
                score += 0.2
            
            scores.append(min(1.0, score))
        
        return statistics.mean(scores) if scores else 0.0
    
    def _calculate_complexity_alignment(self, samples: List[EvaluationSample]) -> float:
        """Calculate alignment between response complexity and student level."""
        aligned_count = 0
        total_count = 0
        
        for sample in samples:
            if sample.student_level:
                total_count += 1
                
                # Simple heuristic for complexity alignment
                response = sample.response_content.lower()
                
                if sample.student_level == "beginner":
                    # Should have simple language, basic concepts
                    if any(word in response for word in ["simple", "basic", "easy", "start"]):
                        aligned_count += 1
                elif sample.student_level == "advanced":
                    # Should have advanced concepts
                    if any(word in response for word in ["complex", "advanced", "optimization", "algorithm"]):
                        aligned_count += 1
                else:  # intermediate
                    # Should be balanced
                    aligned_count += 1  # Assume intermediate is generally aligned
        
        return aligned_count / total_count if total_count > 0 else 0.0
    
    def _estimate_cost_per_query(self, avg_tokens: float) -> float:
        """Estimate cost per query based on token usage."""
        # Approximate OpenAI pricing (should be updated with current rates)
        cost_per_1k_tokens = 0.03  # $0.03 per 1K tokens (example rate)
        return (avg_tokens / 1000) * cost_per_1k_tokens
    
    def _calculate_srl_phase_appropriateness(self, samples: List[EvaluationSample]) -> float:
        """Calculate SRL phase appropriateness."""
        appropriate_count = sum(1 for s in samples 
                               if s.ground_truth_srl_phase == s.predicted_srl_phase)
        return appropriate_count / len(samples) if samples else 0.0
    
    def _assess_conceptual_understanding_support(self, samples: List[EvaluationSample]) -> float:
        """Assess support for conceptual understanding."""
        support_scores = []
        
        for sample in samples:
            score = 0.0
            response = sample.response_content.lower()
            
            # Check for explanatory content
            if any(word in response for word in ["explain", "understand", "concept", "theory"]):
                score += 0.3
            
            # Check for examples
            if "example" in response or "for instance" in response:
                score += 0.2
            
            # Check for analogies or comparisons
            if any(word in response for word in ["like", "similar", "compare", "analogy"]):
                score += 0.2
            
            # Check for step-by-step breakdown
            if any(word in response for word in ["step", "first", "next", "then"]):
                score += 0.3
            
            support_scores.append(min(1.0, score))
        
        return statistics.mean(support_scores) if support_scores else 0.0
    
    def _assess_practical_application_support(self, samples: List[EvaluationSample]) -> float:
        """Assess support for practical application."""
        support_scores = []
        
        for sample in samples:
            score = 0.0
            response = sample.response_content.lower()
            
            # Check for code examples
            if "```" in sample.response_content or "def " in response:
                score += 0.4
            
            # Check for implementation guidance
            if any(word in response for word in ["implement", "code", "write", "create"]):
                score += 0.3
            
            # Check for practical tips
            if any(word in response for word in ["tip", "practice", "apply", "use"]):
                score += 0.3
            
            support_scores.append(min(1.0, score))
        
        return statistics.mean(support_scores) if support_scores else 0.0
    
    def _assess_problem_solving_guidance(self, samples: List[EvaluationSample]) -> float:
        """Assess quality of problem-solving guidance."""
        guidance_scores = []
        
        for sample in samples:
            score = 0.0
            response = sample.response_content.lower()
            
            # Check for problem-solving strategies
            if any(word in response for word in ["approach", "strategy", "method", "solve"]):
                score += 0.3
            
            # Check for debugging guidance
            if any(word in response for word in ["debug", "check", "test", "verify"]):
                score += 0.2
            
            # Check for systematic approach
            if any(word in response for word in ["systematic", "step by step", "methodical"]):
                score += 0.3
            
            # Check for error handling
            if any(word in response for word in ["error", "exception", "handle", "catch"]):
                score += 0.2
            
            guidance_scores.append(min(1.0, score))
        
        return statistics.mean(guidance_scores) if guidance_scores else 0.0
    
    def _assess_metacognitive_support(self, samples: List[EvaluationSample]) -> float:
        """Assess support for metacognitive skills."""
        support_scores = []
        
        for sample in samples:
            score = 0.0
            response = sample.response_content.lower()
            
            # Check for reflection prompts
            if any(word in response for word in ["think", "consider", "reflect", "why"]):
                score += 0.3
            
            # Check for learning strategies
            if any(word in response for word in ["strategy", "approach", "plan", "organize"]):
                score += 0.3
            
            # Check for self-monitoring guidance
            if any(word in response for word in ["monitor", "check", "evaluate", "assess"]):
                score += 0.4
            
            support_scores.append(min(1.0, score))
        
        return statistics.mean(support_scores) if support_scores else 0.0
    
    def _assess_scaffolding_quality(self, samples: List[EvaluationSample]) -> float:
        """Assess quality of educational scaffolding."""
        scaffolding_scores = []
        
        for sample in samples:
            score = 0.0
            response = sample.response_content.lower()
            
            # Check for gradual complexity
            if any(word in response for word in ["start", "begin", "first", "simple"]):
                score += 0.2
            
            # Check for building on concepts
            if any(word in response for word in ["build", "extend", "expand", "develop"]):
                score += 0.3
            
            # Check for guided discovery
            if any(word in response for word in ["discover", "explore", "investigate", "try"]):
                score += 0.3
            
            # Check for support withdrawal
            if any(word in response for word in ["independent", "own", "yourself", "practice"]):
                score += 0.2
            
            scaffolding_scores.append(min(1.0, score))
        
        return statistics.mean(scaffolding_scores) if scaffolding_scores else 0.0
    
    def _generate_overall_assessment(self, 
                                   classification_metrics: ClassificationMetrics,
                                   quality_metrics: ResponseQualityMetrics,
                                   performance_metrics: SystemPerformanceMetrics,
                                   effectiveness_metrics: EducationalEffectivenessMetrics) -> Dict[str, Any]:
        """Generate overall system assessment."""
        # Calculate weighted overall score
        weights = {
            "classification": 0.25,
            "quality": 0.30,
            "performance": 0.20,
            "effectiveness": 0.25
        }
        
        overall_score = (
            classification_metrics.accuracy * weights["classification"] +
            quality_metrics.overall_quality * weights["quality"] +
            min(1.0, 3000.0 / performance_metrics.average_response_time_ms) * weights["performance"] +
            effectiveness_metrics.srl_phase_appropriateness * weights["effectiveness"]
        )
        
        # Determine system grade
        if overall_score >= 0.9:
            grade = "Excellent"
        elif overall_score >= 0.8:
            grade = "Good"
        elif overall_score >= 0.7:
            grade = "Satisfactory"
        elif overall_score >= 0.6:
            grade = "Needs Improvement"
        else:
            grade = "Poor"
        
        # Check if system meets minimum requirements
        meets_requirements = (
            classification_metrics.accuracy >= self.config.min_classification_accuracy and
            quality_metrics.overall_quality >= self.config.min_response_quality and
            effectiveness_metrics.srl_phase_appropriateness >= self.config.min_educational_effectiveness and
            performance_metrics.average_response_time_ms <= self.config.max_response_time_ms
        )
        
        return {
            "overall_score": overall_score,
            "grade": grade,
            "meets_requirements": meets_requirements,
            "strengths": self._identify_strengths(classification_metrics, quality_metrics, 
                                                 performance_metrics, effectiveness_metrics),
            "areas_for_improvement": self._identify_improvement_areas(classification_metrics, quality_metrics,
                                                                     performance_metrics, effectiveness_metrics)
        }
    
    def _generate_recommendations(self, 
                                classification_metrics: ClassificationMetrics,
                                quality_metrics: ResponseQualityMetrics,
                                performance_metrics: SystemPerformanceMetrics,
                                effectiveness_metrics: EducationalEffectivenessMetrics) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Classification recommendations
        if classification_metrics.accuracy < self.config.min_classification_accuracy:
            recommendations.append(
                f"Improve SRL classification accuracy (current: {classification_metrics.accuracy:.3f}, "
                f"target: {self.config.min_classification_accuracy:.3f}). Consider refining classification prompts "
                "or adding more training examples."
            )
        
        # Quality recommendations
        if quality_metrics.overall_quality < self.config.min_response_quality:
            recommendations.append(
                f"Enhance response quality (current: {quality_metrics.overall_quality:.3f}, "
                f"target: {self.config.min_response_quality:.3f}). Focus on improving educational content "
                "and response generation strategies."
            )
        
        # Performance recommendations
        if performance_metrics.average_response_time_ms > self.config.max_response_time_ms:
            recommendations.append(
                f"Optimize response time (current: {performance_metrics.average_response_time_ms:.1f}ms, "
                f"target: <{self.config.max_response_time_ms:.1f}ms). Consider caching, "
                "model optimization, or infrastructure improvements."
            )
        
        # Educational effectiveness recommendations
        if effectiveness_metrics.srl_phase_appropriateness < self.config.min_educational_effectiveness:
            recommendations.append(
                f"Improve educational effectiveness (current: {effectiveness_metrics.srl_phase_appropriateness:.3f}, "
                f"target: {self.config.min_educational_effectiveness:.3f}). Enhance agent specialization "
                "and educational content quality."
            )
        
        # Specific improvement areas
        if quality_metrics.poor_responses_ratio > 0.2:
            recommendations.append(
                "High proportion of poor quality responses detected. Review response generation "
                "and quality assurance processes."
            )
        
        if hasattr(performance_metrics, 'cost_per_query_usd') and performance_metrics.cost_per_query_usd > 0.10:
            recommendations.append(
                "Consider cost optimization strategies such as more efficient prompting, "
                "caching, or model selection."
            )
        
        return recommendations
    
    def _identify_strengths(self, *metrics) -> List[str]:
        """Identify system strengths."""
        strengths = []
        
        classification_metrics, quality_metrics, performance_metrics, effectiveness_metrics = metrics
        
        if classification_metrics.accuracy >= 0.9:
            strengths.append("Excellent SRL classification accuracy")
        
        if quality_metrics.overall_quality >= 0.8:
            strengths.append("High response quality across dimensions")
        
        if performance_metrics.average_response_time_ms <= 2000:
            strengths.append("Fast response times")
        
        if effectiveness_metrics.srl_phase_appropriateness >= 0.8:
            strengths.append("Strong educational effectiveness")
        
        if quality_metrics.excellent_responses_ratio >= 0.6:
            strengths.append("High proportion of excellent responses")
        
        return strengths
    
    def _identify_improvement_areas(self, *metrics) -> List[str]:
        """Identify areas for improvement."""
        areas = []
        
        classification_metrics, quality_metrics, performance_metrics, effectiveness_metrics = metrics
        
        if classification_metrics.accuracy < 0.8:
            areas.append("SRL classification accuracy")
        
        if quality_metrics.overall_quality < 0.7:
            areas.append("Response quality")
        
        if performance_metrics.average_response_time_ms > 3000:
            areas.append("Response time optimization")
        
        if effectiveness_metrics.conceptual_understanding_support < 0.7:
            areas.append("Conceptual understanding support")
        
        if effectiveness_metrics.practical_application_support < 0.7:
            areas.append("Practical application guidance")
        
        return areas


if __name__ == "__main__":
    # Evaluation framework test
    try:
        evaluator = EducationalRAGEvaluator()
        
        # Create test samples
        test_samples = [
            EvaluationSample(
                sample_id="test_1",
                query="How do I implement binary search?",
                ground_truth_srl_phase="FORETHOUGHT",
                predicted_srl_phase="FORETHOUGHT",
                classification_confidence=0.95,
                agent_used="implementation",
                response_content="Binary search is an efficient algorithm...",
                student_level="intermediate",
                response_time_ms=1500,
                tokens_used=150
            ),
            EvaluationSample(
                sample_id="test_2",
                query="My code is giving an IndexError",
                ground_truth_srl_phase="PERFORMANCE",
                predicted_srl_phase="PERFORMANCE",
                classification_confidence=0.88,
                agent_used="debugging",
                response_content="IndexError occurs when...",
                student_level="beginner",
                has_error_message=True,
                response_time_ms=1200,
                tokens_used=120
            )
        ]
        
        # Add samples to evaluator
        for sample in test_samples:
            evaluator.add_evaluation_sample(sample)
        
        # Run evaluations
        print("Testing classification evaluation...")
        classification_results = evaluator.evaluate_classification_performance()
        print(f"Classification accuracy: {classification_results.accuracy:.3f}")
        
        print("\nTesting response quality evaluation...")
        quality_results = evaluator.evaluate_response_quality()
        print(f"Overall quality: {quality_results.overall_quality:.3f}")
        
        print("\nTesting performance evaluation...")
        performance_results = evaluator.evaluate_system_performance()
        print(f"Average response time: {performance_results.average_response_time_ms:.1f}ms")
        
        print("\nTesting educational effectiveness evaluation...")
        effectiveness_results = evaluator.evaluate_educational_effectiveness()
        print(f"SRL appropriateness: {effectiveness_results.srl_phase_appropriateness:.3f}")
        
        print("\nGenerating comprehensive report...")
        report = evaluator.generate_comprehensive_report()
        print(f"Overall score: {report['overall_assessment']['overall_score']:.3f}")
        print(f"Grade: {report['overall_assessment']['grade']}")
        
        print("✅ Evaluation framework test completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation framework test failed: {e}")
        import traceback
        traceback.print_exc()
