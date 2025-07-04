#!/usr/bin/env python3
"""
Classification Accuracy Calculator for CS1QA Dataset

This script calculates classification accuracy metrics directly from the 
processed CS1QA JSON with populated ground truth labels.

Works with the JSON format from cs1qa_processor.py and calculates:
- Overall accuracy
- Precision, Recall, F1-score per class
- Confusion matrix
- Confidence statistics

Author: AgenticEdu-RAG System
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_processed_data(json_file_path: str) -> List[Dict[str, Any]]:
    """Load processed CS1QA data from JSON file."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def calculate_classification_metrics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive classification accuracy metrics.
    
    Args:
        data: List of processed queries with ground truth and predicted labels
        
    Returns:
        Dictionary with accuracy metrics
    """
    logger.info("Calculating classification accuracy metrics...")
    
    # Filter to samples with ground truth labels (high-confidence only)
    valid_samples = [
        item for item in data 
        if item.get('ground_truth_label') is not None
    ]
    
    if not valid_samples:
        logger.error("No samples with ground truth labels found!")
        return {"error": "No ground truth labels"}
    
    logger.info(f"Analyzing {len(valid_samples)} samples with ground truth labels")
    
    # Extract labels and predictions
    ground_truth = [item['ground_truth_label'] for item in valid_samples]
    predictions = [item['srl_label']['phase'] for item in valid_samples]
    confidences = [item['srl_label']['confidence'] for item in valid_samples]
    
    # Calculate basic accuracy metrics
    accuracy = accuracy_score(ground_truth, predictions)
    
    # Calculate per-class metrics
    labels = ['implementation', 'debugging']
    precision = precision_score(ground_truth, predictions, labels=labels, average=None, zero_division=0)
    recall = recall_score(ground_truth, predictions, labels=labels, average=None, zero_division=0)
    f1 = f1_score(ground_truth, predictions, labels=labels, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(ground_truth, predictions, labels=labels)
    
    # Detailed classification report
    class_report = classification_report(
        ground_truth, predictions, 
        labels=labels, 
        output_dict=True,
        zero_division=0
    )
    
    # Confidence statistics
    confidence_stats = {
        'mean_confidence': np.mean(confidences),
        'std_confidence': np.std(confidences),
        'min_confidence': np.min(confidences),
        'max_confidence': np.max(confidences)
    }
    
    # Confidence vs accuracy analysis
    correct_predictions = [gt == pred for gt, pred in zip(ground_truth, predictions)]
    high_conf_mask = np.array(confidences) >= 0.8
    high_conf_accuracy = np.mean([correct_predictions[i] for i in range(len(correct_predictions)) if high_conf_mask[i]]) if high_conf_mask.any() else 0
    
    # Phase distribution analysis
    phase_distribution = {
        'ground_truth': {
            'implementation': ground_truth.count('implementation'),
            'debugging': ground_truth.count('debugging')
        },
        'predictions': {
            'implementation': predictions.count('implementation'),
            'debugging': predictions.count('debugging')
        }
    }
    
    # Compile results
    metrics = {
        'overall_metrics': {
            'accuracy': float(accuracy),
            'total_samples': len(valid_samples),
            'correct_predictions': int(accuracy * len(valid_samples))
        },
        
        'per_class_metrics': {
            'implementation': {
                'precision': float(precision[0]),
                'recall': float(recall[0]),
                'f1_score': float(f1[0])
            },
            'debugging': {
                'precision': float(precision[1]),
                'recall': float(recall[1]),
                'f1_score': float(f1[1])
            }
        },
        
        'confusion_matrix': {
            'matrix': cm.tolist(),
            'labels': labels,
            'true_positives': {
                'implementation': int(cm[0][0]),
                'debugging': int(cm[1][1])
            },
            'false_positives': {
                'implementation': int(cm[1][0]),
                'debugging': int(cm[0][1])
            }
        },
        
        'confidence_analysis': {
            **confidence_stats,
            'high_confidence_accuracy': float(high_conf_accuracy),
            'high_confidence_samples': int(high_conf_mask.sum())
        },
        
        'phase_distribution': phase_distribution,
        
        'sklearn_classification_report': class_report
    }
    
    return metrics

def print_metrics_report(metrics: Dict[str, Any]):
    """Print a comprehensive metrics report."""
    
    logger.info("\n🎯 CLASSIFICATION ACCURACY REPORT")
    logger.info("=" * 50)
    
    # Overall metrics
    overall = metrics['overall_metrics']
    logger.info(f"📊 Overall Accuracy: {overall['accuracy']:.3f} ({overall['correct_predictions']}/{overall['total_samples']})")
    
    # Per-class metrics
    logger.info("\n📈 Per-Class Performance:")
    for phase in ['implementation', 'debugging']:
        class_metrics = metrics['per_class_metrics'][phase]
        logger.info(f"   {phase.title()}:")
        logger.info(f"     - Precision: {class_metrics['precision']:.3f}")
        logger.info(f"     - Recall:    {class_metrics['recall']:.3f}")
        logger.info(f"     - F1-Score:  {class_metrics['f1_score']:.3f}")
    
    # Confusion matrix
    cm = metrics['confusion_matrix']
    logger.info(f"\n🔢 Confusion Matrix:")
    logger.info(f"                 Predicted")
    logger.info(f"               Impl  Debug")
    logger.info(f"   Actual Impl   {cm['matrix'][0][0]:3d}   {cm['matrix'][0][1]:3d}")
    logger.info(f"         Debug   {cm['matrix'][1][0]:3d}   {cm['matrix'][1][1]:3d}")
    
    # Confidence analysis
    conf = metrics['confidence_analysis']
    logger.info(f"\n🎯 Confidence Analysis:")
    logger.info(f"   Mean confidence: {conf['mean_confidence']:.3f}")
    logger.info(f"   High-confidence (≥0.8) accuracy: {conf['high_confidence_accuracy']:.3f}")
    logger.info(f"   High-confidence samples: {conf['high_confidence_samples']}/{overall['total_samples']}")
    
    # Phase distribution
    dist = metrics['phase_distribution']
    logger.info(f"\n📊 Phase Distribution:")
    logger.info(f"   Ground Truth - Implementation: {dist['ground_truth']['implementation']}, Debugging: {dist['ground_truth']['debugging']}")
    logger.info(f"   Predictions  - Implementation: {dist['predictions']['implementation']}, Debugging: {dist['predictions']['debugging']}")

def main():
    """Main function to run classification accuracy analysis."""
    
    # Default test file path (can be modified)
    default_json_path = "test_cs1qa_accuracy/results/cs1qa_srl_labeled_with_ground_truth.json"
    
    # Check if test file exists
    if not Path(default_json_path).exists():
        logger.error(f"Test JSON file not found: {default_json_path}")
        logger.info("Please run test_accuracy_pipeline.py first to generate the test data")
        return
    
    try:
        # Load data
        logger.info(f"Loading data from: {default_json_path}")
        data = load_processed_data(default_json_path)
        logger.info(f"Loaded {len(data)} total samples")
        
        # Calculate metrics
        metrics = calculate_classification_metrics(data)
        
        if 'error' in metrics:
            logger.error(f"Metrics calculation failed: {metrics['error']}")
            return
        
        # Print report
        print_metrics_report(metrics)
        
        # Save detailed metrics
        output_file = Path(default_json_path).parent / "classification_accuracy_detailed.json"
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"\n💾 Detailed metrics saved to: {output_file}")
        
        # Summary
        accuracy = metrics['overall_metrics']['accuracy']
        if accuracy >= 0.95:
            logger.info("🎉 EXCELLENT: Classification accuracy ≥ 95%")
        elif accuracy >= 0.90:
            logger.info("✅ VERY GOOD: Classification accuracy ≥ 90%")
        elif accuracy >= 0.80:
            logger.info("👍 GOOD: Classification accuracy ≥ 80%")
        else:
            logger.info("⚠️ NEEDS IMPROVEMENT: Classification accuracy < 80%")
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
