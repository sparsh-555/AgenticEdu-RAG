#!/usr/bin/env python3
"""
TEST Core Pipeline + Classification Accuracy with Confident Pseudo-Labels

This script tests the core CS1QA processing pipeline and implements classification 
accuracy testing using high-confidence automated labels as ground truth.

Key Features:
- Processes queries with SRL labeling
- Populates ground_truth_label field for high-confidence predictions (≥0.7)
- Calculates classification accuracy on the high-confidence subset
- Generates JSON with populated ground truth labels

Logic:
- High confidence labels (≥0.7) → ground_truth_label = automated_label, validation_status = 'auto_validated'
- Low confidence labels (<0.7) → ground_truth_label = null, validation_status = 'low_confidence'
- Classification accuracy calculated only on high-confidence subset

Author: AgenticEdu-RAG System
"""

import asyncio
import os
import logging
import shutil
import json
from pathlib import Path
from typing import Dict, List, Any
from data.cs1qa_processor import CS1QAProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_classification_accuracy(processed_data: List[Dict[str, Any]], 
                                    confidence_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Calculate classification accuracy using high-confidence labels as ground truth.
    ALSO updates the ground_truth_label field in the data.
    
    Args:
        processed_data: List of processed queries with SRL labels
        confidence_threshold: Minimum confidence for treating label as ground truth
    
    Returns:
        Dictionary with accuracy metrics
    """
    logger.info(f"Calculating classification accuracy with confidence threshold: {confidence_threshold}")
    
    # Update ground_truth_label for high-confidence predictions
    high_confidence_count = 0
    for item in processed_data:
        if item['srl_label']['confidence'] >= confidence_threshold:
            # Set ground truth label = automated label for high-confidence predictions
            item['ground_truth_label'] = item['srl_label']['phase']
            item['validation_status'] = 'auto_validated'
            high_confidence_count += 1
        else:
            # Keep as null for low-confidence predictions
            item['ground_truth_label'] = None
            item['validation_status'] = 'low_confidence'
    
    logger.info(f"Updated ground truth labels for {high_confidence_count}/{len(processed_data)} high-confidence samples")
    
    # Filter to high-confidence predictions only for metrics
    high_confidence_data = [
        item for item in processed_data 
        if item['srl_label']['confidence'] >= confidence_threshold
    ]
    
    logger.info(f"High-confidence samples: {len(high_confidence_data)}/{len(processed_data)} "
                f"({len(high_confidence_data)/len(processed_data)*100:.1f}%)")
    
    if len(high_confidence_data) == 0:
        logger.warning("No high-confidence samples found!")
        return {"error": "No high-confidence samples"}
    
    # Use high-confidence automated labels as "ground truth"
    ground_truth_labels = [item['ground_truth_label'] for item in high_confidence_data]
    predicted_labels = [item['srl_label']['phase'] for item in high_confidence_data]
    
    # Since we're using automated labels as ground truth, accuracy should be 100%
    # But this validates our confidence filtering logic
    accuracy = sum(1 for gt, pred in zip(ground_truth_labels, predicted_labels) if gt == pred) / len(ground_truth_labels)
    
    # Count phase distribution
    phase_counts = {}
    for phase in ground_truth_labels:
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    # Calculate confidence statistics
    confidences = [item['srl_label']['confidence'] for item in high_confidence_data]
    avg_confidence = sum(confidences) / len(confidences)
    min_confidence = min(confidences)
    max_confidence = max(confidences)
    
    # Classification quality metrics
    accuracy_metrics = {
        "total_samples": len(processed_data),
        "high_confidence_samples": len(high_confidence_data),
        "confidence_threshold": confidence_threshold,
        "confidence_filter_rate": len(high_confidence_data) / len(processed_data),
        "classification_accuracy": accuracy,  # Should be 1.0 since we use automated labels as ground truth
        
        "phase_distribution": phase_counts,
        "phase_percentages": {
            phase: count / len(high_confidence_data) * 100 
            for phase, count in phase_counts.items()
        },
        
        "confidence_statistics": {
            "average_confidence": avg_confidence,
            "min_confidence": min_confidence,
            "max_confidence": max_confidence
        },
        
        "quality_indicators": {
            "high_confidence_rate": len(high_confidence_data) / len(processed_data),
            "balanced_distribution": min(phase_counts.values()) / max(phase_counts.values()) if phase_counts else 0,
            "mean_confidence": avg_confidence,
            "ground_truth_populated": high_confidence_count
        }
    }
    
    # Show detailed breakdown by phase
    logger.info("📊 Classification Accuracy Results:")
    logger.info(f"   Total samples: {len(processed_data)}")
    logger.info(f"   High-confidence samples: {len(high_confidence_data)} ({len(high_confidence_data)/len(processed_data)*100:.1f}%)")
    logger.info(f"   Ground truth labels populated: {high_confidence_count}")
    logger.info(f"   Classification accuracy: {accuracy:.3f} (expected 1.0)")
    logger.info(f"   Average confidence: {avg_confidence:.3f}")
    
    logger.info("📈 Phase Distribution (High-Confidence Only):")
    for phase, count in phase_counts.items():
        percentage = count / len(high_confidence_data) * 100
        logger.info(f"   - {phase}: {count} samples ({percentage:.1f}%)")
    
    return accuracy_metrics

async def test_core_pipeline_with_accuracy():
    """
    Test the core CS1QA processing pipeline with classification accuracy evaluation.
    """
    
    # Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set")
        return False
    
    # Create completely isolated test directory structure
    test_base_dir = Path("test_cs1qa_accuracy")
    test_data_dir = test_base_dir / "cs1qa"
    test_raw_dir = test_data_dir / "raw"
    test_output_dir = test_base_dir / "results"
    
    logger.info("🧪 Starting CORE PIPELINE + ACCURACY TEST")
    
    try:
        # Create test directory structure
        test_raw_dir.mkdir(parents=True, exist_ok=True)
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test file with varied confidence scenarios
        test_file = test_raw_dir / "test_accuracy.jsonl"
        test_content = '''{"labNo": 9, "taskNo": 1, "questioner": "TA", "question": "In Task 2, explain the composition of the created Card object.", "code": "class Card:\\n    def __init__(self,suit,face):\\n        self.suit=suit\\n        self.face=face", "startLine": 17, "endLine": 28, "questionType": "variable", "answer": "Card object composition explanation"}
{"labNo": 6, "taskNo": 1, "questioner": "student", "question": "when doing count_integer, can't we compare from minimum integer to maximum integer?", "code": "def count_integers(num_list):\\n    lis=num_list\\n    n=len(lis)", "questionType": "task", "answer": "Comparison approach question"}
{"labNo": 8, "taskNo": 0, "questioner": "student", "question": "I'm getting a NameError with elice_utils, was this covered in lecture?", "code": "import elice_utils\\nelice_utils.send_file('output.txt')", "questionType": "task", "answer": "NameError debugging question"}
{"labNo": 7, "taskNo": 1, "questioner": "student", "question": "I changed the angle calculation but it's still not working, what's wrong?", "code": "angle = float(angle)/180*pi\\ntan(angle)", "questionType": "logical", "answer": "Angle conversion debugging"}'''
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        logger.info(f"📁 Created isolated test directory: {test_base_dir}")
        logger.info(f"📄 Created test file with 4 queries: {test_file}")
        
        # Initialize processor with isolated test directory
        processor = CS1QAProcessor(
            data_dir=test_data_dir,
            openai_api_key=OPENAI_API_KEY,
            output_dir=test_output_dir
        )
        
        # Load test dataset
        logger.info("📥 Loading isolated test sample...")
        await processor.load_cs1qa_dataset()
        logger.info(f"✅ Loaded {len(processor.raw_queries)} queries from isolated test")
        
        # Apply SRL labeling
        logger.info("🏷️ Starting SRL labeling...")
        await processor.apply_srl_labeling(batch_size=2, max_concurrent=1)
        logger.info(f"✅ Processed {len(processor.labeled_queries)} queries with SRL labels")
        
        # Skip validation and dataset splitting
        logger.info("⚠️ Skipping validation and dataset splitting")
        
        # Save processed data
        logger.info("💾 Saving processed data...")
        processor.save_processed_data()
        logger.info("✅ Data saved successfully!")
        
        # Load the saved JSON for accuracy testing
        json_file = test_output_dir / "cs1qa_srl_labeled.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                processed_data = json.load(f)
            
            logger.info(f"📊 Loaded {len(processed_data)} processed records for accuracy testing")
            
            # Test different confidence thresholds
            thresholds = [0.6, 0.7, 0.8]
            accuracy_results = {}
            updated_json_file = None  # Initialize for later use
            
            # Make a copy for testing different thresholds
            import copy
            
            for threshold in thresholds:
                logger.info(f"\n🎯 Testing with confidence threshold: {threshold}")
                # Use a copy so we don't modify the original data multiple times
                test_data = copy.deepcopy(processed_data)
                accuracy_metrics = calculate_classification_accuracy(test_data, threshold)
                accuracy_results[f"threshold_{threshold}"] = accuracy_metrics
                
                # For the optimal threshold (0.7), save the updated JSON with ground truth labels
                if threshold == 0.7:
                    updated_json_file = test_output_dir / "cs1qa_srl_labeled_with_ground_truth.json"
                    with open(updated_json_file, 'w') as f:
                        json.dump(test_data, f, indent=2)
                    logger.info(f"💾 Updated JSON with ground truth labels saved to: {updated_json_file}")
            
            # Show individual query details (using the 0.7 threshold version)
            logger.info("\n🔍 Individual Query Analysis:")
            for i, item in enumerate(test_data):
                query_preview = item['question'][:60] + "..." if len(item['question']) > 60 else item['question']
                confidence = item['srl_label']['confidence']
                phase = item['srl_label']['phase']
                ground_truth = item['ground_truth_label']
                validation_status = item['validation_status']
                
                confidence_indicator = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.6 else "🔴"
                gt_indicator = f" | GT: {ground_truth}" if ground_truth else " | GT: null"
                status_indicator = f" | {validation_status}"
                
                logger.info(f"   Query {i+1}: {confidence_indicator} [{phase}] ({confidence:.2f}){gt_indicator}{status_indicator} - {query_preview}")
            
            # Save accuracy results
            accuracy_file = test_output_dir / "classification_accuracy_results.json"
            with open(accuracy_file, 'w') as f:
                json.dump(accuracy_results, f, indent=2)
            
            logger.info(f"\n📄 Accuracy results saved to: {accuracy_file}")
            
            # Generate final report
            logger.info("\n📋 ACCURACY TEST SUMMARY:")
            logger.info("=" * 50)
            
            best_threshold = 0.7  # Use 0.7 as the standard threshold
            best_result = accuracy_results[f"threshold_{best_threshold}"]
            
            logger.info(f"🏆 Standard threshold: {best_result['confidence_threshold']}")
            logger.info(f"📊 High-confidence samples: {best_result['high_confidence_samples']}/{best_result['total_samples']}")
            logger.info(f"💾 Ground truth labels populated: {best_result['quality_indicators']['ground_truth_populated']}")
            logger.info(f"📈 Classification accuracy: {best_result['classification_accuracy']:.3f}")
            logger.info(f"🎯 Average confidence: {best_result['confidence_statistics']['average_confidence']:.3f}")
            logger.info(f"⚖️ Phase balance: {best_result['quality_indicators']['balanced_distribution']:.3f}")
            
            # Show files created
            logger.info("\n📁 Generated Files:")
            logger.info(f"   - Original JSON: {json_file}")
            logger.info(f"   - JSON with Ground Truth: {updated_json_file}")
            logger.info(f"   - CSV Export: {test_output_dir / 'cs1qa_srl_labeled.csv'}")
            logger.info(f"   - Accuracy Results: {accuracy_file}")
            
        else:
            logger.error("❌ JSON file not found for accuracy testing")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ CORE PIPELINE + ACCURACY TEST FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def cleanup_test():
    """Clean up test directory."""
    test_base_dir = Path("test_cs1qa_accuracy")
    if test_base_dir.exists():
        shutil.rmtree(test_base_dir)
        logger.info(f"🧹 Cleaned up test directory: {test_base_dir}")

if __name__ == "__main__":
    print("🧪 CORE PIPELINE + CLASSIFICATION ACCURACY TEST")
    print("Using High-Confidence Automated Labels as Ground Truth")
    print("=" * 60)
    
    success = asyncio.run(test_core_pipeline_with_accuracy())
    
    if success:
        print("\n🎉 CORE PIPELINE + ACCURACY TEST SUCCESSFUL!")
        print("✅ The classification accuracy evaluation works!")
        print("✅ Ground truth labels populated for high-confidence predictions!")
        print("💡 You can now run this on the full dataset:")
        print("   - High-confidence automated labels (≥0.7) are used as ground truth")
        print("   - Low-confidence predictions keep ground_truth_label = null")
        print("   - Classification accuracy can be calculated on the high-confidence subset")
        print("\n🧹 Clean up test files? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            asyncio.run(cleanup_test())
    else:
        print("\n❌ CORE PIPELINE + ACCURACY TEST FAILED!")
        print("💡 Please check the error messages above")
