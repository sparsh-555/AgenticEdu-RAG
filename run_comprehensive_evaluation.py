#!/usr/bin/env python3
"""
Fixed Evaluation Pipeline for Multi-Agent System Routing Accuracy

This script runs comprehensive evaluation using the processed CS1QA dataset:
1. Loads CS1QA data with ground truth SRL labels
2. Feeds queries to the actual multi-agent system
3. Compares system routing decisions vs ground truth
4. Calculates routing accuracy metrics
5. Runs performance tests

NEW FEATURES:
- ✅ Incremental checkpoint saving (every 10 queries by default)
- ✅ Automatic resume from interruptions/power outages
- ✅ Progress tracking and recovery
- ✅ Atomic checkpoint saves to prevent corruption

If interrupted, simply restart the script - it will automatically resume from 
the last saved checkpoint.

Author: AgenticEdu-RAG System
"""

import asyncio
import os
import json
import time
import argparse
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Import system components
from main import UnifiedAgenticEduRAGSystem
from evaluation.performance_tests import PerformanceTester, LoadTestConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RoutingAccuracyEvaluator:
    """Evaluates routing accuracy of the multi-agent system against ground truth."""
    
    def __init__(self, cs1qa_data_path: str, output_dir: Path):
        """
        Initialize routing accuracy evaluator.
        
        Args:
            cs1qa_data_path: Path to processed CS1QA data with ground truth
            output_dir: Output directory for results
        """
        self.cs1qa_data_path = cs1qa_data_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.system = None
        self.routing_results = []
        
    def initialize_system(self):
        """Initialize the multi-agent system."""
        logger.info("🚀 Initializing multi-agent system...")
        
        self.system = UnifiedAgenticEduRAGSystem(
            auto_initialize=True, 
            load_pdfs=False,  # Skip PDF loading for faster initialization
            enable_demo_mode=False
        )
        
        # Wait for system initialization
        max_wait_time = 60  # seconds
        start_time = time.time()
        
        while not self.system.is_initialized:
            if time.time() - start_time > max_wait_time:
                raise RuntimeError("System initialization timeout")
            time.sleep(1)
            
        logger.info("✅ Multi-agent system initialized successfully")
        
    def load_cs1qa_data(self) -> List[Dict[str, Any]]:
        """Load CS1QA data with ground truth labels."""
        logger.info(f"📂 Loading CS1QA data from: {self.cs1qa_data_path}")
        
        with open(self.cs1qa_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter to only samples with ground truth labels
        ground_truth_data = [
            item for item in data 
            if item.get('ground_truth_label') is not None
        ]
        
        logger.info(f"📊 Loaded {len(ground_truth_data)} samples with ground truth (from {len(data)} total)")
        
        return ground_truth_data
        
    def load_checkpoint(self) -> Tuple[List[Dict], List[Dict], int]:
        """Load existing checkpoint if it exists."""
        checkpoint_file = self.output_dir / "evaluation_checkpoint.json"
        
        if checkpoint_file.exists():
            logger.info(f"📂 Found existing checkpoint: {checkpoint_file}")
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                
                successful_tests = checkpoint.get('successful_tests', [])
                failed_tests = checkpoint.get('failed_tests', [])
                last_processed_index = checkpoint.get('last_processed_index', -1)
                
                logger.info(f"🔄 Resuming from query {last_processed_index + 2} (already processed {last_processed_index + 1} queries)")
                logger.info(f"   📊 Loaded {len(successful_tests)} successful tests, {len(failed_tests)} failed tests")
                logger.info(f"   📅 Checkpoint saved: {checkpoint.get('timestamp', 'unknown')}")
                
                return successful_tests, failed_tests, last_processed_index
            except Exception as e:
                logger.error(f"❌ Error loading checkpoint: {e}")
                logger.info("🔄 Starting fresh...")
                
        return [], [], -1
    
    def save_checkpoint(self, successful_tests: List[Dict], failed_tests: List[Dict], last_processed_index: int):
        """Save current progress to checkpoint file."""
        checkpoint_file = self.output_dir / "evaluation_checkpoint.json"
        
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'last_processed_index': last_processed_index,
            'total_processed': last_processed_index + 1,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'checkpoint_version': '1.0'
        }
        
        # Atomic write - write to temp file first, then rename
        temp_file = checkpoint_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2)
            
            # Atomic rename
            temp_file.replace(checkpoint_file)
            logger.info(f"💾 Checkpoint saved: {last_processed_index + 1} queries processed")
            
        except Exception as e:
            logger.error(f"❌ Error saving checkpoint: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def evaluate_routing_accuracy(self, max_samples: Optional[int] = None, checkpoint_interval: int = 10) -> Dict[str, Any]:
        """
        Evaluate routing accuracy by feeding queries to the multi-agent system.
        Supports incremental saving and resuming from checkpoints.
        
        Args:
            max_samples: Maximum number of samples to test (None for all)
            checkpoint_interval: Save progress every N queries (default: 10)
            
        Returns:
            Routing accuracy metrics
        """
        logger.info("🎯 Starting routing accuracy evaluation with checkpoint support...")
        
        # Load ground truth data
        cs1qa_data = self.load_cs1qa_data()
        
        if max_samples:
            cs1qa_data = cs1qa_data[:max_samples]
            logger.info(f"⚠️ Limited to {max_samples} samples for testing")
        
        # Load existing checkpoint
        successful_tests, failed_tests, last_processed_index = self.load_checkpoint()
        
        # Determine starting index
        start_index = last_processed_index + 1
        
        if start_index >= len(cs1qa_data):
            logger.info("✅ All queries already processed! Loading final results...")
        else:
            # Initialize system
            self.initialize_system()
            
            # Test remaining queries
            remaining_queries = len(cs1qa_data) - start_index
            logger.info(f"🔄 Processing {remaining_queries} remaining queries (starting from {start_index + 1}/{len(cs1qa_data)})...")
            
            for i in range(start_index, len(cs1qa_data)):
                sample = cs1qa_data[i]
                logger.info(f"Testing query {i+1}/{len(cs1qa_data)}: {sample['query_id']}")
                
                try:
                    # Extract query data
                    query = sample['question']
                    code_snippet = sample.get('code_snippet')
                    error_message = sample.get('error_message')
                    ground_truth = sample['ground_truth_label']
                    
                    # Feed to multi-agent system
                    start_time = time.perf_counter()
                    
                    response = self.system.process_query(
                        query=query,
                        code_snippet=code_snippet,
                        error_message=error_message
                    )
                    
                    end_time = time.perf_counter()
                    processing_time = (end_time - start_time) * 1000  # ms
                    
                    # Debug: Check response type
                    logger.info(f"Response type: {type(response)}")
                    
                    # Extract system decision from response - handle both dict and object cases
                    if isinstance(response, dict):
                        raw_prediction = response.get('srl_phase', 'unknown')
                        confidence = response.get('classification_confidence', 0.0)
                        agent_used = response.get('agent_used', 'unknown')
                    else:
                        # If it's an object with attributes
                        raw_prediction = getattr(response, 'srl_phase', 'unknown')
                        confidence = getattr(response, 'classification_confidence', 0.0)
                        agent_used = getattr(response, 'agent_used', 'unknown')
                    
                    # Map SRL phases to ground truth labels for evaluation
                    # FORETHOUGHT -> implementation (planning/coding queries)
                    # PERFORMANCE -> debugging (error/fixing queries)
                    srl_to_ground_truth_mapping = {
                        'FORETHOUGHT': 'implementation',
                        'PERFORMANCE': 'debugging',
                        'forethought': 'implementation',
                        'performance': 'debugging'
                    }
                    
                    system_prediction = srl_to_ground_truth_mapping.get(raw_prediction, raw_prediction)
                    
                    # Record result
                    result = {
                        'query_id': sample['query_id'],
                        'query_index': i,  # Add index for tracking
                        'query_preview': query[:100] + "..." if len(query) > 100 else query,
                        'ground_truth': ground_truth,
                        'raw_srl_prediction': raw_prediction,
                        'system_prediction': system_prediction,
                        'correct': ground_truth == system_prediction,
                        'confidence': confidence,
                        'agent_used': agent_used,
                        'processing_time_ms': processing_time,
                        'original_confidence': sample['srl_label']['confidence']
                    }
                    
                    successful_tests.append(result)
                    
                    # Log result
                    status = "✅" if result['correct'] else "❌"
                    logger.info(f"  {status} GT: {ground_truth} → SRL: {raw_prediction} → Mapped: {system_prediction} ({confidence:.2f})")
                    
                except Exception as e:
                    logger.error(f"  ❌ Failed to process query {sample['query_id']}: {e}")
                    failed_tests.append({
                        'query_id': sample['query_id'],
                        'query_index': i,  # Add index for tracking
                        'error': str(e),
                        'ground_truth': sample.get('ground_truth_label', 'unknown')
                    })
                
                # Save checkpoint every N queries
                if (i + 1) % checkpoint_interval == 0 or i == len(cs1qa_data) - 1:
                    self.save_checkpoint(successful_tests, failed_tests, i)
                
                # Brief pause to avoid overwhelming the system
                time.sleep(0.5)
        
        # Calculate metrics
        routing_metrics = self._calculate_routing_metrics(successful_tests, failed_tests)
        
        # Clean up checkpoint file on successful completion
        self.cleanup_checkpoint()
        
        # Store results
        self.routing_results = successful_tests
        
        return routing_metrics
    
    def cleanup_checkpoint(self):
        """Remove checkpoint file after successful completion."""
        checkpoint_file = self.output_dir / "evaluation_checkpoint.json"
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                logger.info("🧹 Checkpoint file cleaned up after successful completion")
            except Exception as e:
                logger.warning(f"⚠️ Could not remove checkpoint file: {e}")
    
    def force_reset_checkpoint(self):
        """Manually remove checkpoint file to start fresh."""
        checkpoint_file = self.output_dir / "evaluation_checkpoint.json"
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                logger.info("🔄 Checkpoint manually cleared - will start fresh")
                return True
            except Exception as e:
                logger.error(f"❌ Could not remove checkpoint file: {e}")
                return False
        else:
            logger.info("ℹ️ No checkpoint file found")
            return True

        
    def _calculate_routing_metrics(self, successful_tests: List[Dict], failed_tests: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive routing accuracy metrics."""
        logger.info("📊 Calculating routing accuracy metrics...")
        
        if not successful_tests:
            return {"error": "No successful tests to analyze"}
        
        # Extract data for metrics
        ground_truth = [test['ground_truth'] for test in successful_tests]
        predictions = [test['system_prediction'] for test in successful_tests]
        confidences = [test['confidence'] for test in successful_tests]
        processing_times = [test['processing_time_ms'] for test in successful_tests]
        
        # Calculate accuracy metrics
        accuracy = accuracy_score(ground_truth, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truth, predictions, 
            labels=['implementation', 'debugging'], 
            average=None,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(ground_truth, predictions, labels=['implementation', 'debugging'])
        
        # Detailed classification report
        class_report = classification_report(
            ground_truth, predictions,
            labels=['implementation', 'debugging'],
            output_dict=True,
            zero_division=0
        )
        
        # Performance statistics
        avg_processing_time = sum(processing_times) / len(processing_times)
        avg_confidence = sum(confidences) / len(confidences)
        
        # Success statistics
        total_attempts = len(successful_tests) + len(failed_tests)
        success_rate = len(successful_tests) / total_attempts if total_attempts > 0 else 0
        
        # Phase distribution analysis
        gt_dist = {'implementation': ground_truth.count('implementation'), 'debugging': ground_truth.count('debugging')}
        pred_dist = {'implementation': predictions.count('implementation'), 'debugging': predictions.count('debugging')}
        
        # Compile metrics
        metrics = {
            "routing_accuracy": {
                "overall_accuracy": float(accuracy),
                "implementation_precision": float(precision[0]),
                "implementation_recall": float(recall[0]),
                "implementation_f1": float(f1[0]),
                "debugging_precision": float(precision[1]),
                "debugging_recall": float(recall[1]),
                "debugging_f1": float(f1[1])
            },
            
            "confusion_matrix": {
                "matrix": cm.tolist(),
                "labels": ["implementation", "debugging"],
                "true_positives": {"implementation": int(cm[0][0]), "debugging": int(cm[1][1])},
                "false_positives": {"implementation": int(cm[1][0]), "debugging": int(cm[0][1])}
            },
            
            "system_performance": {
                "total_queries_attempted": total_attempts,
                "successful_queries": len(successful_tests),
                "failed_queries": len(failed_tests),
                "success_rate": float(success_rate),
                "avg_processing_time_ms": float(avg_processing_time),
                "avg_confidence": float(avg_confidence)
            },
            
            "phase_distribution": {
                "ground_truth": gt_dist,
                "predictions": pred_dist,
                "distribution_match": gt_dist == pred_dist
            },
            
            "detailed_classification_report": class_report,
            "failed_queries": failed_tests
        }
        
        return metrics

async def run_comprehensive_evaluation(checkpoint_interval: int = 10):
    """Run comprehensive evaluation including routing accuracy and performance tests."""
    
    # Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set")
        return
    
    # Paths - check multiple possible locations for processed data
    possible_data_paths = [
        "cs1qa_full_processed/cs1qa_srl_labeled_with_ground_truth.json",
        "test_cs1qa_accuracy/results/cs1qa_srl_labeled_with_ground_truth.json",
        "evaluation_results/cs1qa_processed/cs1qa_srl_labeled_with_ground_truth.json"
    ]
    
    cs1qa_data_path = None
    for path in possible_data_paths:
        if Path(path).exists():
            cs1qa_data_path = path
            break
    
    if not cs1qa_data_path:
        logger.error("❌ No processed CS1QA data found. Please run one of:")
        logger.error("   - process_full_cs1qa_dataset.py (for full dataset)")
        logger.error("   - test_accuracy_pipeline.py (for test dataset)")
        return
    
    output_dir = Path("comprehensive_evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🎯 Starting Comprehensive Multi-Agent System Evaluation")
    logger.info("=" * 60)
    start_time = datetime.now()
    
    try:
        # ============================================================================
        # Phase 1: Routing Accuracy Evaluation
        # ============================================================================
        
        logger.info("Phase 1: Routing Accuracy Evaluation")
        
        evaluator = RoutingAccuracyEvaluator(cs1qa_data_path, output_dir)
        
        # Run full evaluation on all samples with checkpoint support
        # Saves progress every N queries and can resume from interruptions
        routing_metrics = evaluator.evaluate_routing_accuracy(max_samples=None, checkpoint_interval=checkpoint_interval)
        
        if 'error' in routing_metrics:
            logger.error(f"Routing evaluation failed: {routing_metrics['error']}")
        else:
            # Print routing results
            acc = routing_metrics['routing_accuracy']['overall_accuracy']
            logger.info(f"✅ Routing Accuracy Results:")
            logger.info(f"   Overall Accuracy: {acc:.3f}")
            logger.info(f"   Success Rate: {routing_metrics['system_performance']['success_rate']:.3f}")
            
            # Check if meets research criteria (80% accuracy)
            if acc >= 0.80:
                logger.info("🎉 SUCCESS: Routing accuracy meets ≥80% research target!")
            else:
                logger.info(f"⚠️ NEEDS IMPROVEMENT: Routing accuracy {acc:.1%} < 80% target")
        
        # ============================================================================
        # Phase 2: Performance Testing  
        # ============================================================================
        
        logger.info("\nPhase 2: System Performance Testing")
        
        performance_tester = PerformanceTester(
            openai_api_key=OPENAI_API_KEY,
            output_dir=output_dir / "performance_tests"
        )
        
        # Run performance tests
        logger.info("Running response time tests...")
        response_time_results = await performance_tester.run_response_time_tests(num_iterations=5)
        
        logger.info("Running concurrency tests...")
        load_config = LoadTestConfig(min_users=1, max_users=8, step_size=2)
        concurrency_results = await performance_tester.run_concurrency_tests(load_config)
        
        logger.info("Running scalability tests...")
        scalability_results = await performance_tester.run_scalability_tests(query_loads=[1, 3, 5])
        
        # Generate performance report
        performance_report = performance_tester.generate_performance_report()
        performance_tester.save_results("comprehensive_evaluation")
        
        # ============================================================================
        # Phase 3: Generate Comprehensive Report
        # ============================================================================
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        comprehensive_report = {
            "evaluation_metadata": {
                "timestamp": end_time.isoformat(),
                "duration_minutes": duration.total_seconds() / 60,
                "cs1qa_data_source": cs1qa_data_path
            },
            "routing_accuracy": routing_metrics,
            "performance_metrics": performance_report,
            "research_criteria_assessment": {
                "routing_accuracy_target_80_percent": routing_metrics.get('routing_accuracy', {}).get('overall_accuracy', 0) >= 0.80,
                "response_time_target_3_seconds": performance_report.get('response_time_analysis', {}).get('meets_target_3s', False),
                "system_stability": routing_metrics.get('system_performance', {}).get('success_rate', 0) >= 0.95
            }
        }
        
        # Save comprehensive report
        report_file = output_dir / f"comprehensive_evaluation_report_{end_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
        
        # Print final summary
        logger.info("\n🎉 COMPREHENSIVE EVALUATION COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"⏱️ Total Duration: {duration}")
        logger.info(f"📁 Results Directory: {output_dir}")
        logger.info(f"📄 Comprehensive Report: {report_file}")
        
        # Research criteria summary
        criteria = comprehensive_report["research_criteria_assessment"]
        logger.info("\n🎯 Research Criteria Assessment:")
        for criterion, met in criteria.items():
            status = "✅ PASS" if met else "❌ FAIL"
            logger.info(f"   {criterion}: {status}")
        
        overall_success = all(criteria.values())
        if overall_success:
            logger.info("\n🏆 ALL RESEARCH CRITERIA MET!")
        else:
            logger.info("\n⚠️ Some research criteria need improvement")
            
    except Exception as e:
        logger.error(f"❌ Comprehensive evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive evaluation with checkpoint support")
    parser.add_argument("--checkpoint-interval", type=int, default=10, 
                       help="Save checkpoint every N queries (default: 10)")
    parser.add_argument("--reset-checkpoint", action="store_true",
                       help="Clear existing checkpoint and start fresh")
    
    args = parser.parse_args()
    
    if args.reset_checkpoint:
        # Quick reset without running evaluation
        output_dir = Path("comprehensive_evaluation_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluator = RoutingAccuracyEvaluator("", output_dir)  # dummy path for reset
        if evaluator.force_reset_checkpoint():
            print("✅ Checkpoint cleared successfully!")
        else:
            print("❌ Failed to clear checkpoint")
        exit(0)
    
    asyncio.run(run_comprehensive_evaluation(checkpoint_interval=args.checkpoint_interval))
