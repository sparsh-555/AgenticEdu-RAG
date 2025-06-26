#!/usr/bin/env python3
"""
Example Usage Script for Agentic Edu-RAG Evaluation Framework

This script demonstrates how to use the CS1QA processor and performance tester
together for comprehensive system evaluation.

Run this script to:
1. Process CS1QA dataset with SRL labeling
2. Run performance tests on the system
3. Generate comprehensive evaluation reports

Author: Agentic Edu-RAG System
"""

import asyncio
import os
from pathlib import Path
import logging
from datetime import datetime

from evaluation.cs1qa_processor import CS1QAProcessor
from evaluation.performance_tests import PerformanceTester, LoadTestConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_comprehensive_evaluation():
    """
    Run comprehensive evaluation including dataset processing and performance testing.
    """
    
    # Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set")
        return
    
    # Paths
    data_dir = Path("data/cs1qa")
    output_dir = Path("evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting comprehensive evaluation of Agentic Edu-RAG system")
    start_time = datetime.now()
    
    # ============================================================================
    # Phase 1: CS1QA Dataset Processing and SRL Labeling
    # ============================================================================
    
    logger.info("Phase 1: Processing CS1QA dataset with SRL labeling")
    
    try:
        processor = CS1QAProcessor(
            data_dir=data_dir,
            openai_api_key=OPENAI_API_KEY,
            output_dir=output_dir / "cs1qa_processed"
        )
        
        # Load CS1QA dataset
        await processor.load_cs1qa_dataset()
        logger.info(f"Loaded {len(processor.raw_queries)} queries from CS1QA dataset")
        
        # Apply SRL labeling (using smaller batches for demo)
        await processor.apply_srl_labeling(batch_size=3, max_concurrent=2)
        
        # Validate labeling consistency
        validation_results = processor.validate_labeling_consistency(sample_size=50)
        
        # Create evaluation datasets
        datasets = processor.create_evaluation_datasets()
        
        # Save processed data
        processor.save_processed_data()
        
        # Generate processing report
        processing_report = processor.generate_processing_report()
        
        logger.info("CS1QA processing completed successfully!")
        logger.info(f"Classification success rate: {processing_report['processing_summary']['classification_success_rate']:.2%}")
        logger.info(f"Mean confidence: {validation_results['mean_confidence']:.3f}")
        
    except Exception as e:
        logger.error(f"CS1QA processing failed: {e}")
        return
    
    # ============================================================================
    # Phase 2: System Performance Testing
    # ============================================================================
    
    logger.info("Phase 2: Running system performance tests")
    
    try:
        tester = PerformanceTester(
            openai_api_key=OPENAI_API_KEY,
            output_dir=output_dir / "performance_tests"
        )
        
        # Response time tests
        logger.info("Running response time tests...")
        response_time_results = await tester.run_response_time_tests(
            num_iterations=5,
            warm_up_iterations=2
        )
        
        # Concurrency tests (smaller scale for demo)
        logger.info("Running concurrency tests...")
        load_config = LoadTestConfig(
            min_users=1,
            max_users=8,
            step_size=2,
            duration_per_step=20
        )
        concurrency_results = await tester.run_concurrency_tests(load_config)
        
        # Scalability tests
        logger.info("Running scalability tests...")
        scalability_results = await tester.run_scalability_tests(
            query_loads=[1, 3, 5, 10],
            queries_per_load=5
        )
        
        # API usage analysis (shorter duration for demo)
        logger.info("Running API usage analysis...")
        api_usage_results = await tester.run_api_usage_analysis(
            analysis_duration=120  # 2 minutes
        )
        
        # Generate comprehensive performance report
        performance_report = tester.generate_performance_report()
        
        # Save results and create visualizations
        results_file, report_file = tester.save_results("comprehensive_evaluation")
        viz_file = tester.create_performance_visualizations()
        
        logger.info("Performance testing completed successfully!")
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Visualizations saved to: {viz_file}")
        
        # Print key performance metrics
        if response_time_results:
            avg_response_time = sum(m.response_time for m in response_time_results.values()) / len(response_time_results)
            logger.info(f"Average response time: {avg_response_time:.3f}s")
        
        if api_usage_results:
            cost_per_1000 = api_usage_results['cost_projections']['cost_per_1000_queries']
            logger.info(f"Estimated cost per 1000 queries: ${cost_per_1000:.4f}")
        
    except Exception as e:
        logger.error(f"Performance testing failed: {e}")
        return
    
    # ============================================================================
    # Phase 3: Generate Comprehensive Evaluation Summary
    # ============================================================================
    
    logger.info("Phase 3: Generating comprehensive evaluation summary")
    
    evaluation_summary = {
        "evaluation_metadata": {
            "timestamp": datetime.now().isoformat(),
            "duration_minutes": (datetime.now() - start_time).total_seconds() / 60,
            "evaluation_type": "comprehensive_system_evaluation"
        },
        "dataset_processing": {
            "total_queries_processed": processing_report['processing_summary']['total_queries_loaded'],
            "classification_success_rate": processing_report['processing_summary']['classification_success_rate'],
            "mean_classification_confidence": validation_results['mean_confidence'],
            "phase_distribution": processing_report['phase_distribution']
        },
        "performance_metrics": {
            "meets_response_time_target": performance_report['response_time_analysis'].get('meets_target_3s', False),
            "max_concurrent_users_tested": performance_report['concurrency_analysis'].get('max_concurrent_users_tested', 0),
            "optimal_concurrency_level": performance_report['concurrency_analysis'].get('optimal_concurrency_level', 1),
            "scaling_efficiency": scalability_results.scaling_efficiency if 'scalability' in tester.test_results else 0
        },
        "cost_analysis": {
            "estimated_cost_per_1000_queries": api_usage_results['cost_projections']['cost_per_1000_queries'],
            "tokens_per_minute": api_usage_results['rate_limiting_info']['tokens_per_minute'],
            "recommended_batch_size": api_usage_results['rate_limiting_info']['recommended_batch_size']
        },
        "success_criteria_assessment": {
            "srl_routing_accuracy_target": "≥80%",
            "response_time_target": "≤3 seconds",
            "system_reliability": "Concurrent handling",
            "cost_efficiency": "Token optimization"
        },
        "recommendations": performance_report.get('recommendations', [])
    }
    
    # Save comprehensive summary
    summary_file = output_dir / f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import json
    with open(summary_file, 'w') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    logger.info(f"Comprehensive evaluation summary saved to: {summary_file}")
    
    # ============================================================================
    # Final Report
    # ============================================================================
    
    total_duration = datetime.now() - start_time
    
    print("\n" + "="*80)
    print("AGENTIC EDU-RAG COMPREHENSIVE EVALUATION COMPLETED")
    print("="*80)
    print(f"Total Duration: {total_duration.total_seconds()/60:.1f} minutes")
    print(f"Evaluation Summary: {summary_file}")
    print("\nKey Results:")
    print(f"  ✓ Processed {evaluation_summary['dataset_processing']['total_queries_processed']} CS1QA queries")
    print(f"  ✓ Classification success rate: {evaluation_summary['dataset_processing']['classification_success_rate']:.1%}")
    print(f"  ✓ Mean confidence: {evaluation_summary['dataset_processing']['mean_classification_confidence']:.3f}")
    print(f"  ✓ Max concurrent users tested: {evaluation_summary['performance_metrics']['max_concurrent_users_tested']}")
    print(f"  ✓ Cost per 1000 queries: ${evaluation_summary['cost_analysis']['estimated_cost_per_1000_queries']:.4f}")
    
    if evaluation_summary['recommendations']:
        print(f"\nRecommendations ({len(evaluation_summary['recommendations'])}):")
        for i, rec in enumerate(evaluation_summary['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print("\nOutput Files:")
    print(f"  - Dataset Processing: {output_dir / 'cs1qa_processed'}")
    print(f"  - Performance Results: {output_dir / 'performance_tests'}")
    print(f"  - Evaluation Summary: {summary_file}")
    print("="*80)

def main():
    """Main entry point for the evaluation script."""
    
    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable must be set")
        print("Please add your OpenAI API key to your .env file or environment variables")
        return
    
    # Create data directory if it doesn't exist
    data_dir = Path("data/cs1qa/raw")
    if not data_dir.exists():
        print(f"Creating data directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
        print("Please place your CS1QA dataset files in data/cs1qa/raw/")
        print("Supported formats: JSON, JSONL, CSV, TSV")
    
    # Run comprehensive evaluation
    try:
        asyncio.run(run_comprehensive_evaluation())
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
