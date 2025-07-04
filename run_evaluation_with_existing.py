#!/usr/bin/env python3
"""
Modified Evaluation Script for Agentic Edu-RAG System

This script checks for existing processed CS1QA data and skips processing if files exist,
then runs the performance testing phase.

Author: Agentic Edu-RAG System
"""

import asyncio
import os
import json
from pathlib import Path
import logging
from datetime import datetime

from evaluation.performance_tests import PerformanceTester, LoadTestConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_evaluation_with_existing_data():
    """
    Run evaluation using existing processed CS1QA data if available,
    otherwise prompt user to run processing first.
    """
    
    # Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set")
        return
    
    # Paths
    output_dir = Path("evaluation_results")
    cs1qa_processed_dir = output_dir / "cs1qa_processed"
    cs1qa_json_file = cs1qa_processed_dir / "cs1qa_srl_labeled.json"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting evaluation of Agentic Edu-RAG system")
    start_time = datetime.now()
    
    # ============================================================================
    # Phase 1: Check for Existing CS1QA Data
    # ============================================================================
    
    if cs1qa_json_file.exists():
        logger.info("✅ Found existing processed CS1QA data")
        logger.info(f"📁 Using data from: {cs1qa_json_file}")
        
        # Load and validate existing data
        try:
            with open(cs1qa_json_file, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
            
            logger.info(f"📊 Loaded {len(processed_data)} processed queries")
            
            # Basic validation of the data structure
            sample_record = processed_data[0] if processed_data else {}
            required_fields = ['query_id', 'question', 'srl_label']
            
            if all(field in sample_record for field in required_fields):
                logger.info("✅ Data structure validation passed")
                
                # Show basic statistics
                srl_phases = [record['srl_label']['phase'] for record in processed_data]
                implementation_count = srl_phases.count('implementation')
                debugging_count = srl_phases.count('debugging')
                
                logger.info(f"📈 SRL Phase Distribution:")
                logger.info(f"   - Implementation: {implementation_count} ({implementation_count/len(srl_phases)*100:.1f}%)")
                logger.info(f"   - Debugging: {debugging_count} ({debugging_count/len(srl_phases)*100:.1f}%)")
                
            else:
                logger.warning("⚠️ Data structure validation failed - some required fields missing")
                
        except Exception as e:
            logger.error(f"❌ Failed to load existing data: {e}")
            logger.info("💡 Please run 'python save_cs1qa_processed.py' first")
            return
    
    else:
        logger.error("❌ No processed CS1QA data found")
        logger.info("💡 Please run 'python save_cs1qa_processed.py' first to process the dataset")
        logger.info(f"📁 Expected file location: {cs1qa_json_file}")
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
        
        # Run response time tests
        logger.info("Running response time tests...")
        await tester.run_response_time_tests(num_iterations=5)
        
        # Run lightweight concurrency tests
        logger.info("Running concurrency tests...")
        load_config = LoadTestConfig(min_users=1, max_users=10, step_size=2)
        await tester.run_concurrency_tests(load_config)
        
        # Run scalability tests with smaller loads
        logger.info("Running scalability tests...")
        await tester.run_scalability_tests(query_loads=[1, 3, 5, 10])
        
        # Generate performance report
        performance_report = tester.generate_performance_report()
        tester.save_results()
        
        logger.info("✅ Performance testing completed successfully!")
        logger.info(f"📁 Performance results saved to: {output_dir / 'performance_tests'}")
        
    except Exception as e:
        logger.error(f"❌ Performance testing failed: {e}")
        return
    
    # ============================================================================
    # Final Summary
    # ============================================================================
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("🎉 Comprehensive evaluation completed successfully!")
    logger.info(f"⏱️ Total duration: {duration}")
    logger.info(f"📁 All results saved to: {output_dir}")
    
    # List all generated files
    logger.info("📄 Generated files:")
    for result_file in output_dir.rglob("*.json"):
        logger.info(f"   - {result_file.relative_to(output_dir)}")
    for result_file in output_dir.rglob("*.csv"):
        logger.info(f"   - {result_file.relative_to(output_dir)}")

if __name__ == "__main__":
    asyncio.run(run_evaluation_with_existing_data())
