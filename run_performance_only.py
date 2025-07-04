#!/usr/bin/env python3
"""
Performance Testing Only Script

Run just the performance tests without any dataset processing.
Use this after CS1QA data has been processed.

Author: Agentic Edu-RAG System
"""

import asyncio
import os
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

async def run_performance_tests_only():
    """
    Run only performance tests for the Agentic Edu-RAG system.
    """
    
    # Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set")
        return
    
    # Paths
    output_dir = Path("evaluation_results/performance_tests")
    
    logger.info("Starting performance testing of Agentic Edu-RAG system")
    start_time = datetime.now()
    
    try:
        tester = PerformanceTester(
            openai_api_key=OPENAI_API_KEY,
            output_dir=output_dir
        )
        
        # Run response time tests
        logger.info("Running response time tests...")
        await tester.run_response_time_tests(num_iterations=10)
        
        # Run concurrency tests
        logger.info("Running concurrency tests...")
        load_config = LoadTestConfig(min_users=1, max_users=15, step_size=3)
        await tester.run_concurrency_tests(load_config)
        
        # Run scalability tests
        logger.info("Running scalability tests...")
        await tester.run_scalability_tests(query_loads=[1, 5, 10, 20])
        
        # Run API usage analysis
        logger.info("Running API usage analysis...")
        await tester.run_api_usage_analysis(analysis_duration=180)  # 3 minutes
        
        # Generate comprehensive report
        performance_report = tester.generate_performance_report()
        tester.save_results()
        tester.create_performance_visualizations()
        
        logger.info("✅ Performance testing completed successfully!")
        logger.info(f"📁 Results saved to: {output_dir}")
        
        # Show key metrics
        if performance_report:
            logger.info("📊 Key Performance Metrics:")
            logger.info(f"   - Average response time: {performance_report.get('avg_response_time', 'N/A')} seconds")
            logger.info(f"   - Max concurrent users tested: {performance_report.get('max_concurrent_users', 'N/A')}")
            logger.info(f"   - Total API requests made: {performance_report.get('total_requests', 'N/A')}")
        
    except Exception as e:
        logger.error(f"❌ Performance testing failed: {e}")
        return
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("🎉 Performance testing completed!")
    logger.info(f"⏱️ Duration: {duration}")

if __name__ == "__main__":
    asyncio.run(run_performance_tests_only())
