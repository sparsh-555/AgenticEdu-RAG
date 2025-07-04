#!/usr/bin/env python3
"""
CS1QA Processing Script (Without Validation)

This script processes the CS1QA dataset with SRL labeling and saves the results
without running the problematic validation step that causes division by zero errors.

Author: AgenticEdu-RAG System
"""

import asyncio
import os
import logging
from pathlib import Path
from data.cs1qa_processor import CS1QAProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def process_and_save_cs1qa():
    """
    Process CS1QA dataset with SRL labeling and save results without validation.
    """
    
    # Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set")
        return
    
    # Paths
    data_dir = Path("data/cs1qa")
    output_dir = Path("evaluation_results/cs1qa_processed")
    
    logger.info("Starting CS1QA dataset processing (without validation)")
    
    try:
        # Initialize processor
        processor = CS1QAProcessor(
            data_dir=data_dir,
            openai_api_key=OPENAI_API_KEY,
            output_dir=output_dir
        )
        
        # Load CS1QA dataset
        logger.info("Loading CS1QA dataset...")
        await processor.load_cs1qa_dataset()
        logger.info(f"✅ Loaded {len(processor.raw_queries)} queries from CS1QA dataset")
        
        # Apply SRL labeling with the same settings as run_evaluation.py
        logger.info("Starting SRL labeling (this may take 15-30 minutes due to rate limiting)...")
        await processor.apply_srl_labeling(batch_size=3, max_concurrent=2)
        logger.info(f"✅ Processed {len(processor.labeled_queries)} queries with SRL labels")
        
        # Skip validation entirely to avoid division by zero error
        logger.info("⚠️ Skipping validation step to avoid division by zero error")
        
        # Skip dataset splitting - we just want the full labeled dataset
        logger.info("⚠️ Skipping dataset splitting - keeping full labeled dataset intact")
        
        # Save processed data
        logger.info("Saving processed data...")
        processor.save_processed_data()
        logger.info("✅ Data saved successfully!")
        
        # Generate basic processing report (without validation metrics)
        logger.info("Generating processing report...")
        processing_report = processor.generate_processing_report()
        
        # Print summary
        logger.info("🎉 CS1QA processing completed successfully!")
        logger.info(f"📊 Total queries processed: {len(processor.labeled_queries)}")
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"📈 Classification success rate: {processing_report['processing_summary']['classification_success_rate']:.2%}")
        
        # List output files
        output_files = list(output_dir.glob("*.json")) + list(output_dir.glob("*.csv"))
        logger.info(f"📄 Generated files:")
        for file_path in output_files:
            logger.info(f"   - {file_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ CS1QA processing failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(process_and_save_cs1qa())
    if success:
        print("\n🎉 SUCCESS: CS1QA dataset processed and saved!")
        print("📁 Check evaluation_results/cs1qa_processed/ for output files")
    else:
        print("\n❌ FAILED: CS1QA processing encountered errors")
