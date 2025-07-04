#!/usr/bin/env python3
"""
Full CS1QA Dataset Processing with Incremental Saving

This script processes the complete CS1QA dataset (1,847 queries) with:
- High-confidence ground truth labeling (≥0.7 confidence)
- Incremental batch saving to prevent data loss
- Resume capability if interrupted
- Progress tracking and reporting

Author: AgenticEdu-RAG System
"""

import asyncio
import os
import logging
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from data.cs1qa_processor import CS1QAProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IncrementalCS1QAProcessor:
    """CS1QA processor with incremental saving and resume capability."""
    
    def __init__(self, 
                 data_dir: Path,
                 openai_api_key: str,
                 output_dir: Path,
                 batch_size: int = 10,
                 save_every_n_batches: int = 5):
        """
        Initialize incremental processor.
        
        Args:
            data_dir: CS1QA data directory
            openai_api_key: OpenAI API key
            output_dir: Output directory for results
            batch_size: Queries per batch for API calls
            save_every_n_batches: Save progress every N batches
        """
        self.data_dir = data_dir
        self.openai_api_key = openai_api_key
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.save_every_n_batches = save_every_n_batches
        
        # Progress tracking
        self.progress_file = output_dir / "processing_progress.json"
        self.temp_results_file = output_dir / "temp_processed_results.json"
        self.final_results_file = output_dir / "cs1qa_srl_labeled_with_ground_truth.json"
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processor
        self.processor = CS1QAProcessor(
            data_dir=data_dir,
            openai_api_key=openai_api_key,
            output_dir=output_dir
        )
        
    def save_progress(self, batch_num: int, total_batches: int, processed_results: List[Dict[str, Any]]):
        """Save current progress to allow resuming."""
        progress_data = {
            "batch_num": batch_num,
            "total_batches": total_batches,
            "queries_processed": len(processed_results),
            "last_saved": len(processed_results),
            "confidence_threshold": 0.7
        }
        
        # Save progress metadata
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        # Save current results
        with open(self.temp_results_file, 'w') as f:
            json.dump(processed_results, f, indent=2)
        
        logger.info(f"💾 Progress saved: batch {batch_num}/{total_batches}, {len(processed_results)} queries processed")
    
    def load_progress(self) -> tuple[int, List[Dict[str, Any]]]:
        """Load previous progress if exists."""
        if not self.progress_file.exists() or not self.temp_results_file.exists():
            logger.info("No previous progress found, starting fresh")
            return 0, []
        
        try:
            # Load progress metadata
            with open(self.progress_file, 'r') as f:
                progress_data = json.load(f)
            
            # Load processed results
            with open(self.temp_results_file, 'r') as f:
                processed_results = json.load(f)
            
            start_batch = progress_data["batch_num"]
            queries_count = len(processed_results)
            
            logger.info(f"📂 Resuming from batch {start_batch}, {queries_count} queries already processed")
            return start_batch, processed_results
            
        except Exception as e:
            logger.warning(f"Failed to load progress: {e}, starting fresh")
            return 0, []
    
    def populate_ground_truth_labels(self, processed_data: List[Dict[str, Any]], confidence_threshold: float = 0.7):
        """Populate ground truth labels for high-confidence predictions."""
        # This method is no longer used - ground truth is populated per batch
        pass
    
    async def process_full_dataset(self) -> Dict[str, Any]:
        """Process the complete CS1QA dataset with incremental saving."""
        logger.info("🚀 Starting full CS1QA dataset processing")
        
        try:
            # Load dataset
            await self.processor.load_cs1qa_dataset()
            total_queries = len(self.processor.raw_queries)
            logger.info(f"📊 Loaded {total_queries} queries for processing")
            
            # Calculate batches
            total_batches = (total_queries + self.batch_size - 1) // self.batch_size
            logger.info(f"📦 Processing in {total_batches} batches of {self.batch_size} queries each")
            
            # Load previous progress
            start_batch_num, processed_results = self.load_progress()
            start_query_idx = start_batch_num * self.batch_size
            
            # Convert existing results to the expected format if resuming
            if processed_results:
                # Ensure we have the right data structure
                existing_labeled_queries = []
                for result in processed_results:
                    # Convert back to LabeledQuery format for continued processing
                    # This is a bit hacky but necessary for the processor to continue
                    pass  # We'll work with the JSON directly
            
            # Process remaining batches
            for batch_num in range(start_batch_num, total_batches):
                start_idx = batch_num * self.batch_size
                end_idx = min(start_idx + self.batch_size, total_queries)
                
                logger.info(f"🔄 Processing batch {batch_num + 1}/{total_batches} (queries {start_idx}-{end_idx-1})")
                
                # Create a temporary processor for this batch
                batch_queries = self.processor.raw_queries[start_idx:end_idx]
                temp_processor = CS1QAProcessor(
                    data_dir=self.data_dir,
                    openai_api_key=self.openai_api_key,
                    output_dir=self.output_dir
                )
                temp_processor.raw_queries = batch_queries
                
                try:
                    # Process this batch
                    await temp_processor.apply_srl_labeling(
                        batch_size=min(3, len(batch_queries)), 
                        max_concurrent=2
                    )
                    
                    # Convert batch results to JSON format and populate ground truth immediately
                    batch_results = []
                    for labeled_query in temp_processor.labeled_queries:
                        from dataclasses import asdict
                        result = {
                            **asdict(labeled_query.query),
                            'srl_label': asdict(labeled_query.srl_label),
                            'ground_truth_label': None,
                            'validation_status': 'pending',
                            'processing_timestamp': datetime.now().isoformat()
                        }
                        
                        # Populate ground truth immediately for high-confidence predictions
                        confidence = labeled_query.srl_label.confidence
                        if confidence >= 0.7:
                            result['ground_truth_label'] = labeled_query.srl_label.phase
                            result['validation_status'] = 'auto_validated'
                        else:
                            result['ground_truth_label'] = None
                            result['validation_status'] = 'low_confidence'
                        
                        batch_results.append(result)
                    
                    # Add batch results to overall results
                    processed_results.extend(batch_results)
                    
                    logger.info(f"✅ Batch {batch_num + 1} completed: {len(batch_results)} queries processed")
                    
                except Exception as e:
                    logger.error(f"❌ Batch {batch_num + 1} failed: {e}")
                    # Continue with next batch rather than failing completely
                    continue
                
                # Save progress periodically
                if (batch_num + 1) % self.save_every_n_batches == 0:
                    self.save_progress(batch_num + 1, total_batches, processed_results)
                
                # Brief pause between batches
                await asyncio.sleep(2)
            
            # Final processing: ground truth labels already populated per batch
            logger.info("✅ Ground truth labels populated during batch processing")
            
            # Save final results
            logger.info("💾 Saving final processed dataset...")
            with open(self.final_results_file, 'w') as f:
                json.dump(processed_results, f, indent=2)
            
            # Calculate final statistics
            total_processed = len(processed_results)
            high_confidence_count = sum(1 for r in processed_results if r.get('ground_truth_label') is not None)
            ground_truth_rate = high_confidence_count / total_processed if total_processed > 0 else 0
            
            # Phase distribution
            phase_counts = {
                'implementation': sum(1 for r in processed_results if r.get('ground_truth_label') == 'implementation'),
                'debugging': sum(1 for r in processed_results if r.get('ground_truth_label') == 'debugging')
            }
            
            # Confidence statistics
            confidences = [r['srl_label']['confidence'] for r in processed_results]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Generate final report
            final_report = {
                "processing_summary": {
                    "total_queries_loaded": total_queries,
                    "total_queries_processed": total_processed,
                    "processing_success_rate": total_processed / total_queries,
                    "high_confidence_samples": high_confidence_count,
                    "ground_truth_rate": ground_truth_rate,
                    "average_confidence": avg_confidence
                },
                "phase_distribution": {
                    "ground_truth_labels": phase_counts,
                    "all_predictions": {
                        'implementation': sum(1 for r in processed_results if r['srl_label']['phase'] == 'implementation'),
                        'debugging': sum(1 for r in processed_results if r['srl_label']['phase'] == 'debugging')
                    }
                },
                "confidence_analysis": {
                    "mean": avg_confidence,
                    "min": min(confidences) if confidences else 0,
                    "max": max(confidences) if confidences else 0,
                    "high_confidence_threshold": 0.7,
                    "high_confidence_count": high_confidence_count
                },
                "output_files": {
                    "final_dataset": str(self.final_results_file),
                    "total_records": total_processed,
                    "ground_truth_records": high_confidence_count
                }
            }
            
            # Save processing report
            report_file = self.output_dir / "full_processing_report.json"
            with open(report_file, 'w') as f:
                json.dump(final_report, f, indent=2)
            
            # Clean up temporary files
            if self.progress_file.exists():
                self.progress_file.unlink()
            if self.temp_results_file.exists():
                self.temp_results_file.unlink()
            
            logger.info("🎉 FULL DATASET PROCESSING COMPLETED!")
            logger.info(f"📊 Final Statistics:")
            logger.info(f"   - Total processed: {total_processed}/{total_queries} queries")
            logger.info(f"   - Ground truth labels: {high_confidence_count} ({ground_truth_rate:.1%})")
            logger.info(f"   - Average confidence: {avg_confidence:.3f}")
            logger.info(f"   - Implementation queries: {phase_counts['implementation']}")
            logger.info(f"   - Debugging queries: {phase_counts['debugging']}")
            logger.info(f"📁 Output files:")
            logger.info(f"   - Dataset: {self.final_results_file}")
            logger.info(f"   - Report: {report_file}")
            
            return final_report
            
        except Exception as e:
            logger.error(f"❌ Full dataset processing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Save current progress before failing
            if 'processed_results' in locals() and 'total_batches' in locals():
                current_batch = locals().get('batch_num', 0)
                self.save_progress(current_batch, total_batches, processed_results)
            
            raise

async def main():
    """Main function to run full dataset processing."""
    
    # Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set")
        return
    
    # Paths
    data_dir = Path("data/cs1qa")
    output_dir = Path("cs1qa_full_processed")
    
    logger.info("🚀 Starting Full CS1QA Dataset Processing")
    logger.info("=" * 60)
    logger.info("Features:")
    logger.info("  - Incremental batch processing")
    logger.info("  - Automatic progress saving")
    logger.info("  - Resume capability if interrupted")
    logger.info("  - Ground truth label population")
    logger.info("=" * 60)
    
    try:
        # Initialize processor
        processor = IncrementalCS1QAProcessor(
            data_dir=data_dir,
            openai_api_key=OPENAI_API_KEY,
            output_dir=output_dir,
            batch_size=5,  # Smaller batches for reliability
            save_every_n_batches=3  # Save progress every 3 batches
        )
        
        # Run processing
        report = await processor.process_full_dataset()
        
        print("\n🎉 SUCCESS: Full CS1QA dataset processed!")
        print(f"📁 Check output directory: {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        print("\n❌ FAILED: Check logs for details")
        print("💡 You can resume processing by running this script again")

if __name__ == "__main__":
    asyncio.run(main())
