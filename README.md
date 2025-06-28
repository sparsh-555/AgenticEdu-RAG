# Agentic Edu-RAG System

A research implementation of a multi-agent RAG system for Self-Regulated Learning in CS education. This system demonstrates how specialized agents can effectively route programming help requests based on SRL phases.

## Overview

This system implements a hierarchical multi-agent architecture that routes student programming queries to specialized agents based on Self-Regulated Learning (SRL) phases:

- **Implementation Agent**: Handles forethought phase queries (planning, strategy, "how to code")
- **Debugging Agent**: Handles performance phase queries (error resolution, troubleshooting)
- **Central Orchestrator**: Routes queries and coordinates responses

## Project Structure

```
agentic_edu_rag/
├── agents/                    # Multi-agent system components
├── classification/            # SRL-aware query classification
├── rag/                      # RAG implementation with vector storage
├── evaluation/               # Comprehensive evaluation framework
│   ├── cs1qa_processor.py    # CS1QA dataset processing & SRL labeling
│   ├── performance_tests.py  # System performance testing
│   └── metrics.py           # Evaluation metrics
├── data/                     # Educational content and datasets
│   ├── cs1qa_processor.py   # ← CS1QA dataset processing
│   ├── cs1qa/
│   │   ├── raw/             # Original CS1QA files
│   │   └── processed/       # SRL-labeled evaluation data
│   └── pdfs/                # PDF documents
├── config/                   # System configuration
├── utils/                    # Utility functions
└── main.py                  # Main application entry point
```

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd AgenticEdu-RAG
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key and other settings
   ```

4. **Set up data directories:**
   ```bash
   mkdir -p data/cs1qa/raw
   mkdir -p data/educational_content
   ```

## Evaluation Framework

### CS1QA Dataset Processing

The `cs1qa_processor.py` module handles CS1QA dataset processing and SRL phase labeling:

#### Key Features:
- **Multi-format Support**: Loads CS1QA data from JSON, JSONL, CSV, and TSV formats
- **SRL Classification**: Uses GPT-4o to classify queries into implementation vs debugging phases
- **Validation Framework**: Ensures labeling consistency and quality
- **Dataset Splitting**: Creates train/validation/test splits for evaluation

#### Usage Example:
```python
from data.cs1qa_processor import CS1QAProcessor
import asyncio

async def process_dataset():
    processor = CS1QAProcessor(
        data_dir=Path("data/cs1qa"),
        openai_api_key="your-api-key",
        output_dir=Path("data/cs1qa/processed")
    )
    
    # Load and process dataset
    await processor.load_cs1qa_dataset()
    await processor.apply_srl_labeling(batch_size=5)
    
    # Validate and create evaluation sets
    processor.validate_labeling_consistency()
    datasets = processor.create_evaluation_datasets()
    
    # Save results
    processor.save_processed_data()
    
    return processor.generate_processing_report()

# Run processing
report = asyncio.run(process_dataset())
```

#### Expected CS1QA Data Format:
The processor supports flexible field naming for common CS1QA formats:
- `question`/`query`/`text`: The student's question
- `code`/`code_snippet`/`program`: Code snippet (optional)
- `error`/`error_message`: Error message (optional)
- `tags`/`categories`: Query categories
- `difficulty`/`level`: Difficulty level

### Performance Testing

The `performance_tests.py` module provides comprehensive system performance evaluation:

#### Key Features:
- **Response Time Testing**: Measures single query response times across different types
- **Concurrency Testing**: Tests system performance under concurrent load
- **Scalability Analysis**: Evaluates system behavior with increasing query loads
- **API Usage Tracking**: Monitors OpenAI API costs and token usage
- **Resource Monitoring**: Tracks CPU, memory, and network usage

#### Usage Example:
```python
from evaluation.performance_tests import PerformanceTester, LoadTestConfig
import asyncio

async def run_performance_tests():
    tester = PerformanceTester(
        openai_api_key="your-api-key",
        output_dir=Path("performance_results")
    )
    
    # Response time tests
    await tester.run_response_time_tests(num_iterations=10)
    
    # Concurrency tests
    load_config = LoadTestConfig(min_users=1, max_users=20, step_size=5)
    await tester.run_concurrency_tests(load_config)
    
    # Scalability tests
    await tester.run_scalability_tests(query_loads=[1, 5, 10, 20, 50])
    
    # API usage analysis
    await tester.run_api_usage_analysis(analysis_duration=300)
    
    # Generate comprehensive report
    report = tester.generate_performance_report()
    tester.save_results()
    tester.create_performance_visualizations()
    
    return report

# Run tests
results = asyncio.run(run_performance_tests())
```

#### Performance Metrics:
- **Response Time**: Average, median, P95, P99 response times
- **Throughput**: Requests per second under various loads
- **Error Rates**: Failure rates under stress conditions
- **Resource Usage**: CPU, memory, and network utilization
- **Cost Analysis**: Token usage and estimated API costs

## Configuration

### Environment Variables (.env):
```bash
OPENAI_API_KEY=your_openai_api_key_here
VECTOR_DB_PATH=./data/vector_db
LOG_LEVEL=INFO
MAX_CONCURRENT_REQUESTS=10
RESPONSE_TIMEOUT=30
```

### System Settings (config/settings.py):
- Agent prompt templates
- RAG retrieval parameters
- Classification thresholds
- Performance monitoring settings

## Evaluation Workflow

1. **Dataset Preparation:**
   ```bash
   # Place CS1QA data in data/cs1qa/raw/
   python -m data.cs1qa_processor
   ```

2. **System Training:**
   ```bash
   # Load educational content and create vector embeddings
   python -m rag.content_loader
   ```

3. **Performance Testing:**
   ```bash
   # Run comprehensive performance evaluation
   python -m evaluation.performance_tests
   ```

4. **Classification Evaluation:**
   ```bash
   # Evaluate SRL classification accuracy
   python -m evaluation.metrics
   ```

## Research Metrics

The system tracks several key research metrics:

- **SRL Routing Accuracy**: >80% target for implementation vs debugging classification
- **Response Time**: <3 seconds average for typical queries
- **Concurrent Handling**: Support for multiple simultaneous users
- **Cost Efficiency**: Token usage optimization and API cost tracking

## Educational Content Organization

```
data/educational_content/
├── implementation_guides/    # Forethought phase materials
│   ├── planning_strategies.txt
│   ├── algorithm_design.txt
│   └── implementation_patterns.txt
├── debugging_resources/      # Performance phase materials
│   ├── error_explanations.txt
│   ├── debugging_techniques.txt
│   └── troubleshooting_guides.txt
└── general_programming/      # Shared content
    ├── python_fundamentals.txt
    └── best_practices.txt
```

## Results and Outputs

### CS1QA Processing Outputs:
- `cs1qa_srl_labeled.json`: Labeled dataset with SRL classifications
- `cs1qa_srl_labeled.csv`: CSV format for analysis
- `validation_metrics.json`: Quality validation results

### Performance Testing Outputs:
- `performance_test_YYYYMMDD_HHMMSS.json`: Detailed test results
- `performance_test_report_YYYYMMDD_HHMMSS.json`: Summary report
- `performance_analysis_YYYYMMDD_HHMMSS.png`: Visualization charts

## Success Criteria

✅ **Technical Performance Targets:**
- SRL routing accuracy >80%
- Average response time <3 seconds
- Successful concurrent handling
- Proper error handling and recovery

✅ **Research Contribution:**
- SRL phase-aware query routing
- Specialized agent responses
- Quantitative performance characteristics
- Reusable multi-agent RAG patterns

## Development and Testing

### Running Tests:
```bash
# Unit tests
pytest tests/

# Integration tests
python -m pytest tests/integration/

# Performance benchmarks
python -m evaluation.performance_tests
```

### Development Server:
```bash
# Start development server
python main.py

# Or with uvicorn for API mode
uvicorn main:app --reload
```

## Contributing

1. Follow the established code structure
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure all evaluation metrics pass

## Research Applications

This system serves as a foundation for:
- Educational technology research
- Multi-agent system design patterns
- SRL-aligned intelligent tutoring systems
- RAG system optimization studies

## License

[Specify license here]

## Citation

If you use this system in your research, please cite:
```
[Citation format to be determined]
```

---

For detailed implementation questions, please refer to the individual module documentation or create an issue in the repository.
