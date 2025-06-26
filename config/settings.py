"""
Configuration Management for Agentic Edu-RAG System

This module provides centralized configuration management with validation,
environment variable loading, and secure API key handling. The configuration
system is designed to support both development and production environments
while maintaining security best practices.

Key Design Principles:
1. Environment-based configuration with fallback defaults
2. Validation of critical settings before system initialization
3. Secure handling of sensitive information (API keys)
4. Type safety using Pydantic models
5. Extensibility for research parameter tuning
"""

import os
from typing import Optional, List
from pathlib import Path
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
import yaml


class OpenAIConfig(BaseModel):
    """OpenAI API configuration with validation."""
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(default="gpt-4o", description="Primary model for agent responses")
    embedding_model: str = Field(default="text-embedding-3-small", description="Model for embeddings")
    max_tokens: int = Field(default=8000, ge=100, le=32000, description="Maximum tokens per request")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Temperature for response generation")
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if not v or v == 'your_openai_api_key_here':
            raise ValueError("Valid OpenAI API key is required")
        return v


class ChromaConfig(BaseModel):
    """Chroma vector database configuration."""
    persist_directory: str = Field(default="./data/chroma_db", description="Directory for vector database persistence")
    collection_name: str = Field(default="edu_rag_collection", description="Name of the vector collection")
    
    @validator('persist_directory')
    def validate_persist_directory(cls, v):
        # Ensure directory exists or can be created
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class SystemConfig(BaseModel):
    """General system configuration parameters."""
    log_level: str = Field(default="INFO", description="Logging level")
    max_concurrent_requests: int = Field(default=5, ge=1, le=20, description="Maximum concurrent API requests")
    request_timeout: int = Field(default=30, ge=5, le=120, description="Request timeout in seconds")
    classification_confidence_threshold: float = Field(
        default=0.7, ge=0.5, le=1.0, 
        description="Minimum confidence for SRL classification"
    )
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


class AgentConfig(BaseModel):
    """Configuration for specialized agents."""
    implementation_agent_prompt_template: str = Field(
        default="agent_prompts.yaml",
        description="Path to implementation agent prompt templates"
    )
    debugging_agent_prompt_template: str = Field(
        default="agent_prompts.yaml",
        description="Path to debugging agent prompt templates"
    )
    orchestrator_prompt_template: str = Field(
        default="agent_prompts.yaml",
        description="Path to orchestrator prompt templates"
    )


class EvaluationConfig(BaseModel):
    """Configuration for evaluation and metrics."""
    dataset_path: str = Field(default="./data/cs1qa/processed", description="Path to evaluation dataset")
    results_output_dir: str = Field(default="./evaluation/results", description="Directory for evaluation results")
    min_test_samples: int = Field(default=100, ge=10, description="Minimum samples for valid evaluation")
    
    @validator('results_output_dir')
    def validate_results_dir(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class Settings(BaseModel):
    """
    Main configuration class that aggregates all system settings.
    
    This class provides a centralized configuration interface with validation,
    environment variable loading, and proper error handling for missing or
    invalid configuration values.
    """
    openai: OpenAIConfig
    chroma: ChromaConfig
    system: SystemConfig
    agents: AgentConfig
    evaluation: EvaluationConfig
    
    class Config:
        env_nested_delimiter = '__'
        
    @classmethod
    def load_from_env(cls, env_file: Optional[str] = None) -> 'Settings':
        """
        Load configuration from environment variables and optional .env file.
        
        Args:
            env_file: Path to .env file (optional)
            
        Returns:
            Validated Settings instance
            
        Raises:
            ValueError: If critical configuration is missing or invalid
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()  # Load from .env in current directory if exists
        
        try:
            return cls(
                openai=OpenAIConfig(
                    api_key=os.getenv('OPENAI_API_KEY', ''),
                    model=os.getenv('OPENAI_MODEL', 'gpt-4o'),
                    embedding_model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
                    max_tokens=int(os.getenv('MAX_TOKENS', '8000')),
                    temperature=float(os.getenv('TEMPERATURE', '0.1'))
                ),
                chroma=ChromaConfig(
                    persist_directory=os.getenv('CHROMA_PERSIST_DIRECTORY', './data/chroma_db'),
                    collection_name=os.getenv('CHROMA_COLLECTION_NAME', 'edu_rag_collection')
                ),
                system=SystemConfig(
                    log_level=os.getenv('LOG_LEVEL', 'INFO'),
                    max_concurrent_requests=int(os.getenv('MAX_CONCURRENT_REQUESTS', '5')),
                    request_timeout=int(os.getenv('REQUEST_TIMEOUT', '30')),
                    classification_confidence_threshold=float(os.getenv('CLASSIFICATION_CONFIDENCE_THRESHOLD', '0.7'))
                ),
                agents=AgentConfig(),
                evaluation=EvaluationConfig(
                    dataset_path=os.getenv('EVALUATION_DATASET_PATH', './data/cs1qa/processed'),
                    results_output_dir=os.getenv('RESULTS_OUTPUT_DIR', './evaluation/results')
                )
            )
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {str(e)}")
    
    def validate_system_requirements(self) -> List[str]:
        """
        Validate that all system requirements are met.
        
        Returns:
            List of validation errors (empty if all validations pass)
        """
        errors = []
        
        # Check if required directories exist or can be created
        required_dirs = [
            self.chroma.persist_directory,
            self.evaluation.dataset_path,
            self.evaluation.results_output_dir
        ]
        
        for dir_path in required_dirs:
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot access/create directory {dir_path}: {str(e)}")
        
        # Validate OpenAI API key format (basic check)
        if not self.openai.api_key.startswith('sk-'):
            errors.append("OpenAI API key appears to be invalid (should start with 'sk-')")
        
        return errors


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """
    Get global settings instance (singleton pattern).
    
    Args:
        reload: Force reload from environment
        
    Returns:
        Settings instance
    """
    global _settings
    if _settings is None or reload:
        _settings = Settings.load_from_env()
        
        # Validate system requirements
        errors = _settings.validate_system_requirements()
        if errors:
            raise RuntimeError(f"System validation failed:\n" + "\n".join(f"- {error}" for error in errors))
    
    return _settings


def load_prompt_templates(template_file: str = "prompts.yaml") -> dict:
    """
    Load prompt templates from YAML file.
    
    Args:
        template_file: Path to YAML file containing prompt templates
        
    Returns:
        Dictionary of prompt templates
    """
    template_path = Path(__file__).parent / template_file
    
    if not template_path.exists():
        # Return default templates if file doesn't exist
        return {
            "orchestrator": {
                "classification": "You are an expert educational AI that routes programming queries...",
                "system": "You coordinate between specialized agents..."
            },
            "implementation_agent": {
                "system": "You are a programming implementation specialist...",
                "forethought_phase": "Help students plan and strategize their coding approach..."
            },
            "debugging_agent": {
                "system": "You are a debugging specialist...",
                "performance_phase": "Help students systematically debug and resolve code issues..."
            }
        }
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load prompt templates from {template_path}: {str(e)}")


if __name__ == "__main__":
    # Configuration validation script
    try:
        settings = get_settings()
        print("✅ Configuration validation successful!")
        print(f"OpenAI Model: {settings.openai.model}")
        print(f"Vector DB: {settings.chroma.persist_directory}")
        print(f"Log Level: {settings.system.log_level}")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        exit(1)
