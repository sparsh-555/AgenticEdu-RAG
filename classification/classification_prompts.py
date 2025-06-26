"""
Classification Prompt Templates for SRL Phase Detection

This module contains carefully crafted prompt templates for Self-Regulated Learning
phase classification. The prompts are based on educational research and optimized
for accurate detection of forethought vs. performance phase learning behaviors.

Theoretical Foundation:
These prompts implement Zimmerman's Self-Regulated Learning model, specifically
focusing on distinguishing between:

1. FORETHOUGHT PHASE: Planning, goal setting, strategy selection, self-efficacy
   - Queries about "how to implement", "best approach", "design strategy"
   - Future-oriented thinking and proactive planning
   - Route to Implementation Agent

2. PERFORMANCE PHASE: Self-monitoring, self-control, help-seeking
   - Queries about errors, debugging, "what's wrong", "fix this"
   - Present-focused problem-solving and reactive help-seeking
   - Route to Debugging Agent

The prompts use advanced prompt engineering techniques including:
- Few-shot learning with curated examples
- Chain-of-thought reasoning
- Confidence calibration
- Domain-specific adaptations
"""

from typing import Dict, List, Optional
import json


# Core classification examples for few-shot learning
CLASSIFICATION_EXAMPLES = [
    {
        "query": "How do I implement a binary search algorithm in Python?",
        "context": {
            "code_snippet": None,
            "error_message": None,
            "student_level": "intermediate"
        },
        "classification": "FORETHOUGHT",
        "confidence": 0.95,
        "reasoning": "Student is planning ahead and asking for implementation strategy. Keywords 'how do I implement' indicate forethought phase planning behavior.",
        "indicators": ["how to implement", "algorithm design", "planning ahead", "no existing code"]
    },
    {
        "query": "My binary search function returns the wrong index when searching for elements",
        "context": {
            "code_snippet": "def binary_search(arr, target):\n    left, right = 0, len(arr)\n    # implementation here",
            "error_message": None,
            "student_level": "intermediate"
        },
        "classification": "PERFORMANCE",
        "confidence": 0.92,
        "reasoning": "Student has existing code that isn't working correctly. 'Returns wrong index' indicates performance monitoring and error detection.",
        "indicators": ["wrong result", "existing code", "debugging needed", "monitoring output"]
    },
    {
        "query": "I'm getting an IndexError: list index out of range in my sorting algorithm",
        "context": {
            "code_snippet": "for i in range(len(arr)):\n    if arr[i+1] < arr[i]:",
            "error_message": "IndexError: list index out of range",
            "student_level": "beginner"
        },
        "classification": "PERFORMANCE",
        "confidence": 0.98,
        "reasoning": "Clear error message and existing code indicate performance phase. Student is self-monitoring and seeking help for error correction.",
        "indicators": ["error message", "IndexError", "existing code", "runtime error"]
    },
    {
        "query": "What's the best approach to solve the two-sum problem efficiently?",
        "context": {
            "code_snippet": None,
            "error_message": None,
            "student_level": "advanced"
        },
        "classification": "FORETHOUGHT",
        "confidence": 0.90,
        "reasoning": "Student is seeking strategy and approach before implementation. 'Best approach' and 'efficiently' indicate strategic planning.",
        "indicators": ["best approach", "strategy planning", "efficiency consideration", "pre-implementation"]
    },
    {
        "query": "Why does my recursive function cause a stack overflow error?",
        "context": {
            "code_snippet": "def factorial(n):\n    return n * factorial(n-1)",
            "error_message": "RecursionError: maximum recursion depth exceeded",
            "student_level": "intermediate"
        },
        "classification": "PERFORMANCE",
        "confidence": 0.94,
        "reasoning": "Student has implemented code that's causing runtime errors. This is performance phase self-monitoring and error correction.",
        "indicators": ["runtime error", "stack overflow", "existing implementation", "error diagnosis"]
    },
    {
        "query": "How should I structure my classes for this object-oriented design problem?",
        "context": {
            "code_snippet": None,
            "error_message": None,
            "student_level": "intermediate"
        },
        "classification": "FORETHOUGHT",
        "confidence": 0.88,
        "reasoning": "Student is in planning phase for OOP design. 'How should I structure' indicates forethought phase architectural planning.",
        "indicators": ["how should I", "structure planning", "design phase", "architecture decision"]
    }
]


def get_classification_prompt(strategy: str = "standard", **kwargs) -> str:
    """
    Get the appropriate classification prompt based on strategy.
    
    Args:
        strategy: Classification strategy ("standard", "few_shot", "domain_specific", etc.)
        **kwargs: Additional parameters for prompt customization
        
    Returns:
        Formatted prompt string
    """
    if strategy == "standard":
        return _get_standard_prompt()
    elif strategy == "few_shot":
        return _get_few_shot_prompt(kwargs.get("num_examples", 3))
    elif strategy == "domain_specific":
        return _get_domain_specific_prompt(kwargs.get("domain", "general"))
    elif strategy == "conversation":
        return _get_conversation_aware_prompt(kwargs.get("conversation_history", ""))
    elif strategy == "edge_case":
        return _get_edge_case_prompt()
    elif strategy == "confidence_calibrated":
        return _get_confidence_calibrated_prompt()
    else:
        return _get_standard_prompt()


def _get_standard_prompt() -> str:
    """Standard SRL classification prompt with clear instructions."""
    return """You are an expert educational AI that classifies student programming queries based on Self-Regulated Learning (SRL) phases. Your task is to determine whether a query represents:

**FORETHOUGHT PHASE**: Students planning, strategizing, and preparing to implement solutions
- Characteristics: "How do I...", "What's the best approach...", "How should I design...", "What algorithm should I use..."
- Focus: Planning, strategy selection, goal setting, pre-implementation thinking
- Route to: Implementation Agent

**PERFORMANCE PHASE**: Students monitoring, debugging, and correcting existing work
- Characteristics: Error messages, "My code doesn't work...", "I'm getting...", "Why does my code..."
- Focus: Self-monitoring, error correction, debugging, performance adjustment
- Route to: Debugging Agent

Analyze the student query and respond with a JSON object containing:
{
    "classification": "FORETHOUGHT" or "PERFORMANCE",
    "confidence": float between 0.0 and 1.0,
    "reasoning": "Clear explanation of your classification decision",
    "indicators": ["list", "of", "key", "indicators", "found"]
}

Consider:
- Keywords and phrases that indicate planning vs. debugging
- Presence of code snippets (suggests performance phase)
- Error messages (strongly indicate performance phase)
- Question structure and intent
- Student's current situation (planning vs. executing)

Provide your analysis as a valid JSON object only."""


def _get_few_shot_prompt(num_examples: int = 3) -> str:
    """Few-shot prompt with curated examples for better accuracy."""
    selected_examples = CLASSIFICATION_EXAMPLES[:num_examples]
    
    examples_text = ""
    for i, example in enumerate(selected_examples, 1):
        examples_text += f"""
Example {i}:
Query: "{example['query']}"
Context: {json.dumps(example['context'], indent=2)}

Classification:
{json.dumps({
    'classification': example['classification'],
    'confidence': example['confidence'],
    'reasoning': example['reasoning'],
    'indicators': example['indicators']
}, indent=2)}

"""
    
    return f"""You are an expert educational AI that classifies student programming queries based on Self-Regulated Learning (SRL) phases.

**FORETHOUGHT PHASE**: Planning, strategy selection, goal setting (→ Implementation Agent)
**PERFORMANCE PHASE**: Self-monitoring, error correction, debugging (→ Debugging Agent)

Here are some examples of correct classifications:
{examples_text}

Now classify the following query using the same JSON format. Consider the query content, any provided code context, error messages, and the student's apparent learning phase.

Respond with only a valid JSON object containing: classification, confidence, reasoning, and indicators."""


def _get_domain_specific_prompt(domain: str = "general") -> str:
    """Domain-specific prompt adapted for different programming areas."""
    
    domain_contexts = {
        "algorithms": {
            "forethought_indicators": ["algorithm design", "approach", "complexity", "optimization", "strategy"],
            "performance_indicators": ["runtime error", "wrong output", "infinite loop", "performance issue"],
            "examples": "algorithm implementation, sorting efficiency, search strategies"
        },
        "data_structures": {
            "forethought_indicators": ["structure choice", "design", "which data structure", "organize data"],
            "performance_indicators": ["memory error", "access issue", "insertion problem", "data corruption"],
            "examples": "array vs linked list choice, tree traversal design, hash table implementation"
        },
        "object_oriented": {
            "forethought_indicators": ["class design", "inheritance", "architecture", "design pattern"],
            "performance_indicators": ["method error", "attribute error", "inheritance issue", "polymorphism bug"],
            "examples": "class hierarchy design, method organization, encapsulation decisions"
        },
        "web_development": {
            "forethought_indicators": ["architecture", "framework choice", "API design", "project structure"],
            "performance_indicators": ["HTTP error", "routing issue", "database error", "frontend bug"],
            "examples": "REST API design, component structure, database schema planning"
        }
    }
    
    context = domain_contexts.get(domain, domain_contexts["algorithms"])
    
    return f"""You are an expert educational AI specializing in {domain} programming education. Classify student queries based on Self-Regulated Learning phases.

**DOMAIN CONTEXT: {domain.title()}**
- Forethought indicators: {', '.join(context['forethought_indicators'])}
- Performance indicators: {', '.join(context['performance_indicators'])}
- Common examples: {context['examples']}

**FORETHOUGHT PHASE** (→ Implementation Agent):
Students planning and designing {domain} solutions before implementation.
Examples: "How should I design...", "What's the best {domain} approach...", "Which structure should I use..."

**PERFORMANCE PHASE** (→ Debugging Agent):
Students debugging and fixing existing {domain} code.
Examples: Error messages, "My {domain} code doesn't...", "Why is my implementation..."

Classify the query considering {domain}-specific patterns and respond with JSON:
{{
    "classification": "FORETHOUGHT" or "PERFORMANCE",
    "confidence": float between 0.0 and 1.0,
    "reasoning": "Explanation focusing on {domain} context",
    "indicators": ["domain-specific", "indicators", "found"]
}}"""


def _get_conversation_aware_prompt(conversation_history: str = "") -> str:
    """Conversation-aware prompt that considers previous interactions."""
    return f"""You are an expert educational AI that classifies student programming queries based on Self-Regulated Learning phases, considering conversation context.

**CONVERSATION CONTEXT:**
{conversation_history if conversation_history else "No previous conversation history available."}

**CLASSIFICATION GUIDELINES:**

**FORETHOUGHT PHASE** (→ Implementation Agent):
- Planning new implementations or approaches
- Asking about design strategies or best practices
- Considering multiple options before implementing
- Building on previous discussions about planning

**PERFORMANCE PHASE** (→ Debugging Agent):
- Working with existing code that has issues
- Following up on previous implementation attempts
- Debugging errors or unexpected behavior
- Monitoring and adjusting current implementations

**CONVERSATION PATTERNS:**
- If previous messages discussed planning, current debugging questions may still be PERFORMANCE
- If conversation shows progression from planning to implementation, classify accordingly
- Consider if the student has moved from forethought to performance phase
- Account for context shifts within the learning process

Analyze the current query in context of the conversation history and respond with JSON:
{
    "classification": "FORETHOUGHT" or "PERFORMANCE",
    "confidence": float between 0.0 and 1.0,
    "reasoning": "Explanation considering conversation context",
    "indicators": ["contextual", "indicators", "from", "conversation", "flow"]
}"""


def _get_edge_case_prompt() -> str:
    """Specialized prompt for handling ambiguous or edge cases."""
    return """You are an expert educational AI handling potentially ambiguous programming queries for SRL phase classification.

**EDGE CASE GUIDELINES:**

**Mixed Indicators**: When queries contain both planning and debugging elements:
- Prioritize the PRIMARY intent and current student need
- Consider the dominant action required (planning vs. fixing)
- Look for the most immediate learning objective

**Ambiguous Language**: When intent is unclear:
- Default to FORETHOUGHT if genuinely uncertain (educational scaffolding)
- Use contextual clues (code presence, error messages, question structure)
- Consider student's likely current phase in the learning process

**Meta-Learning Queries**: Questions about learning itself:
- "How do I get better at debugging?" → FORETHOUGHT (planning improvement)
- "Why do I keep making the same mistakes?" → PERFORMANCE (monitoring issues)

**Conceptual Questions with Code**: Understanding requests with existing code:
- If seeking conceptual understanding → FORETHOUGHT
- If troubleshooting specific issues → PERFORMANCE

**DECISION FRAMEWORK:**
1. Is there an error message or broken code? → Likely PERFORMANCE
2. Is the student asking "how to" without existing code? → Likely FORETHOUGHT
3. Are they seeking strategy/approach guidance? → Likely FORETHOUGHT
4. Are they monitoring/evaluating current work? → Likely PERFORMANCE

For edge cases, err on the side of educational support and provide moderate confidence (0.6-0.8).

Respond with JSON classification, explaining your edge case reasoning."""


def _get_confidence_calibrated_prompt() -> str:
    """Prompt specifically designed for accurate confidence estimation."""
    return """You are an expert educational AI with specialized training in confidence calibration for SRL phase classification.

**CONFIDENCE CALIBRATION GUIDELINES:**

**High Confidence (0.85-1.0):**
- Clear error messages with existing code → PERFORMANCE
- Explicit "how to implement" without code → FORETHOUGHT
- Unambiguous indicators align with single phase

**Medium-High Confidence (0.7-0.85):**
- Strong indicators for one phase with minor opposing signals
- Clear intent but some ambiguous language
- Domain-specific patterns strongly suggest one phase

**Medium Confidence (0.5-0.7):**
- Mixed indicators present
- Ambiguous language requiring inference
- Context suggests classification but isn't definitive

**Low Confidence (0.3-0.5):**
- Equal indicators for both phases
- Very ambiguous or unclear intent
- Requires significant assumption to classify

**Very Low Confidence (0.1-0.3):**
- Completely ambiguous queries
- Insufficient information for reliable classification
- Edge cases with no clear dominant pattern

**CALIBRATION PRINCIPLES:**
- Be honest about uncertainty
- Consider multiple interpretations
- Account for potential misclassification impact
- Align confidence with actual classification accuracy

Provide well-calibrated confidence scores that reflect true classification uncertainty.

Respond with JSON including honest confidence assessment and reasoning for confidence level."""


def get_few_shot_prompt(num_examples: int = 3) -> str:
    """
    Get few-shot prompt with specified number of examples.
    
    Args:
        num_examples: Number of examples to include (1-6)
        
    Returns:
        Few-shot prompt string
    """
    return _get_few_shot_prompt(min(num_examples, len(CLASSIFICATION_EXAMPLES)))


def get_confidence_adjusted_prompt(base_strategy: str = "standard") -> str:
    """
    Get confidence-adjusted version of any base prompt.
    
    Args:
        base_strategy: Base strategy to enhance with confidence calibration
        
    Returns:
        Enhanced prompt with confidence guidance
    """
    base_prompt = get_classification_prompt(base_strategy)
    
    confidence_guidance = """

**CONFIDENCE CALIBRATION:**
- 0.9+: Extremely clear indicators, minimal ambiguity
- 0.8-0.9: Strong indicators, minor opposing signals
- 0.7-0.8: Good indicators, some interpretation needed
- 0.6-0.7: Moderate indicators, notable ambiguity
- 0.5-0.6: Weak indicators, significant uncertainty
- <0.5: Very unclear, high uncertainty

Calibrate your confidence to reflect actual classification certainty."""
    
    return base_prompt + confidence_guidance


def validate_prompt_response(response: str) -> Dict[str, any]:
    """
    Validate and parse the JSON response from classification prompts.
    
    Args:
        response: Raw response from the LLM
        
    Returns:
        Parsed and validated response dictionary
        
    Raises:
        ValueError: If response is invalid
    """
    try:
        # Try to extract JSON from response
        if "{" in response and "}" in response:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
        else:
            raise ValueError("No JSON found in response")
        
        # Validate required fields
        required_fields = ["classification", "confidence", "reasoning", "indicators"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate classification value
        valid_classifications = ["FORETHOUGHT", "PERFORMANCE"]
        if data["classification"] not in valid_classifications:
            raise ValueError(f"Invalid classification: {data['classification']}")
        
        # Validate confidence range
        if not isinstance(data["confidence"], (int, float)) or not (0.0 <= data["confidence"] <= 1.0):
            raise ValueError(f"Invalid confidence value: {data['confidence']}")
        
        # Validate indicators is a list
        if not isinstance(data["indicators"], list):
            raise ValueError("Indicators must be a list")
        
        return data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in response: {str(e)}")
    except Exception as e:
        raise ValueError(f"Response validation failed: {str(e)}")


def get_prompt_for_context(context_type: str, **kwargs) -> str:
    """
    Get the most appropriate prompt for a given context.
    
    Args:
        context_type: Type of context ("simple", "complex", "ambiguous", "domain_specific")
        **kwargs: Additional context parameters
        
    Returns:
        Optimized prompt for the context
    """
    if context_type == "simple":
        return get_classification_prompt("standard")
    elif context_type == "complex":
        return get_classification_prompt("few_shot", num_examples=5)
    elif context_type == "ambiguous":
        return get_classification_prompt("edge_case")
    elif context_type == "domain_specific":
        return get_classification_prompt("domain_specific", domain=kwargs.get("domain", "general"))
    elif context_type == "conversation":
        return get_classification_prompt("conversation", conversation_history=kwargs.get("history", ""))
    else:
        return get_classification_prompt("standard")


if __name__ == "__main__":
    # Test prompt generation
    try:
        print("=== Classification Prompt Templates Test ===\n")
        
        # Test standard prompt
        standard = get_classification_prompt("standard")
        print(f"Standard prompt length: {len(standard)} characters")
        
        # Test few-shot prompt
        few_shot = get_few_shot_prompt(3)
        print(f"Few-shot prompt length: {len(few_shot)} characters")
        
        # Test domain-specific prompt
        domain = get_classification_prompt("domain_specific", domain="algorithms")
        print(f"Domain-specific prompt length: {len(domain)} characters")
        
        # Test validation with sample response
        sample_response = '''
        {
            "classification": "FORETHOUGHT",
            "confidence": 0.85,
            "reasoning": "Student is asking for implementation strategy",
            "indicators": ["how to implement", "planning ahead"]
        }
        '''
        
        validated = validate_prompt_response(sample_response)
        print(f"Validation successful: {validated['classification']} with confidence {validated['confidence']}")
        
        print("\n✅ Classification prompts test completed successfully!")
        
    except Exception as e:
        print(f"❌ Classification prompts test failed: {e}")
