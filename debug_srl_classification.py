#!/usr/bin/env python3
"""
Debug script to test SRL classification end-to-end
"""

import sys
import os
sys.path.append('/Users/sparshjain/Documents/GitHub/AgenticEdu-RAG')

from classification.srl_classifier import SRLClassifier, ClassificationContext
from utils.logging_utils import get_logger
import json

def test_srl_classification():
    """Test SRL classification with real queries"""
    
    # Initialize classifier
    classifier = SRLClassifier()
    logger = get_logger()
    
    # Test cases from the CS1QA dataset
    test_cases = [
        {
            "query": "In Task 2, explain the composition of the created Card object.",
            "expected": "FORETHOUGHT",
            "description": "Explanation/conceptual understanding query"
        },
        {
            "query": "My binary search function returns the wrong index when searching for elements",
            "code_snippet": "def binary_search(arr, target):\n    left, right = 0, len(arr)\n    # implementation here",
            "expected": "PERFORMANCE", 
            "description": "Code not working correctly"
        },
        {
            "query": "I'm getting an IndexError: list index out of range in my sorting algorithm",
            "code_snippet": "for i in range(len(arr)):\n    if arr[i+1] < arr[i]:",
            "error_message": "IndexError: list index out of range",
            "expected": "PERFORMANCE",
            "description": "Clear error message"
        },
        {
            "query": "How do I implement a binary search algorithm in Python?",
            "expected": "FORETHOUGHT",
            "description": "Implementation guidance request"
        },
        {
            "query": "What's the best approach to solve the two-sum problem efficiently?",
            "expected": "FORETHOUGHT", 
            "description": "Strategy/approach planning"
        }
    ]
    
    print("=== SRL CLASSIFICATION DEBUG TEST ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['description']}")
        print(f"Query: {test_case['query']}")
        print(f"Expected: {test_case['expected']}")
        
        # Create classification context
        context = ClassificationContext(
            query=test_case["query"],
            code_snippet=test_case.get("code_snippet"),
            error_message=test_case.get("error_message"),
            student_level="intermediate"
        )
        
        try:
            # Perform classification
            result = classifier.classify_query(context)
            
            print(f"Actual: {result.classification.value}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Strategy: {result.classification_strategy}")
            print(f"Fallback Used: {result.fallback_used}")
            print(f"Reasoning: {result.reasoning}")
            print(f"Indicators: {result.indicators}")
            
            # Check if correct
            is_correct = result.classification.value == test_case["expected"]
            print(f"✅ CORRECT" if is_correct else f"❌ INCORRECT")
            
            if not is_correct:
                print(f"  Expected: {test_case['expected']}")
                print(f"  Got: {result.classification.value}")
                print(f"  Confidence: {result.confidence}")
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("-" * 60)
    
    # Test with the exact same prompt format used in the system
    print("\n=== TESTING WITH EXACT SYSTEM PROMPT ===")
    
    # Get the actual prompt being used
    from classification.classification_prompts import get_classification_prompt
    prompt = get_classification_prompt("standard")
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Prompt preview: {prompt[:200]}...")
    
    # Test the most problematic case
    test_query = "In Task 2, explain the composition of the created Card object."
    context = ClassificationContext(
        query=test_query,
        student_level="intermediate"
    )
    
    print(f"\nTesting query: {test_query}")
    result = classifier.classify_query(context)
    
    print(f"Result: {result.classification.value}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Fallback used: {result.fallback_used}")
    
    # Let's also test the API call directly
    print("\n=== TESTING API CALL DIRECTLY ===")
    from utils.api_utils import get_openai_client
    
    client = get_openai_client()
    
    # Construct the exact same message as the classifier
    user_parts = [f"Query: {test_query}"]
    user_parts.append(f"Student level: intermediate")
    user_message = "\n\n".join(user_parts)
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_message}
    ]
    
    print(f"System prompt length: {len(prompt)}")
    print(f"User message: {user_message}")
    
    try:
        api_response = client.create_chat_completion(
            messages=messages,
            temperature=0.1
        )
        
        print(f"API Success: {api_response.success}")
        print(f"Content: {api_response.content}")
        print(f"Response time: {api_response.response_time_ms}ms")
        
        # Try to parse the response
        from classification.srl_classifier import SRLClassifier
        test_classifier = SRLClassifier()
        parsed_result = test_classifier._parse_classification_response(api_response.content, "debug")
        
        print(f"Parsed classification: {parsed_result.classification.value}")
        print(f"Parsed confidence: {parsed_result.confidence}")
        print(f"Parsed reasoning: {parsed_result.reasoning}")
        
    except Exception as e:
        print(f"API call failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_srl_classification()