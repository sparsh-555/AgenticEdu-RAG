#!/usr/bin/env python3
"""
CS1QA Dataset Analysis Report
Analyzes the first 50 queries to understand routing accuracy issues
"""

import json
from collections import defaultdict

def analyze_cs1qa_first_50():
    # Read the JSON file
    with open('/Users/sparshjain/Documents/GitHub/AgenticEdu-RAG/cs1qa_full_processed/cs1qa_srl_labeled_with_ground_truth.json', 'r') as f:
        data = json.load(f)
    
    # Extract first 50 queries
    first_50 = data[:50]
    
    print("=" * 80)
    print("CS1QA DATASET ANALYSIS - FIRST 50 QUERIES")
    print("=" * 80)
    print()
    
    # Basic distribution
    implementation_count = 0
    debugging_count = 0
    
    implementation_queries = []
    debugging_queries = []
    confusing_implementation = []
    
    for i, query in enumerate(first_50):
        ground_truth = query.get('ground_truth_label', 'unknown')
        question = query.get('question', '')
        srl_phase = query.get('srl_label', {}).get('phase', 'unknown')
        has_error = bool(query.get('error_message'))
        
        if ground_truth == 'implementation':
            implementation_count += 1
            implementation_queries.append((i, question, srl_phase, has_error))
            
            # Check for confusing patterns
            question_lower = question.lower()
            confusing_keywords = ['why', 'error', 'wrong', 'weird', 'not work', 'problem', 'fix', 'issue', 'bug', 'broken']
            if any(keyword in question_lower for keyword in confusing_keywords):
                confusing_implementation.append((i, question))
                
        else:
            debugging_count += 1
            debugging_queries.append((i, question, srl_phase, has_error))
    
    print(f"OVERALL DISTRIBUTION:")
    print(f"  Implementation queries: {implementation_count} ({implementation_count/50*100:.1f}%)")
    print(f"  Debugging queries: {debugging_count} ({debugging_count/50*100:.1f}%)")
    print()
    
    print("=" * 80)
    print("IMPLEMENTATION QUERIES ANALYSIS")
    print("=" * 80)
    print()
    
    print("Sample Implementation Queries (first 10):")
    print("-" * 50)
    for i, (query_id, question, srl_phase, has_error) in enumerate(implementation_queries[:10]):
        print(f"{i+1}. Query {query_id}:")
        print(f"   Question: {question}")
        print(f"   SRL Phase: {srl_phase}")
        print(f"   Has Error: {has_error}")
        print()
    
    print("Confusing Implementation Queries (likely to be misclassified):")
    print("-" * 60)
    print(f"Found {len(confusing_implementation)} implementation queries with debugging-like language:")
    print()
    
    for i, (query_id, question) in enumerate(confusing_implementation):
        print(f"{i+1}. Query {query_id}:")
        print(f"   Question: {question}")
        print(f"   Issue: Contains words like 'why', 'wrong', 'fix', etc.")
        print()
    
    print("=" * 80)
    print("DEBUGGING QUERIES ANALYSIS")
    print("=" * 80)
    print()
    
    print("Sample Debugging Queries (first 8):")
    print("-" * 50)
    for i, (query_id, question, srl_phase, has_error) in enumerate(debugging_queries[:8]):
        print(f"{i+1}. Query {query_id}:")
        print(f"   Question: {question}")
        print(f"   SRL Phase: {srl_phase}")
        print(f"   Has Error: {has_error}")
        print()
    
    print("=" * 80)
    print("KEY FINDINGS & MISCLASSIFICATION PATTERNS")
    print("=" * 80)
    print()
    
    print("🔍 MAIN ISSUES CAUSING MISCLASSIFICATION:")
    print()
    
    print("1. CONCEPTUAL 'WHY' QUESTIONS:")
    print("   - Implementation queries asking 'why' or 'explain' are likely misclassified")
    print("   - Example: 'Please explain why you used the if statement in task2'")
    print("   - These are about understanding code design, NOT fixing bugs")
    print()
    
    print("2. AMBIGUOUS LANGUAGE:")
    print("   - Words like 'wrong', 'weird', 'fix' appear in implementation contexts")
    print("   - Example: 'If I enter the number, whether it is wrong or correct...'")
    print("   - Context matters: asking about logic vs reporting errors")
    print()
    
    print("3. IMPLEMENTATION GUIDANCE WITH DEBUGGING VOCABULARY:")
    print("   - Students use debugging-like language for implementation questions")
    print("   - Example: 'How should I fix this logic?' vs 'This code is broken'")
    print("   - Intent is learning, not troubleshooting")
    print()
    
    print("4. PERFORMANCE QUESTIONS:")
    print("   - Some implementation queries mention 'execution', 'timeout', etc.")
    print("   - Could be confused with debugging/performance issues")
    print("   - May need better context analysis")
    print()
    
    print("=" * 80)
    print("RECOMMENDATIONS FOR IMPROVING CLASSIFICATION")
    print("=" * 80)
    print()
    
    print("📈 SUGGESTED IMPROVEMENTS:")
    print()
    
    print("1. CONTEXT-AWARE CLASSIFICATION:")
    print("   - Analyze surrounding code context")
    print("   - Working code + conceptual question = Implementation")
    print("   - Broken code + error message = Debugging")
    print()
    
    print("2. INTENT ANALYSIS:")
    print("   - 'Why' questions about design decisions → Implementation")
    print("   - 'Why' questions about failures → Debugging")
    print("   - 'Explain' requests → Usually Implementation")
    print()
    
    print("3. ERROR MESSAGE PRESENCE:")
    print("   - Strong indicator for debugging classification")
    print("   - Implementation queries rarely have actual error messages")
    print()
    
    print("4. QUESTION PATTERN ANALYSIS:")
    print("   - 'How should I...' → Implementation")
    print("   - 'What's wrong with...' → Debugging")
    print("   - 'Can you explain...' → Implementation")
    print()
    
    print("5. CODE COMPLETENESS:")
    print("   - Complete/working code → Implementation questions")
    print("   - Incomplete/broken code → Debugging questions")
    print()
    
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    
    print(f"The 50% routing accuracy appears to be caused by:")
    print(f"• {len(confusing_implementation)}/{implementation_count} implementation queries contain debugging-like vocabulary")
    print(f"• Classifier may be over-relying on keyword matching")
    print(f"• Need better contextual understanding of student intent")
    print(f"• Implementation queries asking 'why' are particularly problematic")
    print()
    print("Focus on improving context analysis rather than just keyword detection.")

if __name__ == "__main__":
    analyze_cs1qa_first_50()