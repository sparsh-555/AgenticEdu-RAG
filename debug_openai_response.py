import asyncio
import json
import os
from openai import AsyncOpenAI
from pathlib import Path

async def test_openai_response():
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Test with your sample query
    test_query = "In Task 2, explain the composition of the created Card object."
    
    system_prompt = """You are an expert in Self-Regulated Learning (SRL) theory and programming education. 
Your task is to classify programming student queries into SRL phases based on their content and context.

SRL Phases:
1. IMPLEMENTATION (Forethought Phase): Queries about planning, strategy, "how to code", algorithm design, 
   approach selection, implementation guidance, and conceptual understanding.
2. DEBUGGING (Performance Phase): Queries about error resolution, troubleshooting, monitoring execution, 
   fixing broken code, and performance issues."""
   
    classification_prompt = f"""Classify this programming query into SRL phase:

Query: "{test_query}"
Code snippet: import random
from cs1graphics import *
[...code truncated...]


Respond with JSON:
{{
    "phase": "implementation" or "debugging",
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation",
    "features": {{
        "has_error_message": false,
        "has_broken_code": false,
        "seeks_implementation_guidance": true,
        "seeks_troubleshooting_help": false,
        "query_intent_keywords": ["list", "of", "keywords"],
        "complexity_indicators": ["list", "of", "indicators"]
    }}
}}"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": classification_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        print("=== RAW RESPONSE ===")
        print(f"Response type: {type(response)}")
        print(f"Response: {response}")
        
        print("\n=== MESSAGE CONTENT ===")
        message = response.choices[0].message
        print(f"Message type: {type(message)}")
        print(f"Has content: {hasattr(message, 'content')}")
        
        if hasattr(message, 'content'):
            print(f"Content type: {type(message.content)}")
            print(f"Content length: {len(message.content) if message.content else 'None'}")
            print(f"Content repr: {repr(message.content)}")
            print(f"Content: {message.content}")
            
            print("\n=== JSON PARSING TEST ===")
            try:
                # Apply the same fix as in cs1qa_processor.py
                response_text = message.content
                response_text = response_text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]  # Remove ```json
                if response_text.startswith('```'):
                    response_text = response_text[3:]   # Remove ``` 
                if response_text.endswith('```'):
                    response_text = response_text[:-3]  # Remove closing ```
                response_text = response_text.strip()
                
                print(f"Cleaned response (first 200 chars): {repr(response_text[:200])}")
                result = json.loads(response_text)
                print(f"✅ Successfully parsed JSON: {result}")
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print(f"First 200 chars: {repr(response_text[:200])}")
        
    except Exception as e:
        print(f"API call failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_openai_response())