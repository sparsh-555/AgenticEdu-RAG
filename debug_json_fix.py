import json

# This is the exact response content from your debug output
response_text = '```json\n{\n    "phase": "implementation",\n    "confidence": 0.9,\n    "reasoning": "The query is asking for an explanation of the composition of an object, which relates to understanding the design and structure of the code rather than fixing an error or troubleshooting. This aligns with the forethought phase where the focus is on conceptual understanding and implementation guidance.",\n    "features": {\n        "has_error_message": false,\n        "has_broken_code": false,\n        "seeks_implementation_guidance": true,\n        "seeks_troubleshooting_help": false,\n        "query_intent_keywords": ["explain", "composition", "object"],\n        "complexity_indicators": ["composition", "object"]\n    }\n}\n```'

print("=== ORIGINAL RESPONSE ===")
print(f"Length: {len(response_text)}")
print(f"Repr: {repr(response_text)}")
print(f"Content:\n{response_text}")

print("\n=== APPLYING FIX ===")
# Apply the same fix I made to the code
response_text = response_text.strip()
print(f"After strip(): {repr(response_text[:50])}...")

if response_text.startswith('```json'):
    response_text = response_text[7:]  # Remove ```json
    print(f"After removing '```json': {repr(response_text[:50])}...")

if response_text.startswith('```'):
    response_text = response_text[3:]   # Remove ``` 
    print(f"After removing '```': {repr(response_text[:50])}...")

if response_text.endswith('```'):
    response_text = response_text[:-3]  # Remove closing ```
    print(f"After removing closing '```': {repr(response_text[-50:])}...")

response_text = response_text.strip()
print(f"Final after strip(): {repr(response_text[:50])}...")

print("\n=== JSON PARSING TEST ===")
try:
    result = json.loads(response_text)
    print(f"✅ Successfully parsed JSON!")
    print(f"Phase: {result['phase']}")
    print(f"Confidence: {result['confidence']}")
except json.JSONDecodeError as e:
    print(f"❌ JSON parsing failed: {e}")
    print(f"Error position: {e.pos}")
    print(f"First 100 chars: {repr(response_text[:100])}")
    print(f"Last 100 chars: {repr(response_text[-100:])}")
    
    # Let's also check what character is at the error position
    if hasattr(e, 'pos') and e.pos < len(response_text):
        error_char = response_text[e.pos] if e.pos < len(response_text) else "EOF"
        print(f"Character at error position {e.pos}: {repr(error_char)}")
