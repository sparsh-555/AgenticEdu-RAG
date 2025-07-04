#!/usr/bin/env python3
"""
Test script to verify the system can be imported and initialized properly.
"""

import sys
import traceback

def test_system_import():
    """Test if the system can be imported without issues."""
    try:
        print("Testing system import...")
        from main import UnifiedAgenticEduRAGSystem
        print("✅ System import successful")
        
        print("Testing system initialization...")
        system = UnifiedAgenticEduRAGSystem(
            auto_initialize=True, 
            load_pdfs=False,
            enable_demo_mode=False
        )
        print("✅ System object created")
        
        # Wait for initialization
        import time
        max_wait = 10  # seconds
        start_time = time.time()
        
        while not system.is_initialized:
            if time.time() - start_time > max_wait:
                print("❌ System initialization timeout")
                return False
            time.sleep(0.5)
            
        print("✅ System initialized successfully")
        
        # Test a simple query
        print("Testing simple query...")
        response = system.process_query("What is a variable in programming?")
        print(f"✅ Query processed, response type: {type(response)}")
        print(f"Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_system_import()
    sys.exit(0 if success else 1)
