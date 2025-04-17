#!/usr/bin/env python3
# tests/test_local_llm.py

import sys
import os
import asyncio

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import required modules
from qt_sections.llm_manager import llm_config
from llms.client import (
    call_llm_async, call_llm_sync, 
    call_llm_async_json, call_llm_sync_json,
    LOCAL_LLM_CONFIG
)

async def test_local_llm():
    """Test local LLM integration by making a few test calls."""
    print("Testing Local LLM Integration")
    print("=" * 50)
    
    # Enable local LLM mode
    llm_config.use_local_llms = True
    LOCAL_LLM_CONFIG["enabled"] = True
    
    # Current configuration
    print(f"Local LLM Enabled: {llm_config.use_local_llms}")
    print(f"Text Endpoint: {llm_config.local_llm_endpoints['text']}")
    print(f"Text Model: {llm_config.local_llm_models['text']}")
    print(f"JSON Model: {llm_config.local_llm_models['json']}")
    print(f"Vision Model: {llm_config.local_llm_models['vision']}")
    print("=" * 50)
    
    # Test 1: Simple text completion
    print("\nTest 1: Simple text completion")
    try:
        model = f"local-{llm_config.local_llm_models['text']}"
        prompt = "What is the capital of France?"
        print(f"Calling model: {model}")
        print(f"Prompt: {prompt}")
        
        response = await call_llm_async(prompt, model=model, max_tokens=100)
        print(f"Response:\n{response}")
        print("Test 1: SUCCESS")
    except Exception as e:
        print(f"Test 1 failed: {str(e)}")
    
    # Test 2: JSON completion
    print("\nTest 2: JSON completion")
    try:
        model = f"local-{llm_config.local_llm_models['json']}"
        prompt = "List the capitals of 3 European countries. Return as JSON with country names as keys and capitals as values."
        print(f"Calling model: {model}")
        print(f"Prompt: {prompt}")
        
        response = await call_llm_async_json(prompt, model=model, max_tokens=200)
        print(f"Response:\n{response}")
        print("Test 2: SUCCESS")
    except Exception as e:
        print(f"Test 2 failed: {str(e)}")
    
    print("\nTests completed.")

if __name__ == "__main__":
    # Check if the local LLM endpoint is set through environment variable
    endpoint = os.environ.get("LOCAL_TEXT_LLM_ENDPOINT")
    if not endpoint:
        print("Warning: LOCAL_TEXT_LLM_ENDPOINT environment variable is not set.")
        print("Using default: http://localhost:11434/v1")
    
    # Check if the local LLM is already running
    import requests
    try:
        endpoint = llm_config.local_llm_endpoints["text"]
        response = requests.get(f"{endpoint}/models")
        if response.status_code == 200:
            print(f"Local LLM server is running at {endpoint}")
        else:
            print(f"Warning: Local LLM server returned status code {response.status_code}")
    except Exception as e:
        print(f"Warning: Could not connect to local LLM server: {str(e)}")
        print("Please make sure a local LLM server like Ollama is running")
    
    # Run the async test
    asyncio.run(test_local_llm()) 