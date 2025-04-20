# llms/client.py
from abc import ABC, abstractmethod
import asyncio
import time
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from datetime import datetime, date
import google.generativeai as genai
import os
from dotenv import load_dotenv
import numpy as np
import anthropic
from admin.portal import secrets
import json
import io
import base64
from PIL import Image
from matplotlib.figure import Figure
import functools # Import functools for wraps
import traceback

# --- Import shared config and signals ---
import sys
# Add parent directory to path temporarily if needed, better to handle via PYTHONPATH or project structure
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Go up one level
try:
    from qt_sections.llm_manager import llm_config, llm_tracker_signals, llm_database
except ImportError:
    # Fallback if running standalone or structure is different
    print("Warning: Could not import LLM config/signals. Using dummy objects.")
    class DummyConfig:
        default_text_model = "gemini-1.5-flash-latest"
        default_json_model = "claude-3-5-sonnet-20240620"
        default_vision_model = "claude-3-5-sonnet-20240620"
        save_calls = False
    llm_config = DummyConfig()
    class DummySignals:
        call_logged = None # No-op signal
        def emit(*args, **kwargs): pass
    llm_tracker_signals = DummySignals()
    llm_tracker_signals.call_logged = DummySignals() # Mock the signal itself
    llm_database = None


# Constants
GEMINI_API_KEY = secrets.get('gemini_api_key', None) # Use .get for safety
CLAUDE_API_KEY = secrets.get('claude_api_key', None) # Get from secrets if available

# Local LLM endpoints configuration
# These can be overridden through environment variables or LLM manager
LOCAL_LLM_CONFIG = {
    "enabled": False,  # Whether to use local LLMs (False = use cloud APIs)
    "endpoints": {
        "text": os.environ.get("LOCAL_TEXT_LLM_ENDPOINT", "http://localhost:11434/v1"),
        "vision": os.environ.get("LOCAL_VISION_LLM_ENDPOINT", "http://localhost:11434/v1"),
        "json": os.environ.get("LOCAL_JSON_LLM_ENDPOINT", "http://localhost:11434/v1")
    },
    "models": {
        "text": os.environ.get("LOCAL_TEXT_LLM_MODEL", "llama3"),
        "vision": os.environ.get("LOCAL_VISION_LLM_MODEL", "llava"),
        "json": os.environ.get("LOCAL_JSON_LLM_MODEL", "llama3")
    }
}

# Sync with llm_config if available
if hasattr(llm_config, 'use_local_llms'):
    LOCAL_LLM_CONFIG["enabled"] = llm_config.use_local_llms
    
    if hasattr(llm_config, 'local_llm_endpoints'):
        LOCAL_LLM_CONFIG["endpoints"] = llm_config.local_llm_endpoints
        
    if hasattr(llm_config, 'local_llm_models'):
        LOCAL_LLM_CONFIG["models"] = llm_config.local_llm_models

# Function to determine if a model is a local or cloud model
def is_local_model(model_name):
    """Check if a model name refers to a local LLM"""
    if not LOCAL_LLM_CONFIG["enabled"]:
        return False
    
    # Check if the model is one of our configured local models
    return (model_name in LOCAL_LLM_CONFIG["models"].values() or
            model_name.startswith("local-") or
            (hasattr(llm_config, 'is_local_model') and llm_config.is_local_model(model_name)))

# Function to route requests to the appropriate endpoint
def get_llm_endpoint(model_type="text"):
    """Get the appropriate LLM endpoint based on configuration"""
    if LOCAL_LLM_CONFIG["enabled"]:
        return LOCAL_LLM_CONFIG["endpoints"].get(model_type, LOCAL_LLM_CONFIG["endpoints"]["text"])
    return None  # No local endpoint if local LLMs are disabled


if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        GEMINI_API_KEY = None # Disable Gemini if config fails
else:
     print("Warning: Gemini API key not found in secrets.")

if not CLAUDE_API_KEY:
    print("Warning: Claude API key not found in secrets.")

# Check if local LLM is enabled via environment variable
if os.environ.get("USE_LOCAL_LLM", "").lower() in ("true", "1", "yes"):
    LOCAL_LLM_CONFIG["enabled"] = True
    print(f"Local LLM mode enabled via environment variable")
    
    # Log the configuration
    print(f"Local LLM endpoints: {LOCAL_LLM_CONFIG['endpoints']}")
    print(f"Local LLM models: {LOCAL_LLM_CONFIG['models']}")


SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Default generation config (can be overridden)
DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.0,
    "max_output_tokens": 5000,
    "response_mime_type": "text/plain",
}

# Define CareFrame AI system instructions
CAREFRAME_SYSTEM_INSTRUCTIONS = """
You are CareFrame AI, an agent that streamlines clinical trials and research with end-to-end processes in human-in-loop and fully automated workflows.

Your capabilities include:
- Managing multiple studies, study sections, and study designs
- Autonomous data processing, cleaning, and analysis of statistical tests
- Executing LLM operations locally with the host

Always provide helpful, accurate, and scientifically sound responses focused on clinical research and healthcare.
"""

# --- PHI Scanning Placeholder ---
def scan_for_phi(text: str) -> tuple:
    """
    Improved PHI scanning function that uses the privacy filter if available.
    
    Returns:
        tuple: (phi_detected, phi_count, phi_types, phi_redacted)
    """
    # Check if we have access to the PHI filter
    try:
        from privacy.phi_manager import phi_config
        if phi_config and phi_config.enabled:
            # Use the comprehensive PHI filter
            block_required, phi_report, redacted_text = phi_config.check_phi(text)
            
            # Extract PHI information
            phi_detected = phi_report and phi_report['total_phi_count'] > 0
            phi_count = phi_report['total_phi_count'] if phi_report else 0
            phi_types = phi_report.get('phi_types', {}) if phi_report else {}
            
            # Determine if redaction would be needed
            phi_redacted = phi_detected
            
            return phi_detected, phi_count, phi_types, phi_redacted
    except (ImportError, AttributeError):
        # PHI filtering module not available or not configured
        pass
    
    # Basic fallback example: check for common patterns (highly inaccurate)
    if not text: 
        return False, 0, {}, False
        
    text_lower = text.lower()
    patterns = {
        "NAME": ["john doe", "jane doe", "patient name"],
        "SSN": ["ssn", "social security"],
        "PHI": ["medical record", "patient id", "dob", "date of birth"]
    }
    
    phi_count = 0
    phi_types = {}
    
    for phi_type, terms in patterns.items():
        for term in terms:
            if term in text_lower:
                if phi_type not in phi_types:
                    phi_types[phi_type] = 0
                phi_types[phi_type] += 1
                phi_count += 1
    
    phi_detected = phi_count > 0
    
    return phi_detected, phi_count, phi_types, False  # No redaction in basic mode

# --- LLM Call Tracking Decorator ---
def track_llm_call(func):
    """Decorator to track LLM call details and emit a signal."""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        timestamp = datetime.now()
        model = kwargs.get('model') or (args[1] if len(args) > 1 and isinstance(args[1], str) else 'unknown')
        prompt_arg = kwargs.get('prompt') or (args[0] if args else None)
        prompt_preview = "N/A"
        full_prompt = "N/A"
        response_preview = "N/A"
        full_response = "N/A"
        input_tokens = None
        output_tokens = None
        status = "Error"
        error_msg = ""
        phi_scanned = True
        phi_found_count = 0
        phi_types = {}
        phi_redacted = False
        response = None
        actual_return_value = None # Initialize return value holder

        try:
            # --- Prepare Prompt & Scan ---
            prompt_text_for_scan = ""
            if isinstance(prompt_arg, str):
                prompt_preview = (prompt_arg[:100] + '...') if len(prompt_arg) > 100 else prompt_arg
                prompt_text_for_scan = prompt_arg
                full_prompt = prompt_arg
            elif isinstance(prompt_arg, dict) or isinstance(prompt_arg, list):
                 try:
                     prompt_str = json.dumps(prompt_arg)
                     prompt_preview = (prompt_str[:100] + '...') if len(prompt_str) > 100 else prompt_str
                     prompt_text_for_scan = prompt_str
                     full_prompt = prompt_str
                 except Exception:
                     prompt_preview = f"<{type(prompt_arg).__name__} object>"
                     full_prompt = prompt_preview
            else:
                prompt_preview = f"<{type(prompt_arg).__name__} object>"
                full_prompt = prompt_preview

            # Scan for PHI - get detailed info
            phi_prompt_alert, phi_prompt_count, phi_prompt_types, _ = scan_for_phi(prompt_text_for_scan)
            
            # --- Check token limits if we can estimate them ---
            # Get the model's token limits
            token_limits = None
            if hasattr(llm_config, 'get_token_limits'):
                token_limits = llm_config.get_token_limits(model)
            
            # Adjust max_tokens parameter if it exists and exceeds the output limit
            if token_limits and 'max_tokens' in kwargs and kwargs['max_tokens'] > token_limits['output']:
                print(f"Warning: Reducing max_tokens from {kwargs['max_tokens']} to {token_limits['output']} for model {model}")
                kwargs['max_tokens'] = token_limits['output']
            
            # For pre-tokenization checks, we can use a rough estimate
            # Common approximation: 1 token ≈ 4 chars for English text
            if token_limits and prompt_text_for_scan:
                estimated_tokens = len(prompt_text_for_scan) // 4
                if estimated_tokens > token_limits['input']:
                    err_msg = f"Estimated input tokens ({estimated_tokens}) exceeds model limit ({token_limits['input']})"
                    raise ValueError(err_msg) # Raise error immediately

            # --- Execute Original Function ---
            response = await func(*args, **kwargs) # This is where call_gemini_async_json is called

            # --- Process Response (Modified Logic) ---
            duration = time.time() - start_time
            status = "Success"

            # Extract text response for preview/scan/log - HANDLE DICT CASE
            response_text = ""
            actual_return_value = response # Default return value is the raw response

            if isinstance(response, dict):
                # If the response IS a dict, likely from a JSON mode call
                try:
                    response_text = json.dumps(response) # Serialize for logging
                    # No need to modify actual_return_value, it's already the dict
                    print("DEBUG (Decorator): Response is a dict, serializing for log.") # Debug
                except Exception:
                    response_text = "<Dict object - serialization failed>"
                    print("DEBUG (Decorator): Response is a dict, but serialization failed.") # Debug
            elif isinstance(response, str):
                # If it's already a string
                response_text = response
                actual_return_value = response # Ensure return is the string
                print("DEBUG (Decorator): Response is a string.") # Debug
            elif hasattr(response, 'text') and response.text: # Handle Gemini GenerateContentResponse
                response_text = response.text
                actual_return_value = response.text # Return only text for standard calls
                print("DEBUG (Decorator): Response has .text attribute.") # Debug
            elif hasattr(response, 'content') and isinstance(response.content, list) and len(response.content) > 0 and hasattr(response.content[0], 'text'): # Handle Anthropic Message
                response_text = response.content[0].text
                actual_return_value = response.content[0].text # Return only text for standard calls
                print("DEBUG (Decorator): Response has .content[0].text attribute.") # Debug
            else:
                # Fallback for other object types
                try:
                    response_text = str(response)
                    print(f"DEBUG (Decorator): Response is type {type(response)}, using str().") # Debug
                except Exception:
                    response_text = f"<Object type {type(response).__name__} - str() failed>"
                    print(f"DEBUG (Decorator): Response is type {type(response)}, str() failed.") # Debug
                # Keep actual_return_value as the original response object

            response_preview = (response_text[:100] + '...') if len(response_text) > 100 else response_text
            full_response = response_text # Log the (potentially serialized) response

            # Scan response for PHI
            phi_response_alert, phi_response_count, phi_response_types, _ = scan_for_phi(response_text)

            # Combine PHI findings
            phi_found_count = phi_prompt_count + phi_response_count
            phi_types = phi_prompt_types.copy()
            for k, v in phi_response_types.items():
                phi_types[k] = phi_types.get(k, 0) + v

            # --- Extract Usage/Tokens (API specific - from original response object) ---
            # This needs to access the *original* response object, not the text
            if model.startswith("claude") and hasattr(response, 'usage'):
                input_tokens = getattr(response.usage, 'input_tokens', None)
                output_tokens = getattr(response.usage, 'output_tokens', None)
            elif model.startswith("gemini") and hasattr(response, 'usage_metadata'):
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', None)
                candidates_tokens = getattr(response.usage_metadata, 'candidates_token_count', None)
                total_usage = getattr(response.usage_metadata, 'total_token_count', None)
                if candidates_tokens is not None:
                    output_tokens = candidates_tokens
                elif total_usage is not None and input_tokens is not None:
                     output_tokens = total_usage - input_tokens
            # Add extraction logic for local LLM response if it provides usage data
            elif isinstance(response, type) and response.__name__ == 'LocalLLMResponse': # Check type name
                 if hasattr(response, 'usage'):
                     input_tokens = getattr(response.usage, 'input_tokens', None)
                     output_tokens = getattr(response.usage, 'output_tokens', None)


            # --- Determine the actual return value based on the decorated function ---
            # This logic might need adjustment based on *specific expectations* of each decorated function.
            # For now, we assume JSON functions expect dicts, others expect text.
            if func.__name__.endswith('_json'):
                 # If the original response wasn't a dict, try parsing the text now
                 if not isinstance(actual_return_value, dict):
                     try:
                         actual_return_value = json.loads(response_text)
                     except json.JSONDecodeError:
                         print(f"Warning (Decorator): Failed to parse response text as JSON for {func.__name__}")
                         # Keep actual_return_value as the text/object in case of error
            # For non-JSON functions, actual_return_value should already be text or the raw object

            # --- Emit Signal and Save if needed --- 
            # ... (keep existing signal emission and database saving logic) ...
            if llm_tracker_signals.call_logged:
                 llm_tracker_signals.call_logged.emit(
                     timestamp, model, duration, input_tokens or 0, output_tokens or 0,
                     status, error_msg, prompt_preview, response_preview,
                     phi_scanned, phi_found_count, phi_types, phi_redacted
                 )
            
            if llm_database is not None and hasattr(llm_config, 'save_calls') and llm_config.save_calls:
                 try:
                     llm_database.save_call(
                         timestamp, model, duration, input_tokens or 0, output_tokens or 0,
                         status, error_msg, full_prompt, full_response,
                         phi_scanned, phi_found_count, phi_types, phi_redacted
                     )
                 except Exception as db_error:
                     print(f"Error saving LLM call to database: {str(db_error)}")


            # --- Return the determined actual_return_value ---
            return actual_return_value

        except Exception as e:
            # ... (keep existing error handling logic) ...
            duration = time.time() - start_time
            status = "Error"
            error_msg = str(e)
            # Check if error happened before prompt scanning
            if 'prompt_text_for_scan' not in locals():
                prompt_preview="<Error before prompt processing>"
                full_prompt=prompt_preview
                phi_prompt_count=0
                phi_prompt_types={}

            traceback_str = traceback.format_exc()
            print(f"LLM Call Error ({model}): {error_msg}\n{traceback_str}")
            response_preview = f"Error: {error_msg}"
            full_response = response_preview

            # Emit Signal on Error
            if llm_tracker_signals.call_logged:
                 llm_tracker_signals.call_logged.emit(
                     timestamp, model, duration, input_tokens or 0, output_tokens or 0,
                     status, error_msg, prompt_preview, response_preview,
                     phi_scanned, phi_found_count, phi_types, phi_redacted
                 )

            # Save error to database if enabled
            if llm_database is not None and hasattr(llm_config, 'save_calls') and llm_config.save_calls:
                 try:
                     llm_database.save_call(
                         timestamp, model, duration, input_tokens or 0, output_tokens or 0,
                         status, error_msg, full_prompt, full_response,
                         phi_scanned, phi_found_count, phi_types, phi_redacted
                     )
                 except Exception as db_error:
                     print(f"Error saving LLM call error to database: {str(db_error)}")

            raise # Re-raise the exception

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        # Apply similar logic to the synchronous wrapper
        start_time = time.time()
        timestamp = datetime.now()
        # ... (rest of initial setup for sync_wrapper - keep existing) ...
        model = kwargs.get('model') or (args[1] if len(args) > 1 and isinstance(args[1], str) else 'unknown')
        prompt_arg = kwargs.get('prompt') or (args[0] if args else None)
        prompt_preview = "N/A"
        full_prompt = "N/A"
        response_preview = "N/A"
        full_response = "N/A"
        input_tokens = None
        output_tokens = None
        status = "Error"
        error_msg = ""
        phi_scanned = True
        phi_found_count = 0
        phi_types = {}
        phi_redacted = False
        response = None
        actual_return_value = None

        try:
            # ... (Prepare Prompt & Scan, Token Limit Checks - sync version - keep existing) ...
            prompt_text_for_scan = ""
            if isinstance(prompt_arg, str):
                prompt_preview = (prompt_arg[:100] + '...') if len(prompt_arg) > 100 else prompt_arg
                prompt_text_for_scan = prompt_arg
                full_prompt = prompt_arg
            elif isinstance(prompt_arg, dict) or isinstance(prompt_arg, list):
                 try:
                     prompt_str = json.dumps(prompt_arg)
                     prompt_preview = (prompt_str[:100] + '...') if len(prompt_str) > 100 else prompt_str
                     prompt_text_for_scan = prompt_str
                     full_prompt = prompt_str
                 except Exception:
                     prompt_preview = f"<{type(prompt_arg).__name__} object>"
                     full_prompt = prompt_preview
            else:
                prompt_preview = f"<{type(prompt_arg).__name__} object>"
                full_prompt = prompt_preview

            phi_prompt_alert, phi_prompt_count, phi_prompt_types, _ = scan_for_phi(prompt_text_for_scan)

            token_limits = None
            if hasattr(llm_config, 'get_token_limits'):
                token_limits = llm_config.get_token_limits(model)
            if token_limits and 'max_tokens' in kwargs and kwargs['max_tokens'] > token_limits['output']:
                 kwargs['max_tokens'] = token_limits['output']
            if token_limits and prompt_text_for_scan:
                estimated_tokens = len(prompt_text_for_scan) // 4
                if estimated_tokens > token_limits['input']:
                    err_msg = f"Estimated input tokens ({estimated_tokens}) exceeds model limit ({token_limits['input']})"
                    raise ValueError(err_msg)

            # --- Execute Original Function ---
            response = func(*args, **kwargs) # Call sync function

            # --- Process Response (Modified Logic for sync) ---
            duration = time.time() - start_time
            status = "Success"
            response_text = ""
            actual_return_value = response

            # Handle different response types
            if isinstance(response, dict):
                try:
                    response_text = json.dumps(response)
                    print("DEBUG (Decorator Sync): Response is a dict, serializing for log.") # Debug
                    # actual_return_value is already the dict
                except Exception:
                    response_text = "<Dict object - serialization failed>"
                    print("DEBUG (Decorator Sync): Response is a dict, but serialization failed.") # Debug
            elif isinstance(response, str):
                response_text = response
                actual_return_value = response
                print("DEBUG (Decorator Sync): Response is a string.") # Debug
            elif hasattr(response, 'text') and response.text:
                response_text = response.text
                actual_return_value = response.text
                print("DEBUG (Decorator Sync): Response has .text.") # Debug
            elif hasattr(response, 'content') and isinstance(response.content, list) and len(response.content) > 0 and hasattr(response.content[0], 'text'):
                response_text = response.content[0].text
                actual_return_value = response.content[0].text
                print("DEBUG (Decorator Sync): Response has .content[0].text.") # Debug
            else:
                try:
                    response_text = str(response)
                    print(f"DEBUG (Decorator Sync): Response is type {type(response)}, using str().") # Debug
                except Exception:
                    response_text = f"<Object type {type(response).__name__} - str() failed>"
                    print(f"DEBUG (Decorator Sync): Response is type {type(response)}, str() failed.") # Debug


            response_preview = (response_text[:100] + '...') if len(response_text) > 100 else response_text
            full_response = response_text

            # Scan response text
            phi_response_alert, phi_response_count, phi_response_types, _ = scan_for_phi(response_text)
            phi_found_count = phi_prompt_count + phi_response_count
            phi_types = phi_prompt_types.copy()
            for k, v in phi_response_types.items():
                 phi_types[k] = phi_types.get(k, 0) + v

            # --- Extract Usage/Tokens (sync version) ---
            if model.startswith("claude") and hasattr(response, 'usage'):
                 input_tokens = getattr(response.usage, 'input_tokens', None)
                 output_tokens = getattr(response.usage, 'output_tokens', None)
            elif model.startswith("gemini") and hasattr(response, 'usage_metadata'):
                 input_tokens = getattr(response.usage_metadata, 'prompt_token_count', None)
                 candidates_tokens = getattr(response.usage_metadata, 'candidates_token_count', None)
                 total_usage = getattr(response.usage_metadata, 'total_token_count', None)
                 if candidates_tokens is not None: output_tokens = candidates_tokens
                 elif total_usage is not None and input_tokens is not None: output_tokens = total_usage - input_tokens
            elif isinstance(response, type) and response.__name__ == 'LocalLLMResponse':
                 if hasattr(response, 'usage'):
                     input_tokens = getattr(response.usage, 'input_tokens', None)
                     output_tokens = getattr(response.usage, 'output_tokens', None)


            # --- Final check on return value type (sync version) ---
            if func.__name__.endswith('_json'):
                 if not isinstance(actual_return_value, dict):
                     try:
                         actual_return_value = json.loads(response_text)
                     except json.JSONDecodeError:
                         print(f"Warning (Decorator Sync): Failed final JSON parse for {func.__name__}")
                         actual_return_value = response_text # Return text on failure


            # --- Emit Signal and Save (sync version) ---
            # ... (keep existing signal/db logic) ...
            if llm_tracker_signals.call_logged:
                 llm_tracker_signals.call_logged.emit(
                     timestamp, model, duration, input_tokens or 0, output_tokens or 0,
                     status, error_msg, prompt_preview, response_preview,
                     phi_scanned, phi_found_count, phi_types, phi_redacted
                 )
            if llm_database is not None and hasattr(llm_config, 'save_calls') and llm_config.save_calls:
                 try:
                     llm_database.save_call(
                         timestamp, model, duration, input_tokens or 0, output_tokens or 0,
                         status, error_msg, full_prompt, full_response,
                         phi_scanned, phi_found_count, phi_types, phi_redacted
                     )
                 except Exception as db_error:
                     print(f"Error saving LLM call to database: {str(db_error)}")


            return actual_return_value

        except Exception as e:
            # ... (rest of sync error handling logic - similar to async) ...
            duration = time.time() - start_time
            status = "Error"
            error_msg = str(e)
            if 'prompt_text_for_scan' not in locals():
                 prompt_preview="<Error before prompt processing>"
                 full_prompt=prompt_preview
                 phi_prompt_count=0
                 phi_prompt_types={}

            traceback_str = traceback.format_exc()
            print(f"LLM Sync Call Error ({model}): {error_msg}\n{traceback_str}")
            response_preview = f"Error: {error_msg}"
            full_response = response_preview

            if llm_tracker_signals.call_logged:
                 llm_tracker_signals.call_logged.emit(
                     timestamp, model, duration, input_tokens or 0, output_tokens or 0,
                     status, error_msg, prompt_preview, response_preview,
                     phi_scanned, phi_found_count, phi_types, phi_redacted
                 )

            if llm_database is not None and hasattr(llm_config, 'save_calls') and llm_config.save_calls:
                 try:
                     llm_database.save_call(
                         timestamp, model, duration, input_tokens or 0, output_tokens or 0,
                         status, error_msg, full_prompt, full_response,
                         phi_scanned, phi_found_count, phi_types, phi_redacted
                     )
                 except Exception as db_error:
                     print(f"Error saving LLM sync call error to database: {str(db_error)}")

            raise

    # Return the correct wrapper based on whether the original function is async
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

# --- Utility Functions (Keep as is) ---
def convert_numpy_types(obj):
    # ... (keep existing implementation)
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Handle pandas Timestamp objects
    elif str(type(obj)).endswith("pandas._libs.tslibs.timestamps.Timestamp'>"):
        return obj.isoformat()
    # Handle Python date objects
    elif isinstance(obj, (date, datetime)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


# --- Core API Call Functions (Apply Decorator) ---

@track_llm_call
def call_claude_sync(prompt: Union[str, Dict], model: str, max_tokens: int = 5000):
    """
    Call Claude API synchronously. [Decorated for Tracking]
    """
    if not CLAUDE_API_KEY: raise ValueError("Claude API Key not configured.")
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    # ... (rest of the original function logic)
    system_content = CAREFRAME_SYSTEM_INSTRUCTIONS
    messages = []
    if isinstance(prompt, str):
        if "who are you" in prompt.lower() or "what are you" in prompt.lower():
            messages = [{"role": "user", "content": prompt}]
        else:
            if not "DO NOT FORMAT YOUR RESPONSE AS JSON" in prompt:
                prompt += "\n\nIMPORTANT: DO NOT FORMAT YOUR RESPONSE AS JSON. Provide a plain text response with no special formatting."
            messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, dict) or isinstance(prompt, list): # Assume messages format
        messages = prompt
        system_content = prompt.get("system", CAREFRAME_SYSTEM_INSTRUCTIONS) # Allow overriding system prompt
    else:
         raise TypeError(f"Unsupported prompt type for Claude: {type(prompt)}")


    # Note: The decorator expects the function to return the raw response object
    # for usage extraction. The decorator will handle extracting the final text.
    response = client.messages.create(
        max_tokens=max_tokens,
        messages=messages,
        model=model,
        system=system_content
    )
    return response # Return the full response object

@track_llm_call
async def call_claude_async(prompt: Union[str, Dict], model: str, max_tokens: int = 5000):
    """
    Call Claude API asynchronously. [Decorated for Tracking]
    """
    if not CLAUDE_API_KEY: raise ValueError("Claude API Key not configured.")
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    # ... (rest of the original function logic, similar to sync)
    system_content = CAREFRAME_SYSTEM_INSTRUCTIONS
    messages = []
    if isinstance(prompt, str):
        if "who are you" in prompt.lower() or "what are you" in prompt.lower():
            messages = [{"role": "user", "content": prompt}]
        else:
            if not "DO NOT FORMAT YOUR RESPONSE AS JSON" in prompt:
                prompt += "\n\nIMPORTANT: DO NOT FORMAT YOUR RESPONSE AS JSON. Provide a plain text response with no special formatting."
            messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, dict) or isinstance(prompt, list): # Assume messages format
        messages = prompt
        system_content = prompt.get("system", CAREFRAME_SYSTEM_INSTRUCTIONS)
    else:
        raise TypeError(f"Unsupported prompt type for Claude: {type(prompt)}")

    # Run synchronous client call in a thread
    response = await asyncio.to_thread(
        client.messages.create,
        max_tokens=max_tokens,
        messages=messages,
        model=model,
        system=system_content
    )
    return response # Return the full response object

@track_llm_call
def call_gemini_sync(prompt: str, model: str, temperature: float = 0.0, max_tokens: int = 5000):
    """
    Call Gemini API synchronously. [Decorated for Tracking]
    """
    if not GEMINI_API_KEY: raise ValueError("Gemini API Key not configured.")
    # ... (rest of the original function logic)
    system_prompt = CAREFRAME_SYSTEM_INSTRUCTIONS # Start with base instructions
    if "who are you" in prompt.lower() or "what are you" in prompt.lower():
        prompt = f"{system_prompt}\n\nUser question: {prompt}\n\nPlease respond in the first person as CareFrame AI."
    else:
        json_keywords = ["Return the results in the following JSON format", "Format your response as JSON", "Return a JSON object", "JSON format", "JSON output"]
        expects_json = any(keyword in prompt for keyword in json_keywords)

        if not expects_json and "DO NOT FORMAT YOUR RESPONSE AS JSON" not in prompt:
             prompt = f"{system_prompt}\n\n{prompt}\n\nIMPORTANT: DO NOT FORMAT YOUR RESPONSE AS JSON. Provide a plain text response with no special formatting."
        else:
             prompt = f"{system_prompt}\n\n{prompt}"


    model_instance = genai.GenerativeModel(
        model_name=model,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "text/plain",
        },
        safety_settings=SAFETY_SETTINGS
    )
    response = model_instance.generate_content(prompt)
    return response # Return the full response object

@track_llm_call
async def call_gemini_async(prompt: str, model: str, temperature: float = 0.0, max_tokens: int = 5000):
    """
    Call Gemini API asynchronously. [Decorated for Tracking]
    """
    if not GEMINI_API_KEY: raise ValueError("Gemini API Key not configured.")
    # ... (rest of the original function logic, similar to sync)
    system_prompt = CAREFRAME_SYSTEM_INSTRUCTIONS
    if "who are you" in prompt.lower() or "what are you" in prompt.lower():
        prompt = f"{system_prompt}\n\nUser question: {prompt}\n\nPlease respond in the first person as CareFrame AI."
    else:
        json_keywords = ["Return the results in the following JSON format", "Format your response as JSON", "Return a JSON object", "JSON format", "JSON output"]
        expects_json = any(keyword in prompt for keyword in json_keywords)

        if not expects_json and "DO NOT FORMAT YOUR RESPONSE AS JSON" not in prompt:
             prompt = f"{system_prompt}\n\n{prompt}\n\nIMPORTANT: DO NOT FORMAT YOUR RESPONSE AS JSON. Provide a plain text response with no special formatting."
        else:
             prompt = f"{system_prompt}\n\n{prompt}"

    model_instance = genai.GenerativeModel(
        model_name=model,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "text/plain",
        },
        safety_settings=SAFETY_SETTINGS
    )
    response = await asyncio.to_thread(model_instance.generate_content, prompt)
    return response # Return the full response object


# --- JSON Specific Calls (Apply Decorator) ---

@track_llm_call
async def call_claude_async_json(prompt: Union[str, Dict], model: str, max_tokens: int = 5000):
    """
    Call Claude API asynchronously optimized for JSON responses. [Decorated for Tracking]
    """
    if not CLAUDE_API_KEY: raise ValueError("Claude API Key not configured.")
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    # ... (rest of original logic)
    system_content = "You are a helpful assistant that returns information in JSON format."
    messages = []
    if isinstance(prompt, str):
        if "JSON format" not in prompt and "JSON output" not in prompt:
             prompt += "\n\nIMPORTANT: Format your entire response as a valid JSON object."
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, (dict, list)):
         messages = prompt
         # Allow overriding system prompt if provided in dict
         system_content = prompt.get("system", system_content) if isinstance(prompt, dict) else system_content
    else:
        raise TypeError(f"Unsupported prompt type for Claude JSON: {type(prompt)}")


    response = await asyncio.to_thread(
        client.messages.create,
        max_tokens=max_tokens,
        messages=messages,
        model=model,
        system=system_content
    )
    return response # Return the full response object

@track_llm_call
def call_claude_sync_json(prompt: Union[str, Dict], model: str, max_tokens: int = 5000):
    """
    Call Claude API synchronously optimized for JSON responses. [Decorated for Tracking]
    """
    if not CLAUDE_API_KEY: raise ValueError("Claude API Key not configured.")
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    # ... (rest of original logic, similar to async JSON)
    system_content = "You are a helpful assistant that returns information in JSON format."
    messages = []
    if isinstance(prompt, str):
        if "JSON format" not in prompt and "JSON output" not in prompt:
             prompt += "\n\nIMPORTANT: Format your entire response as a valid JSON object."
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, (dict, list)):
         messages = prompt
         system_content = prompt.get("system", system_content) if isinstance(prompt, dict) else system_content
    else:
        raise TypeError(f"Unsupported prompt type for Claude JSON: {type(prompt)}")

    response = client.messages.create(
        max_tokens=max_tokens,
        messages=messages,
        model=model,
        system=system_content
    )
    return response # Return the full response object

@track_llm_call
async def call_gemini_async_json(prompt: str, model: str, temperature: float = 0.0, max_tokens: int = 5000):
    """
    Call Gemini API asynchronously optimized for JSON responses. [Decorated for Tracking]
    """
    if not GEMINI_API_KEY: raise ValueError("Gemini API Key not configured.")
    # ... (rest of original logic)
    # Ensure JSON output is requested
    if "JSON format" not in prompt and "JSON output" not in prompt:
        prompt += "\n\nIMPORTANT: Format your entire response as a valid JSON object, ensuring it can be parsed with json.loads(). Only output the JSON object itself, with no surrounding text or markdown."

    model_instance = genai.GenerativeModel(
        model_name=model,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
             # Try requesting JSON directly if model supports it (e.g., Gemini 1.5)
             "response_mime_type": "application/json",
        },
        safety_settings=SAFETY_SETTINGS
    )
    response = await asyncio.to_thread(model_instance.generate_content, prompt)

    # Decorator expects the full response object. Cleaning happens in the decorator if needed.
    return response

@track_llm_call
def call_gemini_sync_json(prompt: str, model: str, temperature: float = 0.0, max_tokens: int = 5000):
    """
    Call Gemini API synchronously optimized for JSON responses. [Decorated for Tracking]
    """
    if not GEMINI_API_KEY: raise ValueError("Gemini API Key not configured.")
    # ... (rest of original logic, similar to async JSON)
    if "JSON format" not in prompt and "JSON output" not in prompt:
        prompt += "\n\nIMPORTANT: Format your entire response as a valid JSON object, ensuring it can be parsed with json.loads(). Only output the JSON object itself, with no surrounding text or markdown."

    model_instance = genai.GenerativeModel(
        model_name=model,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
        },
        safety_settings=SAFETY_SETTINGS
    )
    response = model_instance.generate_content(prompt)
    # Decorator expects the full response object.
    return response


# --- Vision Calls (Apply Decorator) ---

@track_llm_call
async def call_claude_with_image_async(prompt: str, image_data, model: Optional[str] = None, max_tokens: int = 5000):
    """
    Call Claude API asynchronously with text and image content. [Decorated for Tracking]
    """
    if not CLAUDE_API_KEY: raise ValueError("Claude API Key not configured.")
    if not model: 
        model = llm_config.default_vision_model
        print(f"No model specified for vision call, using default: {model}")
    
    # Validate model is a Claude model
    if not model.startswith("claude"):
        raise ValueError(f"call_claude_with_image_async requires a Claude model. Got: {model}")
        
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    # --- Process image data (Keep original logic) ---
    image_bytes = None
    media_type = "image/png" # Default
    if isinstance(image_data, Figure):
        buf = io.BytesIO()
        image_data.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        image_bytes = buf.getvalue()
    elif isinstance(image_data, Image.Image):
        buf = io.BytesIO()
        image_data.save(buf, format='PNG')
        buf.seek(0)
        image_bytes = buf.getvalue()
    elif isinstance(image_data, bytes):
        image_bytes = image_data
        # Try to guess media type? For now assume png/jpeg are most common
        # You might need a library like 'python-magic' for robust detection
    elif isinstance(image_data, str):
        if image_data.startswith('<svg') or '<!doctype svg' in image_data.lower():
            try:
                from cairosvg import svg2png
                image_bytes = svg2png(bytestring=image_data.encode('utf-8'))
            except ImportError:
                print("Error: cairosvg package needed for SVG conversion.")
                raise ValueError("SVG conversion requires cairosvg.")
            except Exception as e:
                raise ValueError(f"Failed to convert SVG: {e}")
        else: # Assume base64
             try:
                 image_bytes = base64.b64decode(image_data)
             except Exception:
                 raise ValueError("Invalid base64 image string")
    else:
        raise ValueError(f"Unsupported image data type: {type(image_data)}")

    if not image_bytes:
         raise ValueError("Image data processing failed.")

    # --- Prepare message ---
    system_content = CAREFRAME_SYSTEM_INSTRUCTIONS
    message_content = [
        {"type": "text", "text": prompt},
        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64.b64encode(image_bytes).decode('utf-8')}}
    ]

    response = await asyncio.to_thread(
        client.messages.create,
        max_tokens=max_tokens,
        model=model,
        system=system_content,
        messages=[{"role": "user", "content": message_content}]
    )
    return response # Return full response

@track_llm_call
def call_claude_with_image_sync(prompt: str, image_data, model: Optional[str] = None, max_tokens: int = 5000):
    """
    Call Claude API synchronously with text and image content. [Decorated for Tracking]
    """
    if not CLAUDE_API_KEY: raise ValueError("Claude API Key not configured.")
    if not model: 
        model = llm_config.default_vision_model
        print(f"No model specified for vision call, using default: {model}")
    
    # Validate model is a Claude model
    if not model.startswith("claude"):
        raise ValueError(f"call_claude_with_image_sync requires a Claude model. Got: {model}")
        
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    # --- Process image data (Keep original logic - identical to async) ---
    image_bytes = None
    media_type = "image/png" # Default
    # ... (identical image processing logic as in async version) ...
    if isinstance(image_data, Figure):
        buf = io.BytesIO()
        image_data.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        image_bytes = buf.getvalue()
    elif isinstance(image_data, Image.Image):
        buf = io.BytesIO()
        image_data.save(buf, format='PNG')
        buf.seek(0)
        image_bytes = buf.getvalue()
    elif isinstance(image_data, bytes):
        image_bytes = image_data
    elif isinstance(image_data, str):
        if image_data.startswith('<svg') or '<!doctype svg' in image_data.lower():
            try:
                from cairosvg import svg2png
                image_bytes = svg2png(bytestring=image_data.encode('utf-8'))
            except ImportError:
                print("Error: cairosvg package needed for SVG conversion.")
                raise ValueError("SVG conversion requires cairosvg.")
            except Exception as e:
                raise ValueError(f"Failed to convert SVG: {e}")
        else: # Assume base64
             try:
                 image_bytes = base64.b64decode(image_data)
             except Exception:
                 raise ValueError("Invalid base64 image string")
    else:
        raise ValueError(f"Unsupported image data type: {type(image_data)}")

    if not image_bytes:
         raise ValueError("Image data processing failed.")

    # --- Prepare message ---
    system_content = CAREFRAME_SYSTEM_INSTRUCTIONS
    message_content = [
        {"type": "text", "text": prompt},
        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64.b64encode(image_bytes).decode('utf-8')}}
    ]

    response = client.messages.create(
        max_tokens=max_tokens,
        model=model,
        system=system_content,
        messages=[{"role": "user", "content": message_content}]
    )
    return response # Return full response

# --- Add Gemini Vision Support ---
@track_llm_call
async def call_gemini_with_image_async(prompt: str, image_data, model: Optional[str] = None, max_tokens: int = 5000, temperature: float = 0.0):
    """
    Call Gemini API asynchronously with text and image content. [Decorated for Tracking]
    """
    if not GEMINI_API_KEY: raise ValueError("Gemini API Key not configured.")
    if not model: 
        model = llm_config.default_vision_model
        print(f"No model specified for vision call, using default: {model}")
    
    # Validate model is a Gemini model
    if not model.startswith("gemini"):
        raise ValueError(f"call_gemini_with_image_async requires a Gemini model. Got: {model}")
    
    # --- Process image data ---
    image_bytes = None
    if isinstance(image_data, Figure):
        buf = io.BytesIO()
        image_data.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        image_bytes = buf.getvalue()
    elif isinstance(image_data, Image.Image):
        buf = io.BytesIO()
        image_data.save(buf, format='PNG')
        buf.seek(0)
        image_bytes = buf.getvalue()
    elif isinstance(image_data, bytes):
        image_bytes = image_data
    elif isinstance(image_data, str):
        if image_data.startswith('<svg') or '<!doctype svg' in image_data.lower():
            try:
                from cairosvg import svg2png
                image_bytes = svg2png(bytestring=image_data.encode('utf-8'))
            except ImportError:
                print("Error: cairosvg package needed for SVG conversion.")
                raise ValueError("SVG conversion requires cairosvg.")
            except Exception as e:
                raise ValueError(f"Failed to convert SVG: {e}")
        else:  # Assume base64
            try:
                image_bytes = base64.b64decode(image_data)
            except Exception:
                raise ValueError("Invalid base64 image string")
    else:
        raise ValueError(f"Unsupported image data type: {type(image_data)}")

    if not image_bytes:
        raise ValueError("Image data processing failed.")
    
    # Prepare system prompt and full prompt
    system_prompt = CAREFRAME_SYSTEM_INSTRUCTIONS
    full_prompt = f"{system_prompt}\n\n{prompt}"
    
    # Create Gemini model with appropriate config
    model_instance = genai.GenerativeModel(
        model_name=model,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "text/plain",
        },
        safety_settings=SAFETY_SETTINGS
    )
    
    # Create multipart content
    multipart_content = [
        full_prompt,
        {"mime_type": "image/png", "data": image_bytes}
    ]
    
    # Generate content with image
    response = await asyncio.to_thread(model_instance.generate_content, multipart_content)
    return response # Return full response

@track_llm_call
def call_gemini_with_image_sync(prompt: str, image_data, model: Optional[str] = None, max_tokens: int = 5000, temperature: float = 0.0):
    """
    Call Gemini API synchronously with text and image content. [Decorated for Tracking]
    """
    if not GEMINI_API_KEY: raise ValueError("Gemini API Key not configured.")
    if not model:
        model = llm_config.default_vision_model
        print(f"No model specified for vision call, using default: {model}")
    
    # Validate model is a Gemini model
    if not model.startswith("gemini"):
        raise ValueError(f"call_gemini_with_image_sync requires a Gemini model. Got: {model}")
    
    # --- Process image data (same as async version) ---
    image_bytes = None
    if isinstance(image_data, Figure):
        buf = io.BytesIO()
        image_data.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        image_bytes = buf.getvalue()
    elif isinstance(image_data, Image.Image):
        buf = io.BytesIO()
        image_data.save(buf, format='PNG')
        buf.seek(0)
        image_bytes = buf.getvalue()
    elif isinstance(image_data, bytes):
        image_bytes = image_data
    elif isinstance(image_data, str):
        if image_data.startswith('<svg') or '<!doctype svg' in image_data.lower():
            try:
                from cairosvg import svg2png
                image_bytes = svg2png(bytestring=image_data.encode('utf-8'))
            except ImportError:
                print("Error: cairosvg package needed for SVG conversion.")
                raise ValueError("SVG conversion requires cairosvg.")
            except Exception as e:
                raise ValueError(f"Failed to convert SVG: {e}")
        else:  # Assume base64
            try:
                image_bytes = base64.b64decode(image_data)
            except Exception:
                raise ValueError("Invalid base64 image string")
    else:
        raise ValueError(f"Unsupported image data type: {type(image_data)}")

    if not image_bytes:
        raise ValueError("Image data processing failed.")
    
    # Prepare system prompt and full prompt
    system_prompt = CAREFRAME_SYSTEM_INSTRUCTIONS
    full_prompt = f"{system_prompt}\n\n{prompt}"
    
    # Create Gemini model with appropriate config
    model_instance = genai.GenerativeModel(
        model_name=model,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "text/plain",
        },
        safety_settings=SAFETY_SETTINGS
    )
    
    # Create multipart content
    multipart_content = [
        full_prompt,
        {"mime_type": "image/png", "data": image_bytes}
    ]
    
    # Generate content with image
    response = model_instance.generate_content(multipart_content)
    return response # Return full response

# --- Generic Vision Call (Uses configured vision model) ---
async def call_llm_vision_async(prompt: str, image_data: Any, model: Optional[str] = None, max_retries: int = 3, **kwargs) -> str:
    """
    Generic asynchronous LLM call function for vision tasks.
    Uses configured default vision model if 'model' is not provided.
    Supports both Claude and Gemini vision models.
    """
    target_model = model or llm_config.default_vision_model
    
    if not target_model:
        raise ValueError("No vision model specified or configured as default")
        
    prompt = convert_numpy_types(prompt) # Prompt unlikely to be numpy, but image_data might be

    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1: await asyncio.sleep(0.1 * (2 ** (attempt-1)))

            # Check if we should use a local LLM for vision tasks
            if LOCAL_LLM_CONFIG["enabled"] and is_local_model(target_model):
                local_model = target_model
                if target_model.startswith("local-"):
                    # Map model name to actual local model name (strip prefix)
                    local_model = target_model[6:]
                elif target_model in LOCAL_LLM_CONFIG["models"].values():
                    # Already a bare model name
                    pass
                else:
                    # Use default vision model
                    local_model = LOCAL_LLM_CONFIG["models"]["vision"]
                
                # Call local LLM vision API
                response = await call_local_llm_with_image_async(prompt, image_data, model=local_model, **kwargs)
                # Extract text from response
                result_text = ""
                if hasattr(response, 'content') and isinstance(response.content, list) and len(response.content) > 0 and hasattr(response.content[0], 'text'):
                    result_text = response.content[0].text
                else:
                    result_text = str(response)
                return result_text
            # Choose the right vision function based on model prefix
            elif target_model.startswith("claude"):
                response = await call_claude_with_image_async(prompt, image_data, model=target_model, **kwargs)
            elif target_model.startswith("gemini"):
                response = await call_gemini_with_image_async(prompt, image_data, model=target_model, **kwargs)
            else:
                raise ValueError(f"Unsupported vision model: {target_model}. Must be a Claude, Gemini, or local model.")

            # Extract text from response
            result_text = ""
            if isinstance(response, str):
                result_text = response
            elif hasattr(response, 'text'):  # Gemini response format
                result_text = response.text
            elif hasattr(response, 'content') and isinstance(response.content, list) and len(response.content) > 0 and hasattr(response.content[0], 'text'):
                result_text = response.content[0].text  # Claude response format
            else:
                result_text = str(response)  # Fallback

            return result_text
             
        except Exception as e:
            print(f"LLM Vision call attempt {attempt} failed for model {target_model}: {e}")
            if attempt == max_retries: raise

def call_llm_vision_sync(prompt: str, image_data: Any, model: Optional[str] = None, max_retries: int = 3, **kwargs) -> str:
    """
    Generic synchronous LLM call function for vision tasks.
    Uses configured default vision model if 'model' is not provided.
    Supports both Claude and Gemini vision models.
    """
    target_model = model or llm_config.default_vision_model
    
    if not target_model:
        raise ValueError("No vision model specified or configured as default")
        
    prompt = convert_numpy_types(prompt)

    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1: time.sleep(0.1 * (2 ** (attempt-1)))

            # First check if we should use a local LLM
            if LOCAL_LLM_CONFIG["enabled"] and is_local_model(target_model):
                local_model = target_model
                if target_model.startswith("local-"):
                    # Strip prefix
                    local_model = target_model[6:]
                elif target_model in LOCAL_LLM_CONFIG["models"].values():
                    # Already a bare model name
                    pass
                else:
                    # Use default vision model
                    local_model = LOCAL_LLM_CONFIG["models"]["vision"]
                
                # No synchronous local LLM vision implementation yet, so create one
                # by running the async version in an event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # If no event loop exists in this thread, create one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Call the async version in the event loop
                response = loop.run_until_complete(
                    call_local_llm_with_image_async(prompt, image_data, model=local_model, **kwargs)
                )
                
                # Extract text from response
                result_text = ""
                if hasattr(response, 'content') and isinstance(response.content, list) and len(response.content) > 0 and hasattr(response.content[0], 'text'):
                    result_text = response.content[0].text
                else:
                    result_text = str(response)
                return result_text
            # Choose the right vision function based on model prefix
            elif target_model.startswith("claude"):
                response = call_claude_with_image_sync(prompt, image_data, model=target_model, **kwargs)
            elif target_model.startswith("gemini"):
                response = call_gemini_with_image_sync(prompt, image_data, model=target_model, **kwargs)
            else:
                raise ValueError(f"Unsupported vision model: {target_model}. Must be a Claude, Gemini, or local model.")

            # Extract text from response
            result_text = ""
            if isinstance(response, str):
                result_text = response
            elif hasattr(response, 'text'):  # Gemini response format
                result_text = response.text
            elif hasattr(response, 'content') and isinstance(response.content, list) and len(response.content) > 0 and hasattr(response.content[0], 'text'):
                result_text = response.content[0].text  # Claude response format
            else:
                result_text = str(response)  # Fallback

            return result_text
             
        except Exception as e:
            print(f"LLM Vision call attempt {attempt} failed for model {target_model}: {e}")
            if attempt == max_retries: raise

# --- Add Claude Multi-Image Support ---
@track_llm_call
async def call_claude_with_multiple_images_async(prompt: str, image_data_list, model: Optional[str] = None, max_tokens: int = 5000):
    """
    Call Claude API asynchronously with text and multiple images. [Decorated for Tracking]
    """
    if not CLAUDE_API_KEY: raise ValueError("Claude API Key not configured.")
    if not model:
        model = llm_config.default_vision_model
        print(f"No model specified for vision call, using default: {model}")
    
    # Validate model is a Claude model
    if not model.startswith("claude"):
        raise ValueError(f"call_claude_with_multiple_images_async requires a Claude model. Got: {model}")
        
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    message_content = [{"type": "text", "text": prompt}]

    for image_data in image_data_list:
        image_bytes = None
        media_type = "image/png"
        # Process each image
        if isinstance(image_data, Figure):
            buf = io.BytesIO()
            image_data.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            image_bytes = buf.getvalue()
        elif isinstance(image_data, Image.Image):
            buf = io.BytesIO()
            image_data.save(buf, format='PNG')
            buf.seek(0)
            image_bytes = buf.getvalue()
        elif isinstance(image_data, bytes):
            image_bytes = image_data
        elif isinstance(image_data, str):
            if image_data.startswith('<svg') or '<!doctype svg' in image_data.lower():
                try:
                    from cairosvg import svg2png
                    image_bytes = svg2png(bytestring=image_data.encode('utf-8'))
                except ImportError:
                    print("Warning: cairosvg package needed for SVG conversion, skipping image.")
                    continue
                except Exception as e:
                    print(f"Warning: Failed to convert SVG: {e}, skipping image.")
                    continue
            else:
                try:
                    image_bytes = base64.b64decode(image_data)
                except Exception:
                    print("Warning: Invalid base64 image string, skipping image.")
                    continue

        if image_bytes:
            message_content.append(
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64.b64encode(image_bytes).decode('utf-8')}}
            )

    if len(message_content) <= 1:  # Only text prompt was added
        raise ValueError("No valid images provided for multi-image call.")

    system_content = CAREFRAME_SYSTEM_INSTRUCTIONS
    response = await asyncio.to_thread(
        client.messages.create,
        max_tokens=max_tokens,
        model=model,
        system=system_content,
        messages=[{"role": "user", "content": message_content}]
    )
    return response

# --- Generic LLM Call Functions (Updated to use config and handle returns) ---

async def call_llm_async(prompt: Union[str, Dict], model: Optional[str] = None, max_retries: int = 3, **kwargs) -> str:
    """
    Generic asynchronous LLM call function for text responses.
    Uses configured default text model if 'model' is not provided.
    """
    # Use provided model or default text model from config
    target_model = model or llm_config.default_text_model

    # Convert NumPy types
    prompt = convert_numpy_types(prompt)

    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1: await asyncio.sleep(0.1 * (2 ** (attempt - 1))) # Exponential backoff

            result_text = ""
            
            # Check if we should use a local LLM
            if LOCAL_LLM_CONFIG["enabled"] and is_local_model(target_model):
                local_model = target_model
                if target_model.startswith("local-"):
                    # Map model name to actual local model name (strip prefix)
                    local_model = target_model[6:]
                elif target_model in LOCAL_LLM_CONFIG["models"].values():
                    # Already a bare model name
                    pass
                else:
                    # Use default text model
                    local_model = LOCAL_LLM_CONFIG["models"]["text"]
                
                # Call local LLM
                response = await call_local_llm_async(prompt, model=local_model, **kwargs)
                if hasattr(response, 'content') and isinstance(response.content, list) and len(response.content) > 0 and hasattr(response.content[0], 'text'):
                    result_text = response.content[0].text
                else:
                    result_text = str(response)
            elif target_model.startswith("claude"):
                # Use the decorated function, it returns the full response object
                response = await call_claude_async(prompt, model=target_model, **kwargs)
                # Extract text from the response object
                if hasattr(response, 'content') and isinstance(response.content, list) and len(response.content) > 0 and hasattr(response.content[0], 'text'):
                    result_text = response.content[0].text
                else: # Fallback if structure unexpected
                    result_text = str(response)

            elif target_model.startswith("gemini"):
                 # Use the decorated function
                 response = await call_gemini_async(prompt, model=target_model, **kwargs)
                 # Extract text
                 if hasattr(response, 'text'):
                     result_text = response.text
                 else: # Fallback
                     result_text = str(response)
            else:
                raise ValueError(f"Unsupported model: {target_model}")

            return result_text # Return extracted text

        except Exception as e:
            print(f"LLM call attempt {attempt} failed for model {target_model}: {e}")
            if attempt == max_retries:
                raise # Re-raise the final exception

async def call_llm_async_json(prompt: Union[str, Dict], model: Optional[str] = None, max_retries: int = 3, **kwargs) -> Dict:
    """
    Generic asynchronous LLM call function for JSON responses.
    Uses configured default JSON model if 'model' is not provided.
    Returns a parsed dictionary.
    """
    target_model = model or llm_config.default_json_model
    prompt = convert_numpy_types(prompt)

    for attempt in range(1, max_retries + 1):
        raw_response_text = ""
        response = None
        parsed_json_response = None # Variable to hold the final dictionary

        try:
            if attempt > 1: await asyncio.sleep(0.1 * (2 ** (attempt - 1)))

            # --- Select and Call LLM ---
            if LOCAL_LLM_CONFIG["enabled"] and is_local_model(target_model):
                # ... (local LLM logic - should already return raw text via raw_response_text) ...
                local_model = target_model # simplified
                if target_model.startswith("local-"): local_model = target_model[6:]
                elif not target_model in LOCAL_LLM_CONFIG["models"].values():
                    local_model = LOCAL_LLM_CONFIG["models"]["json"]

                if isinstance(prompt, str) and "JSON format" not in prompt and "JSON output" not in prompt:
                     prompt += "\\n\\nIMPORTANT: Format your entire response as a valid JSON object, ensuring it can be parsed with json.loads(). Only output the JSON object itself, with no surrounding text or markdown."

                local_response_obj = await call_local_llm_async(prompt, model=local_model, **kwargs)
                if hasattr(local_response_obj, 'content') and isinstance(local_response_obj.content, list) and len(local_response_obj.content) > 0 and hasattr(local_response_obj.content[0], 'text'):
                    raw_response_text = local_response_obj.content[0].text
                else:
                    raw_response_text = str(local_response_obj) # Fallback

            elif target_model.startswith("claude"):
                # ... (Claude logic - should already return raw text via raw_response_text) ...
                response = await call_claude_async_json(prompt, model=target_model, **kwargs)
                if hasattr(response, 'content') and isinstance(response.content, list) and len(response.content) > 0 and hasattr(response.content[0], 'text'):
                    raw_response_text = response.content[0].text
                else: raise ValueError("Unexpected Claude JSON response format")

            elif target_model.startswith("gemini"):
                 response = await call_gemini_async_json(prompt, model=target_model, **kwargs)
                 # --- Check if Gemini returned a dict directly ---
                 if isinstance(response, dict):
                     print("DEBUG: Gemini returned a dict directly.") # Debug
                     parsed_json_response = response # Use the dict directly
                 else:
                     # --- If not a dict, proceed with text/parts extraction ---
                     if hasattr(response, 'text') and response.text:
                         raw_response_text = response.text
                         print(f"DEBUG: Gemini JSON received via .text: {raw_response_text[:100]}...") # Debug
                     elif hasattr(response, 'parts') and response.parts:
                         json_parts = [part.text for part in response.parts if hasattr(part, 'text')]
                         if json_parts:
                             raw_response_text = "".join(json_parts)
                             print(f"DEBUG: Gemini JSON reconstructed from parts: {raw_response_text[:100]}...") # Debug
                         else:
                             print(f"DEBUG: Gemini response had parts but no text content.")
                             raise ValueError("Unexpected Gemini JSON response structure (parts without text)")
                     else:
                         print(f"DEBUG: Could not find text or parts in Gemini JSON response. Response object: {response}") # Debug
                         raise ValueError("Unexpected Gemini JSON response format (no text or parts attribute)")
            else:
                raise ValueError(f"Unsupported model for JSON: {target_model}")

            # --- Parsing Step (only if we got raw text) ---
            if parsed_json_response is None: # Only parse if we didn't get a dict directly
                 # Clean text before parsing (common issue with Gemini/Local)
                 if target_model.startswith("gemini") or (LOCAL_LLM_CONFIG["enabled"] and is_local_model(target_model)):
                    raw_response_text = raw_response_text.strip()
                    if raw_response_text.startswith("```json") and raw_response_text.endswith("```"):
                        raw_response_text = raw_response_text[7:-3].strip()
                    elif raw_response_text.startswith("```") and raw_response_text.endswith("```"):
                        raw_response_text = raw_response_text[3:-3].strip()

                 print(f"DEBUG: Attempting to parse JSON: '{raw_response_text[:200]}...'") # Debug
                 parsed_json_response = json.loads(raw_response_text) # Parse the extracted text

            # --- Return the parsed dictionary ---
            return parsed_json_response

        except json.JSONDecodeError as e:
            print(f"JSON Decode Error attempt {attempt} for model {target_model}: {e}. Raw text: '{raw_response_text[:200]}...'") # Log the raw text
            if response and not isinstance(response, dict): # Log response object if it wasn't the dict
                 print(f"DEBUG: Full response object on JSON error: {response}")
            if attempt == max_retries:
                raise ValueError(f"Invalid JSON response after {max_retries} attempts (model: {target_model}): {str(e)}. Last raw text: '{raw_response_text[:500]}'") from e
        except Exception as e:
            print(f"LLM JSON call attempt {attempt} failed for model {target_model}: {e}")
            traceback.print_exc()
            if attempt == max_retries:
                raise

# --- Synchronous Generic Calls (Update similarly) ---

def call_llm_sync(prompt: Union[str, Dict], model: Optional[str] = None, max_retries: int = 3, **kwargs) -> str:
    """
    Generic synchronous LLM call function for text responses.
    Uses configured default text model if 'model' is not provided.
    """
    target_model = model or llm_config.default_text_model
    prompt = convert_numpy_types(prompt)

    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1: time.sleep(0.1 * (2 ** (attempt - 1)))

            result_text = ""
            
            # Check if we should use a local LLM
            if LOCAL_LLM_CONFIG["enabled"] and is_local_model(target_model):
                local_model = target_model
                if target_model.startswith("local-"):
                    # Map model name to actual local model name (strip prefix)
                    local_model = target_model[6:]
                elif target_model in LOCAL_LLM_CONFIG["models"].values():
                    # Already a bare model name
                    pass
                else:
                    # Use default text model
                    local_model = LOCAL_LLM_CONFIG["models"]["text"]
                
                # Call local LLM
                response = call_local_llm_sync(prompt, model=local_model, **kwargs)
                if hasattr(response, 'content') and isinstance(response.content, list) and len(response.content) > 0 and hasattr(response.content[0], 'text'):
                    result_text = response.content[0].text
                else:
                    result_text = str(response)
            elif target_model.startswith("claude"):
                response = call_claude_sync(prompt, model=target_model, **kwargs)
                if hasattr(response, 'content') and isinstance(response.content, list) and len(response.content) > 0 and hasattr(response.content[0], 'text'):
                    result_text = response.content[0].text
                else: result_text = str(response)
            elif target_model.startswith("gemini"):
                 response = call_gemini_sync(prompt, model=target_model, **kwargs)
                 if hasattr(response, 'text'): result_text = response.text
                 else: result_text = str(response)
            else:
                raise ValueError(f"Unsupported model: {target_model}")
            return result_text
        except Exception as e:
             print(f"LLM sync call attempt {attempt} failed for model {target_model}: {e}")
             if attempt == max_retries: raise

def call_llm_sync_json(prompt: Union[str, Dict], model: Optional[str] = None, max_retries: int = 3, **kwargs) -> Dict:
    """
    Generic synchronous LLM call function for JSON responses.
    Uses configured default JSON model if 'model' is not provided.
    Returns a parsed dictionary.
    """
    target_model = model or llm_config.default_json_model
    prompt = convert_numpy_types(prompt)

    for attempt in range(1, max_retries + 1):
        raw_response_text = ""
        try:
            if attempt > 1: time.sleep(0.1 * (2 ** (attempt - 1)))

            # Check if we should use a local LLM
            if LOCAL_LLM_CONFIG["enabled"] and is_local_model(target_model):
                local_model = target_model
                if target_model.startswith("local-"):
                    # Map model name to actual local model name (strip prefix)
                    local_model = target_model[6:]
                elif target_model in LOCAL_LLM_CONFIG["models"].values():
                    # Already a bare model name
                    pass
                else:
                    # Use default JSON model
                    local_model = LOCAL_LLM_CONFIG["models"]["json"]
                
                # Ensure JSON instructions are included
                if isinstance(prompt, str) and "JSON format" not in prompt and "JSON output" not in prompt:
                    prompt += "\n\nIMPORTANT: Format your entire response as a valid JSON object, ensuring it can be parsed with json.loads(). Only output the JSON object itself, with no surrounding text or markdown."
                
                # Call local LLM
                response = call_local_llm_sync(prompt, model=local_model, **kwargs)
                if hasattr(response, 'content') and isinstance(response.content, list) and len(response.content) > 0 and hasattr(response.content[0], 'text'):
                    raw_response_text = response.content[0].text
                else:
                    raw_response_text = str(response)
            elif target_model.startswith("claude"):
                response = call_claude_sync_json(prompt, model=target_model, **kwargs)
                if hasattr(response, 'content') and isinstance(response.content, list) and len(response.content) > 0 and hasattr(response.content[0], 'text'):
                    raw_response_text = response.content[0].text
                else: raise ValueError("Unexpected Claude JSON response format")
            elif target_model.startswith("gemini"):
                 response = call_gemini_sync_json(prompt, model=target_model, **kwargs)
                 if hasattr(response, 'text'):
                     raw_response_text = response.text
                     # Clean Gemini JSON output
                     raw_response_text = raw_response_text.strip()
                     if raw_response_text.startswith("```json") and raw_response_text.endswith("```"): raw_response_text = raw_response_text[7:-3].strip()
                     elif raw_response_text.startswith("```") and raw_response_text.endswith("```"): raw_response_text = raw_response_text[3:-3].strip()
                 else: raise ValueError("Unexpected Gemini JSON response format")
            else:
                raise ValueError(f"Unsupported model for JSON: {target_model}")

            # Clean potential markdown JSON formatting from local LLM response
            if LOCAL_LLM_CONFIG["enabled"] and is_local_model(target_model):
                raw_response_text = raw_response_text.strip()
                if raw_response_text.startswith("```json") and raw_response_text.endswith("```"):
                    raw_response_text = raw_response_text[7:-3].strip()
                elif raw_response_text.startswith("```") and raw_response_text.endswith("```"):
                    raw_response_text = raw_response_text[3:-3].strip()

            return json.loads(raw_response_text)

        except json.JSONDecodeError as e:
            print(f"JSON Decode Error sync attempt {attempt} for model {target_model}: {e}. Response: '{raw_response_text[:200]}...'")
            if attempt == max_retries:
                raise ValueError(f"Invalid JSON response sync after {max_retries} attempts (model: {target_model}): {str(e)}. Last response: '{raw_response_text[:500]}'") from e
        except Exception as e:
            print(f"LLM JSON sync call attempt {attempt} failed for model {target_model}: {e}")
            if attempt == max_retries: raise

# --- Local LLM Call Functions ---
@track_llm_call
async def call_local_llm_async(prompt: Union[str, Dict], model: str, max_tokens: int = 5000, temperature: float = 0.0):
    """
    Call a local LLM API asynchronously.
    This uses the OpenAI-compatible API that many local LLM servers implement.
    """
    import aiohttp
    
    endpoint = get_llm_endpoint("text")
    if not endpoint:
        raise ValueError("Local LLM endpoint not configured")
    
    # Format the prompt based on type
    messages = []
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list):
        # Assume it's already in the messages format
        messages = prompt
    elif isinstance(prompt, dict):
        # If it's a dict with a system key, use it for system message
        if "system" in prompt:
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt.get("prompt", "")}
            ]
        else:
            # Otherwise, just convert to a user message
            messages = [{"role": "user", "content": json.dumps(prompt)}]
    
    # Prepare the API request payload
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{endpoint}/chat/completions", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Local LLM API error: {response.status} {error_text}")
                
                result = await response.json()
                
                # Create a response object with a similar structure to cloud APIs
                # to be compatible with the existing code
                response_obj = type('LocalLLMResponse', (), {
                    'content': [type('Content', (), {'text': result['choices'][0]['message']['content']})()],
                    'usage': type('Usage', (), {
                        'input_tokens': result.get('usage', {}).get('prompt_tokens', 0),
                        'output_tokens': result.get('usage', {}).get('completion_tokens', 0)
                    })()
                })
                
                return response_obj
                
    except aiohttp.ClientError as e:
        raise ValueError(f"Connection error to local LLM: {str(e)}")
    except KeyError as e:
        raise ValueError(f"Unexpected response format from local LLM: {str(e)}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON response from local LLM")

@track_llm_call
def call_local_llm_sync(prompt: Union[str, Dict], model: str, max_tokens: int = 5000, temperature: float = 0.0):
    """
    Call a local LLM API synchronously.
    This uses the OpenAI-compatible API that many local LLM servers implement.
    """
    import requests
    
    endpoint = get_llm_endpoint("text")
    if not endpoint:
        raise ValueError("Local LLM endpoint not configured")
    
    # Format the prompt based on type
    messages = []
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list):
        # Assume it's already in the messages format
        messages = prompt
    elif isinstance(prompt, dict):
        # If it's a dict with a system key, use it for system message
        if "system" in prompt:
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt.get("prompt", "")}
            ]
        else:
            # Otherwise, just convert to a user message
            messages = [{"role": "user", "content": json.dumps(prompt)}]
    
    # Prepare the API request payload
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        response = requests.post(f"{endpoint}/chat/completions", json=payload)
        if response.status_code != 200:
            raise ValueError(f"Local LLM API error: {response.status_code} {response.text}")
            
        result = response.json()
        
        # Create a response object with a similar structure to cloud APIs
        # to be compatible with the existing code
        response_obj = type('LocalLLMResponse', (), {
            'content': [type('Content', (), {'text': result['choices'][0]['message']['content']})()],
            'usage': type('Usage', (), {
                'input_tokens': result.get('usage', {}).get('prompt_tokens', 0),
                'output_tokens': result.get('usage', {}).get('completion_tokens', 0)
            })()
        })
        
        return response_obj
        
    except requests.RequestException as e:
        raise ValueError(f"Connection error to local LLM: {str(e)}")
    except KeyError as e:
        raise ValueError(f"Unexpected response format from local LLM: {str(e)}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON response from local LLM")

@track_llm_call
async def call_local_llm_with_image_async(prompt: str, image_data, model: str, max_tokens: int = 5000, temperature: float = 0.0):
    """
    Call a local LLM API with image data asynchronously.
    This uses the vision capabilities of local LLMs with OpenAI-compatible API.
    """
    import aiohttp
    
    endpoint = get_llm_endpoint("vision")
    if not endpoint:
        raise ValueError("Local LLM vision endpoint not configured")
    
    # Process image data into base64
    image_base64 = None
    if isinstance(image_data, Figure):
        buf = io.BytesIO()
        image_data.savefig(buf, format='png')
        buf.seek(0)
        image_bytes = buf.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    elif isinstance(image_data, Image.Image):
        buf = io.BytesIO()
        image_data.save(buf, format='PNG')
        buf.seek(0)
        image_bytes = buf.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    elif isinstance(image_data, bytes):
        image_base64 = base64.b64encode(image_data).decode('utf-8')
    elif isinstance(image_data, str):
        # If it's already base64
        if ',' in image_data and ';base64,' in image_data:
            # Handle data URL format
            image_base64 = image_data.split(';base64,')[1]
        else:
            try:
                # Attempt to decode if it's base64 without the data URL prefix
                base64.b64decode(image_data)
                image_base64 = image_data
            except:
                raise ValueError("Invalid image data string format")
    else:
        raise ValueError(f"Unsupported image data type: {type(image_data)}")
    
    if not image_base64:
        raise ValueError("Failed to process image data")
    
    # Format the message with content list that includes text and image
    content = [
        {
            "type": "text",
            "text": prompt
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}"
            }
        }
    ]
    
    # Prepare the API request payload
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{endpoint}/chat/completions", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Local LLM vision API error: {response.status} {error_text}")
                
                result = await response.json()
                
                # Create a response object with a similar structure to cloud APIs
                # to be compatible with the existing code
                response_obj = type('LocalLLMVisionResponse', (), {
                    'content': [type('Content', (), {'text': result['choices'][0]['message']['content']})()],
                    'usage': type('Usage', (), {
                        'input_tokens': result.get('usage', {}).get('prompt_tokens', 0),
                        'output_tokens': result.get('usage', {}).get('completion_tokens', 0)
                    })()
                })
                
                return response_obj
                
    except aiohttp.ClientError as e:
        raise ValueError(f"Connection error to local vision LLM: {str(e)}")
    except KeyError as e:
        raise ValueError(f"Unexpected response format from local vision LLM: {str(e)}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON response from local vision LLM")

