#!/usr/bin/env python3
"""
Initialize API keys from environment variables into the CouchDB database.
This script can be run during container startup to ensure API keys are available.
"""

import os
import requests
import logging
import base64
from cryptography.fernet import Fernet
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("init_api_keys")

def is_couchdb_available(base_url="http://localhost:5984", auth=("admin", "cfpwd")):
    """Check if CouchDB is available"""
    try:
        response = requests.get(
            f"{base_url}/_up",
            auth=auth,
            timeout=2
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        logger.warning("Could not connect to CouchDB")
        return False

def init_api_keys(use_mock=None):
    """Initialize API keys from environment variables into the database"""
    try:
        base_url = "http://localhost:5984"
        auth = ("admin", "cfpwd")
        
        # Determine whether to use real CouchDB or mock
        if use_mock is None:
            use_mock = not is_couchdb_available(base_url, auth)
        
        if use_mock:
            logger.info("Using mock storage for API keys")
            return init_api_keys_mock()
        
        logger.info("Using CouchDB for API keys")
        
        # Make sure settings database exists
        response = requests.put(
            f"{base_url}/settings",
            auth=auth
        )
        logger.info(f"Settings database check: {response.status_code}")
        
        # Check/create encryption key
        try:
            response = requests.get(
                f"{base_url}/settings/encryption_key",
                auth=auth
            )
            
            if response.status_code == 200:
                key_data = response.json()
                encryption_key = base64.b64decode(key_data['key'])
                logger.info("Using existing encryption key")
            else:
                # Generate new key if not exists
                encryption_key = Fernet.generate_key()
                requests.put(
                    f"{base_url}/settings/encryption_key",
                    auth=auth,
                    json={
                        "_id": "encryption_key",
                        "type": "encryption_key",
                        "key": base64.b64encode(encryption_key).decode()
                    }
                )
                logger.info("Created new encryption key")
            
            cipher_suite = Fernet(encryption_key)
            
        except Exception as e:
            logger.error(f"Error setting up encryption: {e}")
            # Fallback to mock if CouchDB encryption fails
            logger.info("Falling back to mock storage for API keys")
            return init_api_keys_mock()
        
        # Get existing API keys if available
        try:
            response = requests.get(
                f"{base_url}/settings/api_keys",
                auth=auth
            )
            
            if response.status_code == 200:
                doc = response.json()
                logger.info("Found existing API keys document")
            else:
                # Create new empty API keys document
                doc = {
                    "_id": "api_keys",
                    "type": "api_keys",
                    "keys": {}
                }
                logger.info("Created new API keys document")
        except Exception as e:
            logger.error(f"Error getting API keys: {e}")
            # Fallback to mock if getting API keys fails
            logger.info("Falling back to mock storage for API keys")
            return init_api_keys_mock()
        
        # Get API keys from environment variables
        env_keys = {
            "claude_api_key": os.environ.get("CLAUDE_API_KEY", ""),
            "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
            "umls_api_key": os.environ.get("UMLS_API_KEY", ""),
            "unpaywall_email": os.environ.get("UNPAYWALL_EMAIL", ""),
            "zenodo_api_key": os.environ.get("ZENODO_API_KEY", ""),
            "core_api_key": os.environ.get("CORE_API_KEY", ""),
            "entrez_api_key": os.environ.get("ENTREZ_API_KEY", ""),
            "entrez_email": os.environ.get("ENTREZ_EMAIL", "")
        }
        
        # Initialize keys dictionary if it doesn't exist
        if "keys" not in doc:
            doc["keys"] = {}
        
        # Encrypt and update keys that have values
        keys_updated = False
        cloud_llm_keys_configured = False
        
        for key_name, value in env_keys.items():
            if value:
                # Only update if there's a non-empty value
                encrypted_value = base64.b64encode(
                    cipher_suite.encrypt(value.encode())
                ).decode()
                doc["keys"][key_name] = encrypted_value
                keys_updated = True
                logger.info(f"Updated {key_name} from environment variable")
                
                # Check if this is a cloud LLM key
                if key_name in ["claude_api_key", "gemini_api_key"]:
                    cloud_llm_keys_configured = True
        
        # Save if any keys were updated
        if keys_updated:
            try:
                response = requests.put(
                    f"{base_url}/settings/api_keys",
                    auth=auth,
                    json=doc
                )
                
                if response.status_code in (201, 200):
                    logger.info("Successfully saved API keys to database")
                    
                    # Display security warning if cloud LLM keys are configured
                    if cloud_llm_keys_configured:
                        logger.warning("=" * 80)
                        logger.warning("⚠️  SECURITY WARNING: Cloud LLM API keys detected")
                        logger.warning("Data sent to these services will leave your environment!")
                        logger.warning("DO NOT use with sensitive healthcare data or PHI")
                        logger.warning("For production use, configure local LLMs only")
                        logger.warning("=" * 80)
                    
                    return True
                else:
                    logger.error(f"Failed to save API keys: {response.status_code}")
                    # Fallback to mock if saving fails
                    logger.info("Falling back to mock storage for API keys")
                    return init_api_keys_mock(keys=env_keys)
            except Exception as e:
                logger.error(f"Error saving API keys: {e}")
                # Fallback to mock if saving fails
                logger.info("Falling back to mock storage for API keys")
                return init_api_keys_mock(keys=env_keys)
        else:
            logger.info("No API keys found in environment variables")
            return True
            
    except Exception as e:
        logger.error(f"Unexpected error initializing API keys: {e}")
        # Fallback to mock on any error
        logger.info("Falling back to mock storage for API keys")
        return init_api_keys_mock()

def init_api_keys_mock(keys=None):
    """Initialize API keys using local file storage instead of CouchDB"""
    try:
        # Create directory for mock storage
        mock_dir = os.path.join(str(Path.home()), '.careframe', 'mock_couchdb', 'settings', 'docs')
        os.makedirs(mock_dir, exist_ok=True)
        
        # Load or create encryption key
        encryption_key_file = '.encryption_key'
        if os.path.exists(encryption_key_file):
            with open(encryption_key_file, 'rb') as f:
                encryption_key_data = f.read()
                # Validate the key format
                try:
                    # Test if it's a valid Fernet key
                    Fernet(encryption_key_data)
                    encryption_key = encryption_key_data
                    logger.info("Using existing encryption key from file")
                except Exception as e:
                    logger.warning(f"Invalid encryption key format: {e}")
                    # Generate new key
                    encryption_key = Fernet.generate_key()
                    with open(encryption_key_file, 'wb') as f2:
                        f2.write(encryption_key)
                    logger.info("Generated and saved new encryption key")
        else:
            # Generate a new key
            encryption_key = Fernet.generate_key()
            with open(encryption_key_file, 'wb') as f:
                f.write(encryption_key)
            logger.info("Created new encryption key file")
        
        cipher_suite = Fernet(encryption_key)
        
        # Save encryption key in mock database
        encryption_key_path = os.path.join(mock_dir, 'encryption_key.json')
        encryption_key_doc = {
            "_id": "encryption_key",
            "type": "encryption_key",
            "key": base64.b64encode(encryption_key).decode()
        }
        with open(encryption_key_path, 'w') as f:
            json.dump(encryption_key_doc, f, indent=2)
        
        # Load existing API keys document or create new one
        api_keys_path = os.path.join(mock_dir, 'api_keys.json')
        if os.path.exists(api_keys_path):
            with open(api_keys_path, 'r') as f:
                doc = json.load(f)
            logger.info("Loaded existing API keys from mock storage")
        else:
            doc = {
                "_id": "api_keys",
                "type": "api_keys",
                "keys": {}
            }
            logger.info("Created new API keys document in mock storage")
        
        # Get API keys from provided dict or environment variables
        if keys is None:
            keys = {
                "claude_api_key": os.environ.get("CLAUDE_API_KEY", ""),
                "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
                "umls_api_key": os.environ.get("UMLS_API_KEY", ""),
                "unpaywall_email": os.environ.get("UNPAYWALL_EMAIL", ""),
                "zenodo_api_key": os.environ.get("ZENODO_API_KEY", ""),
                "core_api_key": os.environ.get("CORE_API_KEY", ""),
                "entrez_api_key": os.environ.get("ENTREZ_API_KEY", ""),
                "entrez_email": os.environ.get("ENTREZ_EMAIL", "")
            }
        
        # Initialize keys dictionary if it doesn't exist
        if "keys" not in doc:
            doc["keys"] = {}
        
        # Encrypt and update keys that have values
        keys_updated = False
        cloud_llm_keys_configured = False
        
        for key_name, value in keys.items():
            if value:
                # Only update if there's a non-empty value
                encrypted_value = base64.b64encode(
                    cipher_suite.encrypt(value.encode())
                ).decode()
                doc["keys"][key_name] = encrypted_value
                keys_updated = True
                logger.info(f"Updated {key_name} in mock storage")
                
                # Check if this is a cloud LLM key
                if key_name in ["claude_api_key", "gemini_api_key"]:
                    cloud_llm_keys_configured = True
        
        # Save API keys document
        if keys_updated:
            with open(api_keys_path, 'w') as f:
                json.dump(doc, f, indent=2)
            logger.info("Successfully saved API keys to mock storage")
            
            # Display security warning if cloud LLM keys are configured
            if cloud_llm_keys_configured:
                logger.warning("=" * 80)
                logger.warning("⚠️  SECURITY WARNING: Cloud LLM API keys detected")
                logger.warning("Data sent to these services will leave your environment!")
                logger.warning("DO NOT use with sensitive healthcare data or PHI")
                logger.warning("For production use, configure local LLMs only")
                logger.warning("=" * 80)
        else:
            logger.info("No API keys found to update in mock storage")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing API keys in mock storage: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting API keys initialization")
    success = init_api_keys()
    if success:
        logger.info("API keys initialization completed successfully")
    else:
        logger.error("API keys initialization failed") 