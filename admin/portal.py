import requests
import base64
from cryptography.fernet import Fernet
import os
import json
from pathlib import Path

def is_couchdb_available():
    """Check if CouchDB is available"""
    try:
        response = requests.get(
            "http://localhost:5984/_up",
            auth=("admin", "cfpwd"),
            timeout=2
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def get_secrets_from_database():
    try:
        # Check if CouchDB is available
        if is_couchdb_available():
            return get_secrets_from_couchdb()
        else:
            print("CouchDB not available, using mock storage for API keys")
            return get_secrets_from_mock_db()
            
    except Exception as e:
        print(f"Error getting secrets: {str(e)}")
        # Try mock as last resort
        try:
            return get_secrets_from_mock_db()
        except Exception as mock_e:
            print(f"Error getting secrets from mock storage: {str(mock_e)}")
            return {}

def get_secrets_from_couchdb():
    """Get secrets from CouchDB database"""
    try:
        # Get encryption key from database
        encryption_key_response = requests.get(
            "http://localhost:5984/settings/encryption_key",
            auth=("admin", "cfpwd")
        )
        
        if encryption_key_response.status_code != 200:
            print("Error: Could not retrieve encryption key from database")
            return {}
            
        encryption_key = base64.b64decode(encryption_key_response.json()["key"])
        cipher_suite = Fernet(encryption_key)

        # Get encrypted keys from database
        api_keys_response = requests.get(
            "http://localhost:5984/settings/api_keys",
            auth=("admin", "cfpwd")
        )
        
        if api_keys_response.status_code != 200:
            print("Error: Could not retrieve API keys from database")
            return {}

        encrypted_keys = api_keys_response.json()["keys"]
        
        # Decrypt keys
        decrypted_keys = {}
        for key_id, encrypted_value in encrypted_keys.items():
            if encrypted_value:  # Only decrypt non-empty values
                decrypted_value = cipher_suite.decrypt(
                    base64.b64decode(encrypted_value)
                ).decode()
                decrypted_keys[key_id] = decrypted_value

        return decrypted_keys
    except requests.RequestException as e:
        print(f"Request error: {str(e)}")
        return {}

def get_secrets_from_mock_db():
    """Get secrets from mock JSON database"""
    try:
        # Load encryption key
        encryption_key_file = '.encryption_key'
        if not os.path.exists(encryption_key_file):
            print("Error: Encryption key file not found")
            return {}
            
        with open(encryption_key_file, 'rb') as f:
            encryption_key_data = f.read()
            
        # Validate the key format
        try:
            cipher_suite = Fernet(encryption_key_data)
        except Exception as e:
            print(f"Invalid encryption key format: {e}")
            return {}
        
        # Get API keys from mock storage
        mock_dir = os.path.join(str(Path.home()), '.careframe', 'mock_couchdb', 'settings', 'docs')
        api_keys_path = os.path.join(mock_dir, 'api_keys.json')
        
        if not os.path.exists(api_keys_path):
            print("Error: API keys file not found in mock storage")
            return {}
            
        with open(api_keys_path, 'r') as f:
            doc = json.load(f)
            
        encrypted_keys = doc.get("keys", {})
        
        # Decrypt keys
        decrypted_keys = {}
        for key_id, encrypted_value in encrypted_keys.items():
            if encrypted_value:  # Only decrypt non-empty values
                try:
                    decrypted_value = cipher_suite.decrypt(
                        base64.b64decode(encrypted_value)
                    ).decode()
                    decrypted_keys[key_id] = decrypted_value
                except Exception as e:
                    print(f"Error decrypting {key_id}: {e}")
                    
        return decrypted_keys
    except Exception as e:
        print(f"Error getting API keys from mock storage: {e}")
        return {}

secrets = get_secrets_from_database()