#!/usr/bin/env python3
"""
Database initialization script for CareFrame.
This script can be run directly to initialize the database,
or imported to use the initialization functions.
"""

import logging
import os
import sys
import json
from pathlib import Path
import argparse
import requests
import couchdb

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("init_database")

# Import the database setup
from db_ops.generate_tables import DatabaseSetup

def initialize_database(use_mock=None, verbose=False):
    """
    Initialize the database (CouchDB or JSON mock).
    
    Args:
        use_mock (bool, optional): Force using mock DB if True, force using real DB if False.
                                  If None, auto-detect based on CouchDB availability.
        verbose (bool, optional): Print detailed output if True.
    
    Returns:
        dict: Information about the initialized database
    """
    try:
        # Set log level based on verbosity
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Create database setup with the specified mode
        db_setup = DatabaseSetup(use_mock=use_mock)
        
        # Initialize databases and views
        result = db_setup.setup_databases()
        
        # Get info about the database
        db_info = db_setup.get_database_info()
        
        if "error" in db_info:
            logger.error(f"Error getting database info: {db_info['error']}")
        else:
            db_type = db_info.get("server_type", "Unknown")
            db_count = len(db_info.get("databases", []))
            logger.info(f"Using database type: {db_type}")
            logger.info(f"Available databases: {db_count}")
            
            if verbose:
                logger.debug(f"Database list: {', '.join(db_info.get('databases', []))}")
        
        # Return information about the initialization
        return {
            "success": True,
            "database_type": "JSON Mock" if db_setup.use_mock else "CouchDB",
            "databases_created": list(result.keys()),
            "database_info": db_info
        }
    
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_database_status():
    """
    Check the status of the database without initializing.
    
    Returns:
        dict: Status information about the database
    """
    try:
        # Create database setup but don't initialize
        db_setup = DatabaseSetup()
        
        # Get info about the database
        db_info = db_setup.get_database_info()
        
        # Return status information
        return {
            "success": "error" not in db_info,
            "database_type": "JSON Mock" if db_setup.use_mock else "CouchDB",
            "status": "Online" if "error" not in db_info else "Error",
            "error": db_info.get("error", None),
            "database_info": db_info
        }
    
    except Exception as e:
        logger.error(f"Error checking database status: {e}")
        return {
            "success": False,
            "status": "Error",
            "error": str(e)
        }

def create_json_db_config(config_path=None):
    """
    Create a configuration file for the JSON mock database.
    
    Args:
        config_path (str, optional): Path to save the configuration file.
                                    If None, use the default location.
    
    Returns:
        str: Path to the saved configuration file
    """
    if config_path is None:
        # Default to the user's home directory
        home_dir = str(Path.home())
        config_dir = os.path.join(home_dir, '.careframe')
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, 'db_config.json')
    
    config = {
        "use_mock": None,  # Auto-detect
        "mock_db_path": os.path.join(str(Path.home()), '.careframe', 'mock_couchdb'),
        "couchdb_url": "http://localhost:5984",
        "couchdb_auth": ["admin", "cfpwd"]
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created database configuration at {config_path}")
    return config_path

def read_db_config(config_path=None):
    """
    Read the database configuration file.
    
    Args:
        config_path (str, optional): Path to the configuration file.
                                    If None, use the default location.
    
    Returns:
        dict: The configuration settings
    """
    if config_path is None:
        # Default to the user's home directory
        home_dir = str(Path.home())
        config_path = os.path.join(home_dir, '.careframe', 'db_config.json')
    
    # Create default config if it doesn't exist
    if not os.path.exists(config_path):
        config_path = create_json_db_config(config_path)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Initialize database for CareFrame')
    parser.add_argument('--mock', action='store_true', help='Force using JSON mock database')
    parser.add_argument('--couchdb', action='store_true', help='Force using CouchDB')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print verbose output')
    parser.add_argument('--status', action='store_true', help='Check database status without initializing')
    parser.add_argument('--config', action='store_true', help='Create default configuration file')
    
    args = parser.parse_args()
    
    # Handle config creation
    if args.config:
        config_path = create_json_db_config()
        print(f"Created configuration file at {config_path}")
        sys.exit(0)
    
    # Determine database type to use
    use_mock = None
    if args.mock:
        use_mock = True
    elif args.couchdb:
        use_mock = False
    
    # Check status or initialize
    if args.status:
        status = get_database_status()
        if status["success"]:
            print(f"Database Status: {status['status']}")
            print(f"Database Type: {status['database_type']}")
            if args.verbose and "database_info" in status:
                print(f"Databases: {', '.join(status['database_info'].get('databases', []))}")
        else:
            print(f"Error: {status.get('error', 'Unknown error')}")
    else:
        # Initialize database
        result = initialize_database(use_mock=use_mock, verbose=args.verbose)
        if result["success"]:
            print(f"Database initialized successfully.")
            print(f"Using {result['database_type']}")
            print(f"Created databases: {', '.join(result['databases_created'])}")
        else:
            print(f"Error initializing database: {result.get('error', 'Unknown error')}")
            sys.exit(1) 