#!/usr/bin/env python3
"""
Launcher script for CareFrame that initializes the database before starting the application.
This ensures that either a real CouchDB or a JSON mock database is available.
"""

import sys
import logging
import importlib.util
import os
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("app_launcher")

def import_module_from_file(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    """Initialize the database and start the application."""
    # Import the database initialization functions
    if os.path.exists("init_database.py"):
        init_db = import_module_from_file("init_database", "init_database.py")
    else:
        logger.error("init_database.py not found!")
        sys.exit(1)
    
    # Initialize the database - force mock DB if CouchDB is not available
    logger.info("Initializing database...")
    try:
        # Try with auto-detection first
        db_result = init_db.initialize_database()
        
        if not db_result["success"]:
            logger.warning(f"Database initialization with auto-detection failed: {db_result.get('error', 'Unknown error')}")
            logger.info("Retrying with forced JSON mock database...")
            # Force using mock database
            db_result = init_db.initialize_database(use_mock=True)
            
            if not db_result["success"]:
                logger.error(f"Failed to initialize mock database: {db_result.get('error', 'Unknown error')}")
                # Continue anyway as a last resort
    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        # Try one more time with mock DB
        try:
            db_result = init_db.initialize_database(use_mock=True)
        except Exception as e2:
            logger.error(f"Final attempt to initialize database failed: {e2}")
            db_result = {"success": False, "error": str(e2), "database_type": "Failed"}
    
    db_type = db_result.get("database_type", "Unknown")
    logger.info(f"Using database type: {db_type}")
    
    # Initialize API keys
    try:
        if os.path.exists("init_api_keys.py"):
            init_api_keys = import_module_from_file("init_api_keys", "init_api_keys.py")
            logger.info("Initializing API keys...")
            init_api_keys.init_api_keys()
        else:
            logger.warning("init_api_keys.py not found, skipping API key initialization")
    except Exception as e:
        logger.error(f"Error initializing API keys: {e}")
        logger.info("Application will continue without API keys")
    
    # Extract any command line arguments to pass to the app
    app_args = sys.argv[1:]
    
    # Start the application
    logger.info("Starting CareFrame application...")
    try:
        # Check if app.py exists
        if not os.path.exists("app.py"):
            logger.error("app.py not found!")
            sys.exit(1)
        
        # Launch the app using the current Python interpreter
        cmd = [sys.executable, "app.py"] + app_args
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Execute the app and wait for it to complete
        process = subprocess.run(cmd)
        sys.exit(process.returncode)
    
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 