try:
    import couchdb
    from couchdb.http import PreconditionFailed, ResourceNotFound
    COUCHDB_AVAILABLE = True
except ImportError:
    COUCHDB_AVAILABLE = False
    # Define fallback exceptions
    class PreconditionFailed(Exception): pass
    class ResourceNotFound(Exception): pass

import json
from datetime import datetime
import uuid
import logging
import requests
import os
from pathlib import Path

# Import our mock implementation
from db_ops.json_db import MockCouchDBServer, MockCouchDBPreconditionFailed, MockCouchDBResourceNotFound

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("database_setup")

class DatabaseSetup:
    def __init__(self, base_url="http://localhost:5984", auth=("admin", "cfpwd"), use_mock=None):
        self.base_url = base_url
        self.auth = auth
        
        # Determine whether to use the real CouchDB or the mock
        if use_mock is None:
            # Auto-detect CouchDB availability, with minimal retries
            self.use_mock = not self._is_couchdb_available()
        else:
            # Explicitly set based on parameter
            self.use_mock = use_mock
        
        if self.use_mock:
            logger.info("Using JSON-based mock CouchDB")
            try:
                self.server = MockCouchDBServer(base_url, auth)
            except Exception as e:
                logger.error(f"Error creating mock CouchDB server: {e}")
                # Create a fresh instance even if something went wrong
                self.server = MockCouchDBServer(base_url, auth)
        else:
            logger.info("Using real CouchDB")
            self.server = couchdb.Server(base_url)
            self.server.resource.credentials = auth
    
    def _is_couchdb_available(self):
        """Check if CouchDB is available by trying to connect to it."""
        if not COUCHDB_AVAILABLE:
            logger.warning("CouchDB Python library not installed")
            return False
        
        try:
            # Try to connect to CouchDB server with short timeout
            response = requests.get(
                f"{self.base_url}/_up",
                auth=self.auth,
                timeout=1  # Shorter timeout to fail faster
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            logger.warning("Could not connect to CouchDB, will use mock database")
            return False

    def create_database(self, db_name):
        """Create a database if it doesn't exist."""
        try:
            return self.server.create(db_name)
        except (PreconditionFailed, MockCouchDBPreconditionFailed):
            return self.server[db_name]
        except Exception as e:
            logger.error(f"Error creating database {db_name}: {e}")
            # If using real CouchDB and it fails, switch to mock
            if not self.use_mock:
                logger.info(f"Falling back to mock database for {db_name}")
                self.use_mock = True
                self.server = MockCouchDBServer(self.base_url, self.auth)
                # Try again with mock
                try:
                    return self.server.create(db_name)
                except (PreconditionFailed, MockCouchDBPreconditionFailed):
                    return self.server[db_name]

    def setup_databases(self):
        """Set up all required databases and their views."""
        # Create main databases
        study_designs_db = self.create_database('study_designs')
        user_logs_db = self.create_database('user_logs')
        
        # Set up users database if it doesn't exist
        users_db = self.create_database('_users')
        studies_db = self.create_database('studies')
        teams_db = self.create_database('teams')
        logs_db = self.create_database('logs')
        
        # Create settings database for storing API keys and encryption
        settings_db = self.create_database('settings')
        
        # Initialize empty API keys document with placeholders if it doesn't exist
        try:
            settings_db.get('api_keys')
        except (ResourceNotFound, MockCouchDBResourceNotFound):
            # Create a document with empty placeholders for required API keys
            api_keys_doc = {
                "_id": "api_keys",
                "type": "api_keys",
                "keys": {
                    "gemini_api_key": "",
                    "umls_api_key": "",
                    "unpaywall_email": "",
                    "zenodo_api_key": "",
                    "core_api_key": "",
                    "entrez_api_key": "",
                    "entrez_email": "",
                    "claude_api_key": ""
                }
            }
            settings_db.save(api_keys_doc)
            logger.info("Created API keys document with empty placeholders")

        # Set up views for study_designs
        design_doc = {
            '_id': '_design/study_designs',
            'views': {
                'by_client_id': {
                    'map': '''function(doc) {
                        if (doc.client_id) {
                            emit(doc.client_id, doc);
                        }
                    }'''
                },
                'by_study_id': {
                    'map': '''function(doc) {
                        if (doc.study_id) {
                            emit(doc.study_id, doc);
                        }
                    }'''
                }
            }
        }
        self._update_design_doc(study_designs_db, design_doc)

        # Set up views for user_logs
        log_design_doc = {
            '_id': '_design/user_logs',
            'views': {
                'by_client_id': {
                    'map': '''function(doc) {
                        if (doc.client_id) {
                            emit(doc.client_id, doc);
                        }
                    }'''
                },
                'by_timestamp': {
                    'map': '''function(doc) {
                        if (doc.timestamp) {
                            emit(doc.timestamp, doc);
                        }
                    }'''
                },
                'by_action_type': {
                    'map': '''function(doc) {
                        if (doc.action_type) {
                            emit(doc.action_type, doc);
                        }
                    }'''
                }
            }
        }
        self._update_design_doc(user_logs_db, log_design_doc)
        
        # Set up views for studies
        studies_design_doc = {
            '_id': '_design/studies',
            'views': {
                'by_name': {
                    'map': '''function(doc) {
                        if (doc.name) {
                            emit(doc.name, doc);
                        }
                    }'''
                },
                'by_team': {
                    'map': '''function(doc) {
                        if (doc.team_id) {
                            emit(doc.team_id, doc);
                        }
                    }'''
                },
                'by_date': {
                    'map': '''function(doc) {
                        if (doc.created_at) {
                            emit(doc.created_at, doc);
                        }
                    }'''
                }
            }
        }
        self._update_design_doc(studies_db, studies_design_doc)
        
        # Set up views for teams
        teams_design_doc = {
            '_id': '_design/teams',
            'views': {
                'by_name': {
                    'map': '''function(doc) {
                        if (doc.name) {
                            emit(doc.name, doc);
                        }
                    }'''
                }
            }
        }
        self._update_design_doc(teams_db, teams_design_doc)
        
        return {
            'study_designs': study_designs_db,
            'user_logs': user_logs_db,
            'users': users_db,
            'studies': studies_db,
            'teams': teams_db,
            'logs': logs_db,
            'settings': settings_db
        }

    def _update_design_doc(self, db, design_doc):
        """Update or create a design document."""
        try:
            existing = db[design_doc['_id']]
            design_doc['_rev'] = existing.rev
            db.save(design_doc)
        except (ResourceNotFound, MockCouchDBResourceNotFound):
            db.save(design_doc)

    def log_user_action(self, client_id, action_type, details=None):
        """Log a user action to the user_logs database."""
        log_entry = {
            '_id': str(uuid.uuid4()),
            'client_id': client_id,
            'action_type': action_type,
            'timestamp': datetime.utcnow().isoformat(),
            'details': details or {}
        }
        
        try:
            user_logs_db = self.server['user_logs']
            user_logs_db.save(log_entry)
            logger.debug(f"Logged user action: {action_type}")
            return True
        except Exception as e:
            logger.error(f"Error logging user action: {e}")
            return False

    def save_study_design(self, client_id, study_design_data):
        """Save a study design document with client_id."""
        try:
            study_designs_db = self.server['study_designs']
            
            # Generate a unique study_id if not provided
            if 'study_id' not in study_design_data:
                study_design_data['study_id'] = f"{client_id}_{str(uuid.uuid4())[:8]}"
            
            # Add client_id and timestamp
            doc = {
                '_id': study_design_data['study_id'],
                'client_id': client_id,
                'timestamp': datetime.utcnow().isoformat(),
                'data': study_design_data
            }
            
            # Check if document already exists
            try:
                existing = study_designs_db[doc['_id']]
                doc['_rev'] = existing.rev
            except (ResourceNotFound, MockCouchDBResourceNotFound):
                pass
            
            study_designs_db.save(doc)
            logger.info(f"Saved study design: {doc['_id']}")
            return doc['_id']
        except Exception as e:
            logger.error(f"Error saving study design: {e}")
            return None

    def get_database_info(self):
        """Get information about all databases."""
        if self.use_mock:
            try:
                return {
                    "server_type": "JSON Mock",
                    "databases": self.server.mock_db.all_dbs()
                }
            except Exception as e:
                logger.error(f"Error getting database info from mock: {e}")
                return {"error": str(e)}
        else:
            try:
                response = requests.get(
                    f"{self.base_url}/_all_dbs",
                    auth=self.auth,
                    timeout=1  # Shorter timeout to fail faster
                )
                if response.status_code == 200:
                    return {
                        "server_type": "CouchDB",
                        "databases": response.json()
                    }
                else:
                    logger.error(f"Error getting database info: status code {response.status_code}")
                    # Fallback to mock if real CouchDB fails
                    logger.info("Falling back to mock database")
                    self.use_mock = True
                    self.server = MockCouchDBServer(self.base_url, self.auth)
                    return {
                        "server_type": "JSON Mock (fallback)",
                        "databases": self.server.mock_db.all_dbs()
                    }
            except requests.exceptions.RequestException as e:
                logger.error(f"Error getting database info: {e}")
                # Fallback to mock if real CouchDB fails
                logger.info("Falling back to mock database")
                self.use_mock = True
                self.server = MockCouchDBServer(self.base_url, self.auth)
                return {
                    "server_type": "JSON Mock (fallback)",
                    "databases": self.server.mock_db.all_dbs()
                }

if __name__ == "__main__":
    db_setup = DatabaseSetup()
    result = db_setup.setup_databases()
    print("Database setup completed successfully.")
    print(f"Created/verified databases: {', '.join(result.keys())}")
    
    # Print info about the database server
    info = db_setup.get_database_info()
    if "error" in info:
        print(f"Error getting database info: {info['error']}")
    else:
        print(f"Server type: {info['server_type']}")
        print(f"Available databases: {', '.join(info['databases'])}")