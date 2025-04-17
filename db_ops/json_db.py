"""
JSON-based mock for CouchDB for use when CouchDB is unavailable.
This implementation saves data to JSON files in a directory structure.
"""

import os
import json
import uuid
from datetime import datetime
import shutil
import re
from pathlib import Path
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("json_db")

class MockCouchDBResourceNotFound(Exception):
    """Exception raised when a resource is not found in the mock CouchDB."""
    pass

class MockCouchDBPreconditionFailed(Exception):
    """Exception raised when a precondition fails in the mock CouchDB."""
    pass

class MockCouchDBDocument:
    """Mock document wrapper to mimic CouchDB document behavior."""
    def __init__(self, doc_id, data):
        self.id = doc_id
        self.rev = data.get('_rev', f"1-{uuid.uuid4().hex[:8]}")
        self._data = data
    
    def __getitem__(self, key):
        return self._data.get(key)
    
    def __setitem__(self, key, value):
        self._data[key] = value
    
    def get(self, key, default=None):
        return self._data.get(key, default)

class MockCouchDBView:
    """Mock view implementation for JSON-based CouchDB."""
    def __init__(self, db, view_name, map_fn):
        self.db = db
        self.view_name = view_name
        self.map_fn = map_fn
        # Parse the JavaScript map function to extract emit calls
        self.key_pattern = re.compile(r'emit\((.+?),')
    
    def __call__(self, **kwargs):
        """Execute the view and return results."""
        results = []
        
        # Try to extract the emit key from the map function
        key_matches = self.key_pattern.findall(self.map_fn)
        
        # Get all documents from the database
        for doc_id, doc_data in self.db._documents.items():
            # Create a simple JavaScript-like context
            doc = doc_data.copy()
            
            # For each key pattern extracted from the map function
            for key_pattern in key_matches:
                key_pattern = key_pattern.strip()
                
                # Handle the common pattern: doc.field_name
                if key_pattern.startswith('doc.'):
                    field_name = key_pattern.split('.')[1].strip()
                    
                    # If the field exists in the document and matches any query parameter
                    if field_name in doc:
                        if 'key' in kwargs and kwargs['key'] != doc[field_name]:
                            continue
                            
                        # Add to results
                        results.append({
                            'id': doc_id,
                            'key': doc[field_name],
                            'value': doc
                        })
        
        return {'rows': results, 'total_rows': len(results)}

class MockCouchDBDatabase:
    """Mock database implementation for JSON-based CouchDB."""
    def __init__(self, name, db_dir):
        self.name = name
        self.db_dir = os.path.join(db_dir, name)
        self._documents = {}
        self._design_docs = {}
        self._views = {}
        self._lock = threading.RLock()
        
        # Create the database directory if it doesn't exist
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Load existing documents
        self._load_documents()
    
    def _load_documents(self):
        """Load all documents from the database directory."""
        with self._lock:
            # Regular documents
            docs_dir = os.path.join(self.db_dir, 'docs')
            if os.path.exists(docs_dir):
                for filename in os.listdir(docs_dir):
                    if filename.endswith('.json'):
                        doc_id = filename[:-5]  # Remove .json
                        with open(os.path.join(docs_dir, filename), 'r') as f:
                            self._documents[doc_id] = json.load(f)
            
            # Design documents
            design_dir = os.path.join(self.db_dir, 'design')
            if os.path.exists(design_dir):
                for filename in os.listdir(design_dir):
                    if filename.endswith('.json'):
                        design_id = filename[:-5]  # Remove .json
                        with open(os.path.join(design_dir, filename), 'r') as f:
                            design_doc = json.load(f)
                            self._design_docs[design_id] = design_doc
                            
                            # Initialize views from design documents
                            if 'views' in design_doc:
                                for view_name, view_data in design_doc['views'].items():
                                    if 'map' in view_data:
                                        self._views[f"{design_id}/{view_name}"] = MockCouchDBView(
                                            self, view_name, view_data['map']
                                        )
    
    def _save_document(self, doc_id, data):
        """Save a document to disk."""
        with self._lock:
            # Check if it's a design document
            is_design = doc_id.startswith('_design/')
            
            if is_design:
                # Save design document
                design_dir = os.path.join(self.db_dir, 'design')
                os.makedirs(design_dir, exist_ok=True)
                design_name = doc_id.split('/')[-1]
                
                with open(os.path.join(design_dir, f"{design_name}.json"), 'w') as f:
                    json.dump(data, f, indent=2)
                
                self._design_docs[design_name] = data
                
                # Initialize or update views
                if 'views' in data:
                    for view_name, view_data in data['views'].items():
                        if 'map' in view_data:
                            self._views[f"{design_name}/{view_name}"] = MockCouchDBView(
                                self, view_name, view_data['map']
                            )
            else:
                # Save regular document
                docs_dir = os.path.join(self.db_dir, 'docs')
                os.makedirs(docs_dir, exist_ok=True)
                
                with open(os.path.join(docs_dir, f"{doc_id}.json"), 'w') as f:
                    json.dump(data, f, indent=2)
    
    def __getitem__(self, doc_id):
        """Get a document by ID."""
        with self._lock:
            # Handle design documents
            if doc_id.startswith('_design/'):
                design_name = doc_id.split('/')[-1]
                if design_name in self._design_docs:
                    return MockCouchDBDocument(doc_id, self._design_docs[design_name])
                else:
                    raise MockCouchDBResourceNotFound(f"Design document {doc_id} not found")
            
            # Handle regular documents
            if doc_id in self._documents:
                return MockCouchDBDocument(doc_id, self._documents[doc_id])
            else:
                raise MockCouchDBResourceNotFound(f"Document {doc_id} not found")
    
    def __setitem__(self, doc_id, data):
        """Set a document by ID - not used directly, use save()."""
        self.save(data, doc_id)
    
    def save(self, doc, doc_id=None):
        """Save a document."""
        with self._lock:
            # Generate ID if not provided
            if doc_id is None:
                if '_id' in doc:
                    doc_id = doc['_id']
                else:
                    doc_id = str(uuid.uuid4())
                    doc['_id'] = doc_id
            
            # Check for conflicts with existing document
            if doc_id in self._documents:
                existing_doc = self._documents[doc_id]
                if '_rev' in existing_doc and (
                    '_rev' not in doc or doc['_rev'] != existing_doc['_rev']
                ):
                    raise MockCouchDBPreconditionFailed("Document update conflict")
                
                # Update revision
                if '_rev' in existing_doc:
                    rev_num = int(existing_doc['_rev'].split('-')[0]) + 1
                    doc['_rev'] = f"{rev_num}-{uuid.uuid4().hex[:8]}"
                else:
                    doc['_rev'] = f"1-{uuid.uuid4().hex[:8]}"
            else:
                # New document
                doc['_rev'] = f"1-{uuid.uuid4().hex[:8]}"
            
            # Save document in memory and on disk
            if doc_id.startswith('_design/'):
                design_name = doc_id.split('/')[-1]
                self._design_docs[design_name] = doc
            else:
                self._documents[doc_id] = doc
            
            self._save_document(doc_id, doc)
            return doc_id, doc['_rev']
    
    def view(self, design_doc, view_name, **kwargs):
        """Query a view."""
        view_key = f"{design_doc}/{view_name}"
        if view_key in self._views:
            return self._views[view_key](**kwargs)
        else:
            raise MockCouchDBResourceNotFound(f"View {view_key} not found")

    def get(self, doc_id):
        """Get a document by ID.
        This method implements the same functionality as CouchDB's get method.
        """
        try:
            doc = self[doc_id]
            # Convert the MockCouchDBDocument to a dict format similar to CouchDB
            return doc._data
        except MockCouchDBResourceNotFound:
            return None

class MockCouchDB:
    """JSON-based mock for CouchDB that saves data to the filesystem."""
    def __init__(self, base_dir=None):
        if base_dir is None:
            # Default to user's home directory
            home_dir = str(Path.home())
            self.base_dir = os.path.join(home_dir, '.careframe', 'mock_couchdb')
        else:
            self.base_dir = base_dir
        
        # Create base directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Keep track of databases
        self._databases = {}
        
        # Load existing databases
        self._load_databases()
        
        logger.info(f"MockCouchDB initialized with data directory: {self.base_dir}")
    
    def _load_databases(self):
        """Load all databases from the base directory."""
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path):
                self._databases[item] = MockCouchDBDatabase(item, self.base_dir)
    
    def __contains__(self, db_name):
        """Check if a database exists."""
        return db_name in self._databases
    
    def __getitem__(self, db_name):
        """Get a database by name."""
        if db_name in self._databases:
            return self._databases[db_name]
        else:
            raise MockCouchDBResourceNotFound(f"Database {db_name} not found")
    
    def create(self, db_name):
        """Create a database."""
        if db_name in self._databases:
            raise MockCouchDBPreconditionFailed(f"Database {db_name} already exists")
        
        # Create database directory
        db_dir = os.path.join(self.base_dir, db_name)
        os.makedirs(db_dir, exist_ok=True)
        
        # Create and store the database
        self._databases[db_name] = MockCouchDBDatabase(db_name, self.base_dir)
        
        logger.info(f"Created database: {db_name}")
        return self._databases[db_name]
    
    def delete(self, db_name):
        """Delete a database."""
        if db_name not in self._databases:
            raise MockCouchDBResourceNotFound(f"Database {db_name} not found")
        
        # Delete database directory
        db_dir = os.path.join(self.base_dir, db_name)
        shutil.rmtree(db_dir)
        
        # Remove from memory
        del self._databases[db_name]
        
        logger.info(f"Deleted database: {db_name}")
        return True
    
    def all_dbs(self):
        """Get a list of all databases."""
        return list(self._databases.keys())

# Helper class to authenticate with the mock CouchDB
class MockCouchDBServer:
    """Mock CouchDB server that authenticates users and provides access to databases."""
    def __init__(self, base_url=None, auth=None):
        """Initialize the mock server with authentication credentials."""
        self.base_url = base_url  # Not used, but kept for compatibility
        self.resource = MockCouchDBResourceManager(auth)
        self.mock_db = MockCouchDB()
    
    def create(self, db_name):
        """Create a database."""
        return self.mock_db.create(db_name)
    
    def __getitem__(self, db_name):
        """Get a database by name."""
        return self.mock_db[db_name]
    
    def __contains__(self, db_name):
        """Check if a database exists."""
        return db_name in self.mock_db

class MockCouchDBResourceManager:
    """Mock resource manager for authentication."""
    def __init__(self, credentials=None):
        self.credentials = credentials

# Test code
if __name__ == "__main__":
    # Create a mock CouchDB
    db = MockCouchDB()
    
    # Create a database
    studies_db = db.create("studies")
    
    # Create a design document
    design_doc = {
        "_id": "_design/studies",
        "views": {
            "by_client_id": {
                "map": """function(doc) {
                    if (doc.client_id) {
                        emit(doc.client_id, doc);
                    }
                }"""
            }
        }
    }
    studies_db.save(design_doc)
    
    # Create some documents
    doc1 = {
        "client_id": "client1",
        "name": "Study 1",
        "date": "2023-01-01"
    }
    studies_db.save(doc1)
    
    doc2 = {
        "client_id": "client2",
        "name": "Study 2",
        "date": "2023-02-01"
    }
    studies_db.save(doc2)
    
    # Query a view
    results = studies_db.view("studies", "by_client_id")
    print(json.dumps(results, indent=2))
    
    # Query with a key
    results = studies_db.view("studies", "by_client_id", key="client1")
    print(json.dumps(results, indent=2)) 