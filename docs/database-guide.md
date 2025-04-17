# Database Guide

This guide provides detailed information about CareFrame's database architecture, including both CouchDB and the JSON mock database options.

## Database Architecture

CareFrame uses a document-oriented database approach and supports two backend options:

1. **CouchDB** (Default/Production): A mature, distributed NoSQL database
2. **JSON Mock Database** (Fallback/Development): A file-based JSON storage system

## Database Configuration

The database connection is configured in `db_ops/generate_tables.py`. Default settings:

- **URL**: http://localhost:5984
- **Authentication**: Basic auth with username "admin" and password "cfpwd"

You can customize these settings by:
- Editing the `app_launcher.py` parameters
- Using command-line arguments
- Using environment variables

## Database Selection Logic

On startup, CareFrame automatically detects which database to use:

1. Attempts to connect to CouchDB at the configured URL
2. If CouchDB is available, uses it
3. If CouchDB is unavailable, falls back to the JSON mock database
4. Command-line flags can override this auto-detection

## Databases and Their Purposes

CareFrame initializes multiple databases, each serving a specific purpose:

| Database | Purpose | Key Document Types |
|----------|---------|-------------------|
| `study_designs` | Study design docs | Design protocols, parameters |
| `user_logs` | User activity logs | User actions, timestamps |
| `users` | User management | User profiles, credentials |
| `studies` | Study data | Study metadata, relationships |
| `teams` | Team management | Team members, permissions |
| `logs` | Application logs | System events, errors |
| `settings` | Application settings | API keys, preferences |

## Database Views

CareFrame creates various database views to optimize common queries:

### study_designs views
- `by_client_id`: Retrieves designs filtered by client ID
- `by_study_id`: Retrieves designs filtered by study ID

### user_logs views
- `by_client_id`: User actions filtered by client ID
- `by_timestamp`: Actions ordered by timestamp
- `by_action_type`: Actions filtered by type

### studies views
- `by_name`: Studies filtered by name
- `by_team`: Studies filtered by team ID
- `by_date`: Studies ordered by creation date

### teams views
- `by_name`: Teams filtered by name

## JSON Mock Database

When CouchDB is unavailable, CareFrame uses a JSON-based mock implementation:

- Stored in `~/.careframe/mock_couchdb/` by default
- Mimics CouchDB's API for seamless switching
- Each database becomes a directory
- Documents are stored as individual JSON files
- Supports basic querying and views

## Security Considerations

- API keys in the `settings` database are encrypted
- Password hashes in the `users` database use secure hashing
- The `.encryption_key` file should be protected
- For production, configure proper CouchDB security

## Database Initialization

The database initialization process:

1. Creates all required databases if they don't exist
2. Sets up necessary views for each database
3. Creates empty API key placeholders
4. Establishes encryption keys for secure storage

This initialization is handled by:
- `init_database.py` script
- `DatabaseSetup` class in `db_ops/generate_tables.py`

## Monitoring Database Status

You can check the database status with:

```bash
python init_database.py --status
```

This command will report:
- Database type in use (CouchDB or JSON Mock)
- Connection status
- List of available databases

## Troubleshooting

### CouchDB Connection Issues

If CareFrame can't connect to CouchDB:

1. Verify CouchDB is running: `curl http://localhost:5984`
2. Check credentials in the configuration
3. Ensure the CouchDB port is accessible
4. Check the application logs for connection errors

### JSON Mock Database Issues

If there are issues with the JSON mock database:

1. Check if the `~/.careframe/mock_couchdb/` directory exists and is writable
2. Look for error messages in the application logs
3. Try removing the mock database directory and restart

### Data Migration

To migrate data between database types:

1. Use CouchDB's built-in export/import tools for CouchDB data
2. For the JSON mock DB, data is stored as plain JSON files in the mock directory
3. Custom scripts may be needed for complex migrations

## Advanced Configuration

### External CouchDB Configuration

For using an external CouchDB server:

```python
# Example configuration
base_url = "https://my-couchdb-server.example.com"
auth = ("my-username", "my-password")
db_setup = DatabaseSetup(base_url=base_url, auth=auth)
```

### Custom Database Location

To customize the JSON mock database location:

```python
# In a custom script or modified app_launcher.py
import os
from pathlib import Path
os.environ['CAREFRAME_DB_PATH'] = str(Path.home() / 'custom/path/to/db')
# Then initialize as normal
``` 