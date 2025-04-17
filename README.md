# CareFrame

CareFrame is an integrated research and clinical data platform designed to streamline the process of data collection, analysis, and interpretation for healthcare and research professionals.

> ## ⚠️ IMPORTANT SECURITY WARNING
> 
> CareFrame v0.1 is **EXPERIMENTAL** and by default may use cloud-based LLM providers (Anthropic Claude, Google Gemini) when API keys are configured.
>
> **DO NOT use cloud-based LLMs with sensitive healthcare data or PHI.**
>
> For production or clinical use:
> - Use only local LLM models
> - Remove all external API keys
> - Configure appropriate security measures
> - Follow all applicable data privacy regulations (HIPAA, GDPR, etc.)
>
> See [LICENSE](LICENSE) for full disclaimer.

## Features

- **Research Planning & Hypothesis Generation**: Design studies and generate hypotheses
- **Literature Search & Evidence Collection**: Find and organize research literature
- **Data Collection & Management**: Collect, clean, reshape, and manage clinical data
- **Analysis Tools**: Statistical analysis, sensitivity analysis, subgroup analysis, and more
- **Blockchain Integration**: Secure data exchange and validation
- **AI Assistant**: Built-in agent to help with research tasks
- **BioNLP Annotation**: Tools for biomedical text annotation
- **Secure API Key Management**: Encrypted storage of API keys for external services
- **Local LLM Support**: Use local LLM models instead of cloud-based APIs for privacy

## Getting Started

### Prerequisites

- Python 3.9+
- PyQt6
- FastAPI
- Various scientific and data analysis libraries

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/CareFrameAI/careframe-research.git
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure your API keys (optional, see API Keys Configuration section below)

4. Launch the application:
   ```
   python app_launcher.py
   ```

### Database Options

CareFrame can use either CouchDB or a built-in JSON-based mock database:

- **CouchDB**: If CouchDB is installed and running, CareFrame will connect to it automatically.
  - Requires CouchDB 3.x+ to be installed separately
  - Default credentials: admin/cfpwd

- **JSON Mock Database**: If CouchDB is not available, CareFrame will automatically fall back to using a JSON-based mock database that stores data in files.
  - Better for development or when CouchDB isn't available
  - Data is stored in files for persistence

You can control this behavior with the following options:

```bash
# Check database status without starting the app
python init_database.py --status

# Force using the JSON mock database
python app_launcher.py --mock

# Force using CouchDB (will fail if not available)
python app_launcher.py --couchdb

# Generate a default database configuration file
python init_database.py --config
```

The JSON mock database stores files in `~/.careframe/mock_couchdb/` by default.

### API Keys Configuration

CareFrame requires various API keys to access external services for full functionality. There are multiple ways to configure these keys:

#### Method 1: Using the Settings UI

On first run, the application will create placeholders for these keys in the database. You can then configure them through the UI:

1. Launch CareFrame
2. Navigate to the Settings section
3. Enter your API keys in the corresponding fields
4. Click "Save API Keys"

#### Method 2: Using Environment Variables

You can set API keys through environment variables:

1. Copy the `.env.example` file to `.env`
2. Edit the `.env` file with your actual API keys
3. The keys will be automatically loaded and encrypted in the database at startup

The following API keys are supported:

- **Claude API Key** (`CLAUDE_API_KEY`): For advanced AI language and vision capabilities
  - Obtain from: [Anthropic Console](https://console.anthropic.com/)

- **Gemini API Key** (`GEMINI_API_KEY`): For AI-powered language capabilities
  - Obtain from: [Google AI Studio](https://ai.google.dev/)

- **UMLS API Key** (`UMLS_API_KEY`): For medical terminology and concepts
  - Obtain from: [UMLS Terminology Services](https://uts.nlm.nih.gov/)

- **Unpaywall Email** (`UNPAYWALL_EMAIL`): For retrieving open access research papers
  - Register at: [Unpaywall](https://unpaywall.org/products/api)

- **Zenodo API Key** (`ZENODO_API_KEY`): For accessing research data repositories
  - Obtain from: [Zenodo](https://zenodo.org/)

- **CORE API Key** (`CORE_API_KEY`): For research paper aggregation
  - Register at: [CORE API](https://core.ac.uk/services/api)

- **Entrez API Key/Email** (`ENTREZ_API_KEY`/`ENTREZ_EMAIL`): For accessing PubMed and other NCBI databases
  - Register at: [NCBI](https://www.ncbi.nlm.nih.gov/account/)

**Security Note**: All API keys are securely encrypted using Fernet encryption before being stored in the database.

### Local LLM Integration

CareFrame supports using local LLM models instead of cloud-based APIs:

1. Install a local LLM server like [Ollama](https://ollama.ai/) or [LM Studio](https://lmstudio.ai/)
2. Start your local LLM server
3. In CareFrame, navigate to LLM Manager > Local LLMs tab
4. Enable Local LLMs and configure your endpoints and models
5. Test the connection

For detailed instructions, see [Local LLM Support](docs/local_llm_support.md).

## Database Schema and Tables

CareFrame automatically initializes several databases and their corresponding views:

- **study_designs**: Stores study design documents
- **user_logs**: Tracks user actions and interactions
- **users**: User management and authentication
- **studies**: Storage for study data and configurations
- **teams**: Team management and permissions
- **logs**: General application logs
- **settings**: Application settings including encrypted API keys

The database initialization happens automatically when you use `app_launcher.py`. This process:
- Creates necessary databases if they don't exist
- Sets up required views for each database
- Initializes empty API key placeholders
- Establishes encryption keys for secure storage

## Project Structure

- `app.py`: Main application entry point with PyQt6 UI
- `server.py`: WebSocket server for communications
- `app_launcher.py`: Launcher that initializes the database before starting the app
- `init_database.py`: Database initialization and management utilities
- `init_api_keys.py`: Script to initialize API keys from environment variables
- `bionlp/`: BioNLP annotation tools
- `data/`: Data management and analysis modules
- `exchange/`: Blockchain data exchange components
- `literature_search/`: Literature search tools
- `study_model/`: Study management models
- `qt_sections/`: UI components
- `db_ops/`: Database operations and JSON mock implementation
- `llms/`: Language model integration modules

## Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- List of contributors and acknowledgments

## Documentation

Detailed documentation is available in the `docs` directory:

- [API Keys Management](docs/api-keys-guide.md): Guide to managing API keys for external services
- [Database Guide](docs/database-guide.md): Details about database architecture and configuration
- [Local LLM Support](docs/local_llm_support.md): Using local LLM models for privacy 