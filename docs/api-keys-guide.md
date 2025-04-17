# API Keys Management Guide

This guide explains how CareFrame manages API keys for external services and provides instructions for configuring and securing them.

> ## ⚠️ IMPORTANT SECURITY WARNING
> 
> API keys for cloud-based LLMs (Claude, Gemini) will send your data to external services.
>
> **DO NOT use cloud-based LLMs with sensitive healthcare data or PHI.**
>
> For production or clinical use:
> - Use only local LLM models
> - Remove all external API keys or leave them empty
> - Consider implementing a local proxy server for LLM calls
>
> See [LICENSE](../LICENSE) for full disclaimer.

## Supported API Keys

CareFrame supports integration with various external services through API keys:

| Service | Environment Variable | Description |
|---------|---------------------|-------------|
| Anthropic Claude | `CLAUDE_API_KEY` | Advanced AI capabilities for language and vision tasks |
| Google Gemini | `GEMINI_API_KEY` | Google's AI for language processing |
| UMLS | `UMLS_API_KEY` | Medical terminology and concepts access |
| Unpaywall | `UNPAYWALL_EMAIL` | Email registration for open access paper retrieval |
| Zenodo | `ZENODO_API_KEY` | Research data repository access |
| CORE | `CORE_API_KEY` | Research paper aggregation service |
| Entrez | `ENTREZ_API_KEY` & `ENTREZ_EMAIL` | Access to PubMed and NCBI databases |

## API Key Security

CareFrame implements multiple security features to protect your API keys:

1. **Encryption**: All API keys are encrypted using Fernet symmetric encryption before storage
2. **Secure Storage**: Keys are stored in the CouchDB database, not in plain text files
3. **Machine-Specific Encryption**: The encryption key is bound to the host machine
4. **Environment Variable Support**: Keys can be provided via environment variables instead of UI input

## Configuration Methods

### Method 1: Settings UI

The simplest way to configure API keys is through the Settings interface:

1. Launch CareFrame application
2. Navigate to the Settings section
3. Enter your API keys in the corresponding fields
4. Click "Save API Keys"

All keys are encrypted before storage in the database.

### Method 2: Environment Variables

For automated deployments, you can set API keys via environment variables:

1. Create a `.env` file (or copy from `.env.example`)
2. Add your API keys:
   ```
   CLAUDE_API_KEY=sk-ant-api03-your-key-here
   GEMINI_API_KEY=your-gemini-key-here
   # ... other keys ...
   ```
3. When the application starts, these keys will be automatically loaded

### Method 3: Direct Database Configuration

For advanced users, API keys can be directly configured in the CouchDB database:

1. Access CouchDB at http://localhost:5984/_utils/ 
2. Navigate to the `settings` database
3. Edit the `api_keys` document

**Note:** You should never store unencrypted keys in the database. Use the UI or environment variables which handle encryption automatically.

## API Key Initialization

The application automatically handles API key initialization:

1. On first run, the application creates the settings database
2. Empty placeholders for all supported API keys are created
3. The encryption system is initialized
4. Any keys provided through environment variables are securely stored

This happens automatically through the database initialization in `app_launcher.py`.

## Obtaining API Keys

### Claude API Key
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Create an account or log in
3. Navigate to API Keys section
4. Generate a new API key

### Gemini API Key
1. Go to [Google AI Studio](https://ai.google.dev/)
2. Create an account or log in
3. Navigate to API Keys section
4. Generate a new API key

### UMLS API Key
1. Register at [UMLS Terminology Services](https://uts.nlm.nih.gov/)
2. Complete the application process
3. Once approved, obtain your API key from your account

### Other API Keys
See the links in the README.md file for instructions on obtaining other API keys.

## Troubleshooting

### API Key Not Working

If an API key isn't working:

1. Verify the key is valid by testing it directly with the service's API
2. Check that the key is entered correctly (no extra spaces or characters)
3. Ensure the key has the correct permissions and is active
4. Try re-entering the key through the Settings UI

### Encryption Issues

If you encounter encryption-related errors:

1. Check if the `.encryption_key` file exists and is not corrupt
2. In some cases, you may need to remove the `.encryption_key` file to generate a new one 