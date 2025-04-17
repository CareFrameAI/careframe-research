# Local LLM Support

CareFrame now supports integration with local LLMs, allowing you to use your own locally-hosted language models instead of cloud-based APIs. This provides several benefits:

- **Privacy**: All data processing happens on your machine, with no data sent to external servers
- **Cost-free**: No usage costs compared to cloud API providers
- **Offline use**: Works without an internet connection
- **Flexibility**: Use any compatible local LLM server

## Supported Local LLM Servers

CareFrame supports any LLM server with an OpenAI-compatible API, including:

- [Ollama](https://ollama.ai/) - Run Llama, Mistral, Gemma and other models locally
- [LM Studio](https://lmstudio.ai/) - User-friendly local LLM server
- [LocalAI](https://localai.io/) - Self-hosted, community-driven local LLM API
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput and memory-efficient inference server

## Setup Instructions

### 1. Install and Run a Local LLM Server

#### Using Ollama (Recommended)

1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull a model (example: `ollama pull llama3`)
3. Start the server (typically runs automatically after installation)

#### Using LM Studio

1. Install LM Studio from [lmstudio.ai](https://lmstudio.ai/)
2. Download your desired model
3. Start the local server from the application

### 2. Configure CareFrame to Use Local LLMs

You can configure local LLM settings in two ways:

#### Using the UI

1. Open CareFrame
2. Navigate to the LLM Manager
3. Go to the "Local LLMs" tab
4. Check "Use Local LLMs"
5. Configure your endpoints and model names
6. Click "Test Local LLM Connection" to verify everything is working

#### Using Environment Variables

Set these environment variables before starting CareFrame:

```bash
# Enable Local LLM support
export USE_LOCAL_LLM=true

# Configure endpoints
export LOCAL_TEXT_LLM_ENDPOINT="http://localhost:11434/v1"
export LOCAL_VISION_LLM_ENDPOINT="http://localhost:11434/v1"
export LOCAL_JSON_LLM_ENDPOINT="http://localhost:11434/v1"

# Configure model names (must be available on your local LLM server)
export LOCAL_TEXT_LLM_MODEL="llama3"
export LOCAL_VISION_LLM_MODEL="llava"
export LOCAL_JSON_LLM_MODEL="llama3"
```

## Available Models

CareFrame includes predefined configurations for these local models:

- `local-llama3` - General text generation and comprehension
- `local-mistral` - Alternative general-purpose model
- `local-llava` - Vision model capable of image understanding

You can also configure any other model available on your local LLM server through the UI.

## Usage Notes

- **Performance**: Local LLMs require significant compute resources. Performance will depend on your hardware.
- **Vision Support**: Not all local LLMs support image understanding. To use vision features, make sure your local server has a multimodal model like LLaVA.
- **Token Limits**: Local models typically have lower context windows than cloud models. Adjust the token limits in the UI if needed.
- **Model Quality**: Local models may provide different quality results than cloud models, especially on complex tasks.

## Troubleshooting

### Common Issues

1. **Connection Error**: Make sure your local LLM server is running and the endpoint is correct.
2. **Model Not Found**: Ensure the model name exactly matches what's available on your server.
3. **Slow Responses**: Consider using a smaller model or upgrading your hardware.
4. **Out of Memory**: Reduce the max tokens parameter or use a smaller model.

### Testing Local LLM Integration

You can run the included test script to verify your setup:

```bash
python tests/test_local_llm.py
```

This will test basic text and JSON completion using your local LLM configuration. 