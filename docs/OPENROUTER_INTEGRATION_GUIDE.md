# OpenRouter Integration Guide

**Date**: 2025-01-14  
**Status**: ✅ Active - Fully Integrated  

## Overview

OpenRouter has been successfully integrated into the Quark brain system, providing access to multiple LLM providers through a unified API interface. This integration allows the system to leverage various AI models including OpenAI GPT, Anthropic Claude, Google Gemini, and others through a single endpoint.

## Features

- **Multi-Provider Access**: Access to 20+ different AI models through one API
- **Cost Optimization**: Competitive pricing across different providers
- **Fallback Support**: Automatic routing when primary providers are unavailable  
- **Unified Interface**: Same API structure as OpenAI for easy integration
- **Expert Routing**: Intelligent model selection based on task type

## Configuration

### API Key Setup

Your OpenRouter API key has been securely stored in:
- **File**: `data/credentials/all_api_keys.json`
- **Services Section**: `services.openrouter.api_key`
- **Top-Level**: `openrouter_api_key` (for backward compatibility)

### Integration Points

The OpenRouter integration is implemented across these components:

1. **API Loader** (`brain/architecture/neural_core/language/api_loader.py`)
   - Loads OpenRouter API key from secure credentials
   - Validates key format and availability
   - Provides fallback to environment variables

2. **API Clients** (`brain/architecture/neural_core/language/language_processing/api_clients.py`)
   - Initializes OpenRouter client using OpenAI-compatible interface
   - Provides `query_openrouter()` method for model queries
   - Includes service availability checking

3. **Expert Router** (`brain/architecture/neural_core/language/language_processing/expert_router.py`)
   - Includes OpenRouter in routing preferences
   - Weighted selection based on task intent
   - Fallback routing when other services unavailable

## Available Models

OpenRouter provides access to models from multiple providers:

### Tested Models
- **OpenAI**: `openai/gpt-3.5-turbo`, `openai/gpt-4`
- **Anthropic**: `anthropic/claude-3-haiku`, `anthropic/claude-3-sonnet`
- **Google**: `google/gemma-2-9b-it`
- **Meta**: `meta-llama/llama-3.1-8b-instruct`
- **Microsoft**: `microsoft/phi-3-mini-128k-instruct`

### Model Selection Strategy

The expert router assigns OpenRouter with these preferences by task type:
- **General Conversation**: 20% preference
- **Scientific Analysis**: 20% preference  
- **Creative Writing**: 20% preference
- **Technical Explanation**: 20% preference
- **Problem Solving**: 20% preference
- **Learning Assistance**: 20% preference

## Usage Examples

### Direct API Usage

```python
from brain.architecture.neural_core.language.language_processing.api_clients import LanguageAPIClients

# Initialize clients
clients = LanguageAPIClients()

# Query OpenRouter with default model
response = clients.query_openrouter("What is machine learning?")

# Query specific model
response = clients.query_openrouter(
    "Explain quantum computing", 
    model="anthropic/claude-3-haiku"
)
```

### Service Availability Check

```python
# Check if OpenRouter is available
available_services = clients.get_available_services()
if available_services['openrouter']:
    print("OpenRouter is ready!")
```

### Expert Routing Integration

The expert router automatically considers OpenRouter when selecting the best service for a task:

```python
from brain.architecture.neural_core.language.language_processing.expert_router import ExpertRouter

router = ExpertRouter()
intent = router.route_to_expert("Help me write a Python function")
service = router.select_api_service(intent, available_services)
# May select 'openrouter' based on preferences and availability
```

## Benefits

1. **Cost Efficiency**: Competitive pricing compared to direct provider APIs
2. **Reliability**: Multiple provider fallbacks reduce downtime
3. **Model Diversity**: Access to latest models from various providers
4. **Simplified Management**: Single API key for multiple providers
5. **Rate Limit Distribution**: Spread requests across providers

## Monitoring & Troubleshooting

### Service Status
The system automatically reports OpenRouter status during initialization:
- ✅ "OpenRouter client initialized from secure credentials"
- ❌ "OpenRouter key not found in credentials directory"

### Error Handling
- Connection failures are logged with specific error details
- Automatic fallback to other available services
- Non-blocking initialization (system works without OpenRouter)

### Rate Limits
OpenRouter implements its own rate limiting across providers. Monitor usage through:
- OpenRouter dashboard
- API response headers
- Error messages for rate limit exceeded

## Security

- API key stored in secure credentials directory
- No hardcoded keys in source code
- Automatic key validation on startup
- Environment variable fallback support

## Future Enhancements

Potential improvements for the OpenRouter integration:

1. **Dynamic Model Selection**: Automatically choose best model based on task complexity
2. **Cost Tracking**: Monitor usage costs across different providers
3. **Performance Metrics**: Track response times and quality by provider
4. **Load Balancing**: Distribute requests optimally across available models
5. **Custom Routing**: User-defined preferences for specific use cases

## Integration Status

| Component | Status | Notes |
|-----------|---------|--------|
| API Key Storage | ✅ Complete | Securely stored in credentials |
| Client Initialization | ✅ Complete | OpenAI-compatible client |
| Query Methods | ✅ Complete | Full chat completion support |
| Expert Routing | ✅ Complete | Weighted preferences implemented |
| Error Handling | ✅ Complete | Graceful fallbacks |
| Documentation | ✅ Complete | This guide |

---

**Last Updated**: 2025-01-14  
**Integration Version**: 1.0  
**Compatibility**: All Quark brain modules  
**Dependencies**: `openai` package (already installed)
