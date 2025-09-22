# Speech Integration for Quark Language Cortex

**Date**: 2025-01-22  
**Purpose**: Text-to-speech integration for Quark's language processing system

## Overview

This directory contains the speech integration system that enables text-to-speech (TTS) capabilities for Quark's Language Cortex. The integration supports multiple TTS providers including the Live-Voice Gemini system and system-native TTS engines.

## Components

### Core Files

| File | Description | Status |
|------|-------------|--------|
| [`speech_integration.py`](speech_integration.py) | Main TTS integration module with provider abstraction | ✔ active |
| [`speech_cli.py`](speech_cli.py) | Command-line interface for testing and configuration | ✔ active |
| [`live_voice/`](live_voice/) | Cloned Live-Voice Gemini repository | ✔ active |
| [`language_processing/__init__.py`](language_processing/__init__.py) | Updated with speech integration | ✔ active |

### Live-Voice Integration

The `live_voice/` directory contains the cloned repository from [cam-douglas/Live-Voice](https://github.com/cam-douglas/Live-Voice), which provides:

- **Gemini-based voice encoding**: Real-time audio processing with Google's Gemini 2.5 Flash model
- **3D Audio Visualization**: WebGL-based visual feedback for audio processing
- **TypeScript/JavaScript Implementation**: Web-based interface for voice interaction

## Features

### TTS Providers

1. **Live-Voice Gemini** (`live_voice_gemini`)
   - Uses Google's Gemini 2.5 Flash model with native audio dialog
   - Requires Gemini API key configuration
   - Provides high-quality voice synthesis
   - Currently integrated as TypeScript/JavaScript system

2. **System TTS** (`system_tts`)
   - Uses native system TTS engines
   - macOS: `say` command with configurable voices
   - Linux: `espeak` with speech rate and volume control
   - No external API dependencies

3. **Disabled** (`disabled`)
   - TTS functionality turned off
   - Default state for new installations

### Configuration

TTS settings are stored in `/Users/camdouglas/quark/data/credentials/speech_config.json`:

```json
{
  "enabled": false,
  "provider": "disabled",
  "voice_name": "Orus",
  "language_code": "en-US",
  "speech_rate": 1.0,
  "volume": 0.8
}
```

## Usage

### Language Cortex Integration

```python
from brain.architecture.neural_core.language.language_processing import LanguageCortex

# Initialize with speech enabled
cortex = LanguageCortex(enable_speech=True)

# Enable TTS with system provider
cortex.enable_speech_output("system_tts")

# Process input - response will be spoken automatically
response = cortex.process_input("Hello, how are you?")

# Disable speech output
cortex.disable_speech_output()

# Check speech status
status = cortex.get_speech_status()
print(f"Speech enabled: {status['enabled']}")
```

### Direct Speech Integration

```python
from brain.architecture.neural_core.language.speech_integration import (
    get_speech_integration, speak, enable_speech, disable_speech
)

# Enable system TTS
enable_speech("system_tts")

# Speak text directly
speak("This is a test of the speech system")

# Disable TTS
disable_speech()
```

### Command-Line Interface

```bash
# Show current status
python speech_cli.py --status

# Test TTS with system provider
python speech_cli.py --test "Hello world" --provider system_tts

# Enable TTS
python speech_cli.py --enable system_tts

# Test Language Cortex with speech
python speech_cli.py --cortex-test "What is the weather?" --speech

# List available providers
python speech_cli.py --providers

# Install Live-Voice dependencies
python speech_cli.py --install-deps
```

## Setup Instructions

### System TTS Setup

**macOS**: No additional setup required - uses built-in `say` command

**Linux**: Install espeak
```bash
sudo apt-get install espeak  # Ubuntu/Debian
sudo yum install espeak      # CentOS/RHEL
```

### Live-Voice Gemini Setup

1. **Install Node.js dependencies**:
   ```bash
   cd live_voice/
   npm install
   ```

2. **Configure Gemini API Key**:
   - Add your Gemini API key to `/Users/camdouglas/quark/data/credentials/all_api_keys.json`
   - Or create `.env.local` in the `live_voice/` directory:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```

3. **Test Live-Voice**:
   ```bash
   cd live_voice/
   npm run dev
   ```

## Integration Architecture

```
Language Cortex
├── LanguageAPIClients (existing)
├── ExpertRouter (existing)  
├── ModelSelector (existing)
└── SpeechIntegration (new)
    ├── TTSProvider enum
    ├── TTSConfig dataclass
    └── Provider implementations
        ├── Live-Voice Gemini
        ├── System TTS (say/espeak)
        └── Disabled
```

## Configuration Management

The speech integration uses a layered configuration approach:

1. **Default Configuration**: Built-in defaults with TTS disabled
2. **File Configuration**: JSON config file for persistent settings
3. **Runtime Configuration**: Dynamic enable/disable via API calls
4. **Provider-Specific Settings**: Voice names, rates, volumes per provider

## Error Handling

- **Graceful Degradation**: Falls back to text-only if TTS fails
- **Provider Fallback**: Can switch between providers if one fails
- **Dependency Checking**: Validates provider availability before use
- **Configuration Validation**: Ensures valid settings and handles corruption

## Testing

### Unit Tests
```bash
# Run speech integration tests
pytest tests/test_speech_integration.py

# Test with specific provider
python speech_cli.py --test "Testing speech" --provider system_tts
```

### Integration Tests
```bash
# Test full Language Cortex with speech
python speech_cli.py --cortex-test "Tell me about the weather" --speech
```

## Troubleshooting

### Common Issues

1. **"Speech integration not available"**
   - Check that `speech_integration.py` is in the correct location
   - Verify import paths are correct

2. **"No system TTS found"**
   - macOS: Ensure `say` command is available
   - Linux: Install `espeak` package

3. **"Live-Voice dependencies not installed"**
   - Run `npm install` in the `live_voice/` directory
   - Check Node.js is installed and accessible

4. **"Gemini API key not found"**
   - Add API key to credentials file
   - Check API key permissions and quota

### Debug Mode

Enable debug logging:
```python
import logging
logging.getLogger('brain.architecture.neural_core.language.speech_integration').setLevel(logging.DEBUG)
```

## Future Enhancements

- [ ] **Enhanced Live-Voice Integration**: Direct Python bindings for Gemini TTS
- [ ] **Voice Cloning**: Custom voice training and synthesis
- [ ] **Speech Recognition**: Input speech processing (STT)
- [ ] **Emotion Synthesis**: Emotional tone control in TTS output
- [ ] **Multi-Language Support**: Expanded language and accent options
- [ ] **Real-time Processing**: Streaming TTS for long responses

## Dependencies

### Python Dependencies
- `subprocess` (built-in)
- `json` (built-in)
- `pathlib` (built-in)
- `logging` (built-in)

### System Dependencies
- **macOS**: `say` command (built-in)
- **Linux**: `espeak` package
- **Node.js**: For Live-Voice integration

### API Dependencies
- **Google Gemini API**: For Live-Voice provider
- **Credentials**: Stored in `/Users/camdouglas/quark/data/credentials/`

## Security Considerations

- API keys stored in secure credentials directory
- No sensitive data logged in speech processing
- Provider isolation prevents cross-contamination
- Configuration validation prevents injection attacks

---

**Last Updated**: 2025-01-22  
**Maintainer**: Quark Language Cortex Team  
**Integration Status**: Active and ready for use
