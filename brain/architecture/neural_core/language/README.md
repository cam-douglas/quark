# Language Processing & Speech Integration

**Date**: 2025-01-22  
**Path**: `brain/architecture/neural_core/language`

**Purpose**: Language processing and speech integration for Quark's neural core, providing text-to-speech capabilities and multi-provider language model routing.

## 🧠 **Core Components**

### **Language Processing** ([`language_processing/`](language_processing/))
Multi-provider language model system with intelligent routing (4 files):
- **[`__init__.py`](language_processing/__init__.py)** - Main LanguageCortex class with speech integration
- **[`api_clients.py`](language_processing/api_clients.py)** - Secure API client management  
- **[`expert_router.py`](language_processing/expert_router.py)** - Semantic routing for model selection
- **[`model_selector.py`](language_processing/model_selector.py)** - Intelligent model selection logic

### **Speech Integration** 
Text-to-speech system with multiple provider support:
- **[`speech_integration.py`](speech_integration.py)** - Core TTS integration with provider abstraction ✔ active
- **[`speech_cli.py`](speech_cli.py)** - Command-line interface for TTS testing and control ✔ active
- **[`test_speech_demo.py`](test_speech_demo.py)** - Demonstration script for speech capabilities ✔ active
- **[`SPEECH_INTEGRATION.md`](SPEECH_INTEGRATION.md)** - Comprehensive documentation ✔ active

### **Live-Voice Integration** ([`live_voice/`](live_voice/))
Gemini-based voice processing system (cloned from [cam-douglas/Live-Voice](https://github.com/cam-douglas/Live-Voice)):
- **TypeScript/JavaScript Implementation** - Web-based voice interface ✔ active
- **Gemini 2.5 Flash Integration** - Real-time audio processing ✔ active
- **3D Audio Visualization** - WebGL-based visual feedback ✔ active
- **Node.js Dependencies** - Installed and ready for use ✔ active

## 🔊 **Speech Features**

### **TTS Providers**
1. **System TTS** (`system_tts`) - Native OS text-to-speech (macOS `say`, Linux `espeak`)
2. **Live-Voice Gemini** (`live_voice_gemini`) - Google Gemini 2.5 Flash with native audio
3. **Disabled** (`disabled`) - TTS functionality turned off

### **Configuration Management**
- **Persistent Settings**: JSON configuration in `/Users/camdouglas/quark/data/credentials/speech_config.json`
- **Runtime Control**: Enable/disable TTS via API calls
- **Provider Switching**: Dynamic switching between TTS engines
- **Voice Customization**: Configurable voices, rates, and volumes

## 📋 **Usage Examples**

### **Language Cortex with Speech**
```python
from brain.architecture.neural_core.language.language_processing import LanguageCortex

# Initialize with speech enabled
cortex = LanguageCortex(enable_speech=True)

# Enable system TTS
cortex.enable_speech_output("system_tts")

# Process input - response will be spoken automatically
response = cortex.process_input("Hello, how are you?")
```

### **Direct TTS Control**
```python
from brain.architecture.neural_core.language.speech_integration import speak, enable_speech

# Enable and use TTS
enable_speech("system_tts")
speak("This text will be spoken aloud")
```

### **Command-Line Interface**
```bash
# Show status
python speech_cli.py --status

# Test TTS
python speech_cli.py --test "Hello world" --provider system_tts

# Test Language Cortex with speech
python speech_cli.py --cortex-test "What is AI?" --speech
```

## 🛠️ **Setup Instructions**

### **System TTS Setup**
- **macOS**: No setup required (uses built-in `say`)
- **Linux**: Install `espeak` package

### **Live-Voice Gemini Setup**
1. Install dependencies: `cd live_voice && npm install` ✅ **Completed**
2. Configure Gemini API key in credentials file
3. Test with: `npm run dev`

## 📁 **Directory Structure**

```
language/
├── language_processing/          # Core language processing
│   ├── __init__.py              # Main LanguageCortex (with speech)
│   ├── api_clients.py           # Secure API management
│   ├── expert_router.py         # Semantic routing
│   └── model_selector.py        # Model selection logic
├── live_voice/                  # Cloned Live-Voice repository
│   ├── index.tsx               # Main TypeScript interface
│   ├── package.json            # Node.js dependencies
│   └── [additional files]      # Voice processing components
├── speech_integration.py        # Core TTS integration
├── speech_cli.py               # CLI for testing/control
├── test_speech_demo.py         # Demo script
├── SPEECH_INTEGRATION.md       # Detailed documentation
├── language_cortex.py          # Legacy compatibility layer
└── README.md                   # This file
```

## 🔗 **Integration Status**

| Component | Status | Description |
|-----------|--------|-------------|
| **Language Cortex** | ✅ **Active** | Multi-provider LLM with speech integration |
| **Speech Integration** | ✅ **Active** | TTS system with multiple providers |
| **System TTS** | ✅ **Ready** | Native OS text-to-speech available |
| **Live-Voice** | ✅ **Installed** | Gemini voice system dependencies ready |
| **CLI Tools** | ✅ **Active** | Command-line testing and control |
| **Configuration** | ✅ **Active** | Persistent settings management |

## 🧪 **Testing**

All speech integration functionality has been tested and verified:
- ✅ TTS provider detection and switching
- ✅ System TTS on macOS (`say` command)
- ✅ Live-Voice dependency installation
- ✅ Language Cortex speech integration
- ✅ Configuration persistence
- ✅ CLI interface functionality

## 🔗 **Links**
- [Speech Integration Documentation](SPEECH_INTEGRATION.md)
- [Live-Voice Repository](https://github.com/cam-douglas/Live-Voice)
- [Neural Core README](../README.md)
- [Brain Architecture](../../README.md)
- [Root README](../../../../README.md)
