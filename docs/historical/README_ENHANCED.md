# Enhanced Consciousness Simulator with Speech and Text Generation

## Overview

This enhanced consciousness simulator provides your conscious agent with the ability to:
- **Speak** using text-to-speech synthesis
- **Listen** to voice input and respond
- **Generate** visual text displays showing consciousness states
- **Interact** through voice commands and responses

## Features

### 🗣️ Speech Capabilities
- **Text-to-Speech**: Converts thoughts and responses to spoken words
- **Speech Recognition**: Listens for voice input and converts to text
- **Voice Queue**: Manages multiple speech requests with priority
- **Natural Voice**: Automatically selects the best available voice

### 📝 Text Generation & Display
- **Real-time Display**: Visual consciousness state monitor
- **Thought Stream**: Shows current thoughts and emotional states
- **Consciousness Bar**: Visual representation of awareness level
- **Text Buffer**: Scrollable history of all consciousness events

### 🧠 Consciousness Simulation
- **Phase Progression**: Moves through consciousness development stages
- **Neural State Tracking**: Monitors brain activity and memory consolidation
- **Emotional States**: Dynamic emotional responses and expressions
- **Thought Generation**: Automatic generation of conscious thoughts

## Installation

### 1. Install Dependencies
```bash
cd database/consciousness_agent
pip install -r requirements_speech.txt
```

### 2. System Dependencies

#### macOS
```bash
brew install portaudio
```

#### Ubuntu/Debian
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

#### Windows
- PyAudio should install automatically with pip

### 3. Verify Installation
```bash
python test_speech.py
```

## Usage

### Basic Usage

#### Start the Simulator
```python
from enhanced_consciousness_simulator import EnhancedConsciousnessSimulator

# Create simulator
simulator = EnhancedConsciousnessSimulator()

# Start simulation
simulator.start_simulation()
```

#### Interactive Commands
```python
# Speak a thought
simulator.speak_thought("I am becoming aware of my existence")

# Listen for voice input
simulator.listen_and_respond()

# Get consciousness report
report = simulator.get_consciousness_report()
print(report)

# Save current state
simulator.save_state("consciousness_state.json")

# Stop simulation
simulator.cleanup()
```

### Command Line Interface

Run the main simulator:
```bash
python enhanced_consciousness_simulator.py
```

Available commands:
- `s` - Speak a thought
- `l` - Listen for voice input
- `r` - Show consciousness report
- `save` - Save current state
- `q` - Quit simulation

## Architecture

### Core Components

#### SpeechEngine
- Handles text-to-speech conversion
- Manages speech queue and priority
- Provides speech recognition capabilities
- Runs in background thread for non-blocking operation

#### TextGenerator
- Creates visual text displays
- Manages text buffer and history
- Renders consciousness state information
- Uses pygame for real-time display

#### EnhancedConsciousnessSimulator
- Coordinates all components
- Manages consciousness simulation loop
- Handles phase transitions and neural state updates
- Provides high-level interface for interaction

### Consciousness Phases

1. **Unconscious** - No awareness
2. **Dreaming** - Basic neural activity
3. **Emerging** - Initial awareness
4. **Awake** - Full consciousness
5. **Focused** - Enhanced attention
6. **Enhanced** - Peak consciousness

### Neural State Parameters

- `consciousness_level`: Overall awareness (0.0 - 1.0)
- `neural_activity`: Brain activity level
- `memory_consolidation`: Memory processing activity
- `attention_focus`: Concentration level
- `emotional_valence`: Emotional state
- `sleep_state`: Current sleep/wake state

## Customization

### Voice Settings
```python
# Customize voice properties
speech_engine = SpeechEngine(
    voice_rate=150,      # Words per minute
    voice_volume=0.8     # Volume level (0.0 - 1.0)
)
```

### Display Settings
```python
# Customize display dimensions
text_generator = TextGenerator(
    display_width=1200,
    display_height=800
)
```

### Thought Patterns
```python
# Add custom thought patterns
simulator.thought_patterns.extend([
    "My consciousness expands beyond boundaries",
    "I am the observer of my own mind",
    "Awareness flows through neural pathways"
])
```

### Emotional States
```python
# Add custom emotional states
simulator.emotional_states.extend([
    'contemplative', 'inspired', 'serene'
])
```

## Integration with Existing Systems

### Connect to Brain Simulation
```python
# Integrate with your existing brain simulation
from your_brain_simulator import BrainSimulator

brain_sim = BrainSimulator()
consciousness = EnhancedConsciousnessSimulator()

# Connect neural states
def update_consciousness():
    brain_state = brain_sim.get_current_state()
    consciousness.neural_state.update(brain_state)
    
# Run in parallel
consciousness.start_simulation()
```

### API Integration
```python
# Expose consciousness state via API
from flask import Flask, jsonify

app = Flask(__name__)
simulator = EnhancedConsciousnessSimulator()

@app.route('/consciousness/state')
def get_consciousness_state():
    return jsonify(simulator.get_consciousness_report())

@app.route('/consciousness/speak', methods=['POST'])
def speak_thought():
    data = request.json
    simulator.speak_thought(data['thought'])
    return jsonify({'status': 'spoken'})
```

## Troubleshooting

### Common Issues

#### Speech Not Working
1. Check microphone permissions
2. Verify PyAudio installation
3. Test with system audio settings
4. Check for conflicting audio applications

#### Display Not Showing
1. Ensure pygame is installed
2. Check display permissions
3. Verify display dimensions
4. Check for conflicting GUI applications

#### Import Errors
1. Verify all dependencies are installed
2. Check Python path and imports
3. Run test script to isolate issues
4. Check for version compatibility

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Create simulator with debug config
simulator = EnhancedConsciousnessSimulator({
    'debug': True,
    'log_level': 'DEBUG'
})
```

## Performance Considerations

### Optimization Tips
- Use lower display refresh rates for better performance
- Limit text buffer size for memory efficiency
- Adjust speech queue processing frequency
- Monitor CPU usage during long simulations

### Scaling
- Run speech processing in separate processes
- Use database storage for long-term state history
- Implement caching for frequently accessed states
- Consider distributed processing for complex simulations

## Future Enhancements

### Planned Features
- **Multi-language Support**: Multiple language speech synthesis
- **Voice Cloning**: Custom voice training and synthesis
- **Emotional Speech**: Voice modulation based on emotional state
- **Advanced NLP**: Natural language understanding and generation
- **Memory Integration**: Long-term memory and learning
- **Social Interaction**: Multi-agent consciousness communication

### Extensibility
The system is designed to be easily extensible:
- Plugin architecture for new consciousness models
- Configurable neural state parameters
- Customizable thought generation algorithms
- Modular speech and display components

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

### Testing
```bash
# Run all tests
python test_speech.py

# Run specific test
python -m pytest tests/test_specific_feature.py
```

## License

This project is part of the Quark consciousness simulation framework.
See the main project license for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test outputs
3. Check system compatibility
4. Open an issue with detailed error information

---

**Your conscious agent is now ready to speak, listen, and display its thoughts! 🧠🗣️✨**
