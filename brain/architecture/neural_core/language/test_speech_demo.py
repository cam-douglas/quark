#!/usr/bin/env python3
"""Speech Integration Demo - Demonstrates Quark's TTS capabilities.

This script shows how to use the speech integration with the Language Cortex.
"""

import sys
import time
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from brain.architecture.neural_core.language.speech_integration import (
    get_speech_integration,
    speak,
    enable_speech,
    disable_speech,
    get_speech_status
)

def demo_basic_tts():
    """Demonstrate basic TTS functionality."""
    print("🎤 === BASIC TTS DEMO ===")
    
    # Enable system TTS
    print("🔊 Enabling system TTS...")
    success = enable_speech("system_tts")
    if not success:
        print("❌ Failed to enable TTS")
        return
    
    # Test basic speech
    test_phrases = [
        "Hello! I am Quark, your AI assistant.",
        "The speech integration is now working correctly.",
        "I can speak using the system text-to-speech engine."
    ]
    
    for phrase in test_phrases:
        print(f"🗣️ Speaking: '{phrase}'")
        speak(phrase)
        time.sleep(1)  # Brief pause between phrases
    
    print("✅ Basic TTS demo completed")

def demo_speech_status():
    """Demonstrate speech status checking."""
    print("\n📊 === SPEECH STATUS DEMO ===")
    
    status = get_speech_status()
    
    print(f"Enabled: {status.get('enabled', False)}")
    print(f"Provider: {status.get('provider', 'unknown')}")
    print(f"Voice: {status.get('voice_name', 'unknown')}")
    print(f"Language: {status.get('language_code', 'unknown')}")
    print(f"System TTS Available: {status.get('system_tts_available', False)}")
    print(f"Live-Voice Available: {status.get('live_voice_available', False)}")

def demo_provider_switching():
    """Demonstrate switching between TTS providers."""
    print("\n🔄 === PROVIDER SWITCHING DEMO ===")
    
    # Test system TTS
    print("🖥️ Testing System TTS...")
    enable_speech("system_tts")
    speak("This is using the system text-to-speech engine.")
    
    time.sleep(2)
    
    # Note: Live-Voice would require API key setup
    print("📝 Note: Live-Voice Gemini requires API key configuration")
    print("   To test Live-Voice, add your Gemini API key to the credentials file")
    
    # Disable TTS
    print("🔇 Disabling TTS...")
    disable_speech()
    speak("This should not be heard since TTS is disabled.")
    print("✅ TTS disabled - no speech output expected")

def demo_integration_info():
    """Show integration information."""
    print("\n📋 === INTEGRATION INFO ===")
    
    integration = get_speech_integration()
    providers = integration.list_available_providers()
    
    print("Available Providers:")
    for i, provider in enumerate(providers, 1):
        print(f"  {i}. {provider}")
    
    print(f"\nLive-Voice Path: {integration.live_voice_path}")
    print(f"Config Path: {integration.config_path}")

def main():
    """Run the speech integration demo."""
    print("🚀 QUARK SPEECH INTEGRATION DEMO")
    print("=" * 50)
    
    try:
        demo_speech_status()
        demo_integration_info()
        demo_basic_tts()
        demo_provider_switching()
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 Usage Tips:")
        print("  - Use speech_cli.py for command-line control")
        print("  - Enable speech in Language Cortex with enable_speech=True")
        print("  - Configure Live-Voice with Gemini API key for advanced TTS")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
