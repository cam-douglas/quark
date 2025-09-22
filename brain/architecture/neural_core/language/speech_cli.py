#!/usr/bin/env python3
"""Speech CLI - Command-line interface for testing and controlling Quark speech integration.

This module provides a simple CLI for testing text-to-speech functionality
and managing speech settings for the Quark language cortex.

Integration: Testing and control interface for speech integration.
Rationale: Provides easy way to test and configure TTS functionality.
"""

import argparse
import sys
import json
from typing import Optional
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from brain.architecture.neural_core.language.speech_integration import (
    SpeechIntegration, 
    TTSProvider,
    get_speech_integration,
    speak,
    enable_speech,
    disable_speech,
    get_speech_status
)

def test_speech(text: str, provider: Optional[str] = None) -> None:
    """Test text-to-speech with given text and optional provider."""
    print(f"🧪 Testing TTS with text: '{text}'")
    
    if provider:
        print(f"🔧 Enabling TTS with provider: {provider}")
        if not enable_speech(provider):
            print("❌ Failed to enable speech")
            return
    
    success = speak(text)
    if success:
        print("✅ TTS test completed successfully")
    else:
        print("❌ TTS test failed")

def show_status() -> None:
    """Show current speech integration status."""
    status = get_speech_status()
    
    print("\n" + "="*50)
    print("🔊 QUARK SPEECH INTEGRATION STATUS")
    print("="*50)
    
    print(f"📊 Enabled: {status.get('enabled', False)}")
    print(f"🎯 Provider: {status.get('provider', 'unknown')}")
    print(f"🗣️ Voice: {status.get('voice_name', 'unknown')}")
    print(f"🌍 Language: {status.get('language_code', 'unknown')}")
    print(f"⚡ Speech Rate: {status.get('speech_rate', 1.0)}")
    print(f"🔊 Volume: {status.get('volume', 0.8)}")
    
    print(f"\n🔧 Live-Voice Available: {status.get('live_voice_available', False)}")
    print(f"🖥️ System TTS Available: {status.get('system_tts_available', False)}")
    print(f"📁 Config Path: {status.get('config_path', 'unknown')}")
    
    if 'error' in status:
        print(f"❌ Error: {status['error']}")
    
    print("="*50)

def list_providers() -> None:
    """List available TTS providers."""
    try:
        integration = get_speech_integration()
        providers = integration.list_available_providers()
        
        print("\n📋 Available TTS Providers:")
        for i, provider in enumerate(providers, 1):
            print(f"  {i}. {provider}")
        
        if not providers:
            print("  No providers available")
            
    except Exception as e:
        print(f"❌ Error listing providers: {e}")

def enable_tts(provider: str) -> None:
    """Enable TTS with specified provider."""
    print(f"🔊 Enabling TTS with provider: {provider}")
    
    success = enable_speech(provider)
    if success:
        print("✅ TTS enabled successfully")
    else:
        print("❌ Failed to enable TTS")

def disable_tts() -> None:
    """Disable TTS."""
    print("🔇 Disabling TTS...")
    
    success = disable_speech()
    if success:
        print("✅ TTS disabled successfully")
    else:
        print("❌ Failed to disable TTS")

def test_language_cortex(text: str, enable_speech_flag: bool = False) -> None:
    """Test the Language Cortex with speech integration."""
    try:
        from brain.architecture.neural_core.language.language_processing import LanguageCortex
        
        print(f"🧠 Testing Language Cortex with speech={'enabled' if enable_speech_flag else 'disabled'}")
        
        # Initialize Language Cortex with speech option
        cortex = LanguageCortex(enable_speech=enable_speech_flag)
        
        if enable_speech_flag:
            # Enable system TTS by default for testing
            cortex.enable_speech_output("system_tts")
        
        # Process the input text
        response = cortex.process_input(text)
        
        print(f"\n📝 Input: {text}")
        print(f"🤖 Response: {response}")
        
        # Show processing status
        status = cortex.get_processing_status()
        speech_status = status.get('speech_integration', {})
        print(f"\n🔊 Speech Status: {speech_status.get('enabled', False)}")
        
    except Exception as e:
        print(f"❌ Error testing Language Cortex: {e}")

def install_live_voice_deps() -> None:
    """Install Live-Voice dependencies."""
    try:
        integration = get_speech_integration()
        
        print("📦 Installing Live-Voice dependencies...")
        
        # This will trigger dependency installation if needed
        live_voice_available = integration._check_live_voice_setup()
        
        if live_voice_available:
            print("✅ Live-Voice dependencies installed successfully")
        else:
            print("❌ Failed to install Live-Voice dependencies")
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Quark Speech Integration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test "Hello, this is a test"
  %(prog)s --status
  %(prog)s --enable system_tts
  %(prog)s --disable
  %(prog)s --providers
  %(prog)s --cortex-test "What is the weather like?" --speech
  %(prog)s --install-deps
        """
    )
    
    parser.add_argument(
        '--test', 
        metavar='TEXT',
        help='Test TTS with given text'
    )
    
    parser.add_argument(
        '--provider',
        choices=['system_tts', 'live_voice_gemini', 'disabled'],
        help='TTS provider to use for testing'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current speech integration status'
    )
    
    parser.add_argument(
        '--providers',
        action='store_true',
        help='List available TTS providers'
    )
    
    parser.add_argument(
        '--enable',
        metavar='PROVIDER',
        choices=['system_tts', 'live_voice_gemini'],
        help='Enable TTS with specified provider'
    )
    
    parser.add_argument(
        '--disable',
        action='store_true',
        help='Disable TTS'
    )
    
    parser.add_argument(
        '--cortex-test',
        metavar='TEXT',
        help='Test Language Cortex with given input text'
    )
    
    parser.add_argument(
        '--speech',
        action='store_true',
        help='Enable speech output for cortex test'
    )
    
    parser.add_argument(
        '--install-deps',
        action='store_true',
        help='Install Live-Voice dependencies'
    )
    
    args = parser.parse_args()
    
    # Handle commands
    if args.status:
        show_status()
    elif args.providers:
        list_providers()
    elif args.enable:
        enable_tts(args.enable)
    elif args.disable:
        disable_tts()
    elif args.test:
        test_speech(args.test, args.provider)
    elif args.cortex_test:
        test_language_cortex(args.cortex_test, args.speech)
    elif args.install_deps:
        install_live_voice_deps()
    else:
        # Default: show status
        show_status()

if __name__ == '__main__':
    main()
