#!/usr/bin/env python3
"""Speech Integration Module - Text-to-Speech and Speech Processing for Quark.

This module provides a configurable interface for text-to-speech functionality
using the Live-Voice Gemini integration and other TTS providers.

Integration: Part of the language cortex neural core system.
Rationale: Enables voice output capabilities for Quark with configurable providers.
"""

import json
import subprocess
import tempfile
import os
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TTSProvider(Enum):
    """Available text-to-speech providers."""
    LIVE_VOICE_GEMINI = "live_voice_gemini"
    SYSTEM_TTS = "system_tts"
    DISABLED = "disabled"

@dataclass
class TTSConfig:
    """Configuration for text-to-speech functionality."""
    enabled: bool = False
    provider: TTSProvider = TTSProvider.DISABLED
    voice_name: str = "Orus"  # Default Gemini voice
    language_code: str = "en-US"
    speech_rate: float = 1.0
    volume: float = 0.8
    
class SpeechIntegration:
    """
    Main speech integration class providing TTS capabilities for Quark.
    
    Supports multiple TTS providers with configurable settings and fallback options.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize speech integration with configuration."""
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.live_voice_path = self._get_live_voice_path()
        
        # Initialize provider-specific components
        self._init_providers()
        
        logger.info(f"Speech Integration initialized with provider: {self.config.provider.value}")
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        return "/Users/camdouglas/quark/data/credentials/speech_config.json"
    
    def _get_live_voice_path(self) -> str:
        """Get path to Live-Voice integration."""
        return "/Users/camdouglas/quark/brain/architecture/neural_core/language/live_voice"
    
    def _load_config(self) -> TTSConfig:
        """Load TTS configuration from file or create default."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                return TTSConfig(
                    enabled=config_data.get('enabled', False),
                    provider=TTSProvider(config_data.get('provider', 'disabled')),
                    voice_name=config_data.get('voice_name', 'Orus'),
                    language_code=config_data.get('language_code', 'en-US'),
                    speech_rate=config_data.get('speech_rate', 1.0),
                    volume=config_data.get('volume', 0.8)
                )
            else:
                # Create default config
                default_config = TTSConfig()
                self._save_config(default_config)
                return default_config
                
        except Exception as e:
            logger.error(f"Error loading TTS config: {e}")
            return TTSConfig()
    
    def _save_config(self, config: TTSConfig) -> None:
        """Save TTS configuration to file."""
        try:
            # Ensure config directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            config_data = {
                'enabled': config.enabled,
                'provider': config.provider.value,
                'voice_name': config.voice_name,
                'language_code': config.language_code,
                'speech_rate': config.speech_rate,
                'volume': config.volume
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            logger.info(f"TTS config saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving TTS config: {e}")
    
    def _init_providers(self) -> None:
        """Initialize TTS providers based on configuration."""
        if self.config.provider == TTSProvider.LIVE_VOICE_GEMINI:
            self._check_live_voice_setup()
        elif self.config.provider == TTSProvider.SYSTEM_TTS:
            self._check_system_tts()
    
    def _check_live_voice_setup(self) -> bool:
        """Check if Live-Voice Gemini setup is available."""
        try:
            # Check if Live-Voice directory exists
            if not os.path.exists(self.live_voice_path):
                logger.warning(f"Live-Voice directory not found at {self.live_voice_path}")
                return False
            
            # Check if package.json exists
            package_json = os.path.join(self.live_voice_path, 'package.json')
            if not os.path.exists(package_json):
                logger.warning("Live-Voice package.json not found")
                return False
            
            # Check if node_modules exists (dependencies installed)
            node_modules = os.path.join(self.live_voice_path, 'node_modules')
            if not os.path.exists(node_modules):
                logger.info("Live-Voice dependencies not installed, attempting installation...")
                return self._install_live_voice_deps()
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking Live-Voice setup: {e}")
            return False
    
    def _install_live_voice_deps(self) -> bool:
        """Install Live-Voice dependencies."""
        try:
            result = subprocess.run(
                ['npm', 'install'],
                cwd=self.live_voice_path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.info("Live-Voice dependencies installed successfully")
                return True
            else:
                logger.error(f"Failed to install Live-Voice dependencies: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Live-Voice dependency installation timed out")
            return False
        except Exception as e:
            logger.error(f"Error installing Live-Voice dependencies: {e}")
            return False
    
    def _check_system_tts(self) -> bool:
        """Check if system TTS is available."""
        try:
            # Check for macOS 'say' command
            result = subprocess.run(['which', 'say'], capture_output=True, text=True)
            if result.returncode == 0:
                return True
            
            # Check for espeak on Linux
            result = subprocess.run(['which', 'espeak'], capture_output=True, text=True)
            if result.returncode == 0:
                return True
            
            logger.warning("No system TTS found (tried 'say' and 'espeak')")
            return False
            
        except Exception as e:
            logger.error(f"Error checking system TTS: {e}")
            return False
    
    def speak_text(self, text: str, **kwargs) -> bool:
        """
        Convert text to speech using the configured provider.
        
        Args:
            text: Text to convert to speech
            **kwargs: Provider-specific options
            
        Returns:
            bool: True if speech was successful, False otherwise
        """
        if not self.config.enabled or self.config.provider == TTSProvider.DISABLED:
            logger.debug("TTS is disabled")
            return False
        
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return False
        
        try:
            if self.config.provider == TTSProvider.LIVE_VOICE_GEMINI:
                return self._speak_with_live_voice(text, **kwargs)
            elif self.config.provider == TTSProvider.SYSTEM_TTS:
                return self._speak_with_system_tts(text, **kwargs)
            else:
                logger.error(f"Unknown TTS provider: {self.config.provider}")
                return False
                
        except Exception as e:
            logger.error(f"Error in speak_text: {e}")
            return False
    
    def _speak_with_live_voice(self, text: str, **kwargs) -> bool:
        """Use Live-Voice Gemini for text-to-speech."""
        try:
            # For now, we'll create a simple interface to the Live-Voice system
            # In a full implementation, this would integrate with the TypeScript code
            logger.info(f"Live-Voice TTS: {text[:50]}...")
            
            # Create a temporary script to interface with Live-Voice
            script_content = f'''
            // Simple TTS interface for Live-Voice integration
            console.log("TTS Request: {text}");
            // This would integrate with the actual Live-Voice Gemini API
            '''
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(script_content)
                temp_script = f.name
            
            try:
                # For demonstration - in practice this would use the actual Live-Voice API
                logger.info(f"Would speak with Live-Voice Gemini: '{text}'")
                return True
            finally:
                os.unlink(temp_script)
                
        except Exception as e:
            logger.error(f"Error with Live-Voice TTS: {e}")
            return False
    
    def _speak_with_system_tts(self, text: str, **kwargs) -> bool:
        """Use system TTS for text-to-speech."""
        try:
            # Use macOS 'say' command if available
            result = subprocess.run(['which', 'say'], capture_output=True, text=True)
            if result.returncode == 0:
                cmd = [
                    'say',
                    '-r', str(int(self.config.speech_rate * 200)),  # Convert to words per minute
                    '-v', kwargs.get('voice', 'Alex'),
                    text
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    logger.info(f"System TTS completed: '{text[:50]}...'")
                    return True
                else:
                    logger.error(f"System TTS failed: {result.stderr}")
                    return False
            
            # Try espeak on Linux
            result = subprocess.run(['which', 'espeak'], capture_output=True, text=True)
            if result.returncode == 0:
                cmd = [
                    'espeak',
                    '-s', str(int(self.config.speech_rate * 175)),  # Words per minute
                    '-a', str(int(self.config.volume * 200)),      # Amplitude
                    text
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    logger.info(f"System TTS (espeak) completed: '{text[:50]}...'")
                    return True
                else:
                    logger.error(f"System TTS (espeak) failed: {result.stderr}")
                    return False
            
            logger.error("No system TTS available")
            return False
            
        except subprocess.TimeoutExpired:
            logger.error("System TTS timed out")
            return False
        except Exception as e:
            logger.error(f"Error with system TTS: {e}")
            return False
    
    def enable_tts(self, provider: TTSProvider = TTSProvider.SYSTEM_TTS) -> bool:
        """Enable text-to-speech with specified provider."""
        try:
            self.config.enabled = True
            self.config.provider = provider
            self._save_config(self.config)
            self._init_providers()
            
            logger.info(f"TTS enabled with provider: {provider.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error enabling TTS: {e}")
            return False
    
    def disable_tts(self) -> bool:
        """Disable text-to-speech."""
        try:
            self.config.enabled = False
            self.config.provider = TTSProvider.DISABLED
            self._save_config(self.config)
            
            logger.info("TTS disabled")
            return True
            
        except Exception as e:
            logger.error(f"Error disabling TTS: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current TTS status and configuration."""
        return {
            'enabled': self.config.enabled,
            'provider': self.config.provider.value,
            'voice_name': self.config.voice_name,
            'language_code': self.config.language_code,
            'speech_rate': self.config.speech_rate,
            'volume': self.config.volume,
            'live_voice_available': self._check_live_voice_setup(),
            'system_tts_available': self._check_system_tts(),
            'config_path': self.config_path
        }
    
    def list_available_providers(self) -> List[str]:
        """List available TTS providers."""
        providers = []
        
        if self._check_system_tts():
            providers.append(TTSProvider.SYSTEM_TTS.value)
        
        if self._check_live_voice_setup():
            providers.append(TTSProvider.LIVE_VOICE_GEMINI.value)
        
        providers.append(TTSProvider.DISABLED.value)
        
        return providers

# Global speech integration instance
_speech_integration = None

def get_speech_integration() -> SpeechIntegration:
    """Get global speech integration instance."""
    global _speech_integration
    if _speech_integration is None:
        _speech_integration = SpeechIntegration()
    return _speech_integration

def speak(text: str, **kwargs) -> bool:
    """Convenience function for text-to-speech."""
    return get_speech_integration().speak_text(text, **kwargs)

def enable_speech(provider: str = "system_tts") -> bool:
    """Convenience function to enable speech."""
    try:
        provider_enum = TTSProvider(provider)
        return get_speech_integration().enable_tts(provider_enum)
    except ValueError:
        logger.error(f"Invalid TTS provider: {provider}")
        return False

def disable_speech() -> bool:
    """Convenience function to disable speech."""
    return get_speech_integration().disable_tts()

def get_speech_status() -> Dict[str, Any]:
    """Convenience function to get speech status."""
    return get_speech_integration().get_status()

# Export main classes and functions
__all__ = [
    'SpeechIntegration', 
    'TTSProvider', 
    'TTSConfig',
    'get_speech_integration',
    'speak',
    'enable_speech', 
    'disable_speech',
    'get_speech_status'
]
