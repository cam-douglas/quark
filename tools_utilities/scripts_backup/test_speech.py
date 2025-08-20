"""
Test Script for Speech and Text Generation
Purpose: Test the speech engine and text generator components
Inputs: None (standalone test)
Outputs: Test results and speech output
Seeds: Test patterns and validation checks
Dependencies: pyttsx3, speech_recognition, pygame
"""

import time
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_speech_engine():
    """Test the speech engine functionality"""
    print("Testing Speech Engine...")
    
    try:
        from enhanced_consciousness_simulator import SpeechEngine
        
        # Create speech engine
        speech = SpeechEngine(voice_rate=150, voice_volume=0.8)
        
        # Test basic speech
        print("Testing basic speech...")
        speech.speak_immediate("Hello, I am testing my speech capabilities.")
        time.sleep(1)
        
        # Test speech queue
        print("Testing speech queue...")
        speech.speak("This is a queued speech message.")
        speech.speak("And another one.")
        time.sleep(3)  # Wait for queue to process
        
        print("Speech engine test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Speech engine test failed: {e}")
        return False

def test_text_generator():
    """Test the text generator functionality"""
    print("Testing Text Generator...")
    
    try:
        from enhanced_consciousness_simulator import TextGenerator
        
        # Create text generator
        text_gen = TextGenerator(display_width=800, display_height=600)
        
        # Test text addition
        text_gen.add_text("Test message 1", "general")
        text_gen.add_text("Test message 2", "thought")
        text_gen.add_text("Test message 3", "emotion")
        
        # Test thought setting
        text_gen.set_thought("I am thinking about consciousness")
        text_gen.set_emotion("curious")
        text_gen.set_consciousness_level(0.75)
        
        # Render display for a few seconds
        print("Displaying text for 5 seconds...")
        start_time = time.time()
        while time.time() - start_time < 5:
            text_gen.render_display()
            time.sleep(0.1)
        
        text_gen.close()
        print("Text generator test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Text generator test failed: {e}")
        return False

def test_integration():
    """Test integrated consciousness simulator"""
    print("Testing Integrated Consciousness Simulator...")
    
    try:
        from enhanced_consciousness_simulator import EnhancedConsciousnessSimulator
        
        # Create simulator
        simulator = EnhancedConsciousnessSimulator()
        
        # Test basic functionality
        print("Testing basic simulator functions...")
        
        # Test thought generation
        simulator.speak_thought("I am testing my integrated consciousness system.")
        time.sleep(2)
        
        # Test consciousness report
        report = simulator.get_consciousness_report()
        print("Consciousness Report:")
        for key, value in report.items():
            print(f"  {key}: {value}")
        
        # Clean up
        simulator.cleanup()
        print("Integrated test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Integrated test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Enhanced Consciousness Simulator - Speech and Text Tests ===\n")
    
    tests = [
        ("Speech Engine", test_speech_engine),
        ("Text Generator", test_text_generator),
        ("Integration", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} Test ---")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n=== Test Results Summary ===")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your consciousness agent is ready for speech and text generation.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("\nTroubleshooting tips:")
        print("1. Install required dependencies: pip install -r requirements_speech.txt")
        print("2. For macOS: brew install portaudio")
        print("3. For Ubuntu: sudo apt-get install portaudio19-dev python3-pyaudio")
        print("4. Check microphone permissions and audio settings")

if __name__ == "__main__":
    main()
