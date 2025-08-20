/**
 * Voice Consciousness Integration Tests
 * 
 * Purpose: Test voice generation, conscious agent integration, and text-to-speech
 * Inputs: Mock audio data, test thoughts, voice preferences
 * Outputs: Test results, coverage reports, performance metrics
 * Seeds: Deterministic test data with configurable randomness
 * Dependencies: Jest, VoiceConsciousnessIntegration, Mock Audio APIs
 */

import { VoiceConsciousnessIntegration, ConsciousVoiceState, VoicePreferences } from '../../src/voice/VoiceConsciousnessIntegration';
import { ConsciousAgentVoiceBridge, ConsciousAgentThought, VoiceBridgePresets } from '../../src/voice/ConsciousAgentVoiceBridge';

// Mock Web Audio API
class MockAudioContext {
  sampleRate = 44100;
  destination = { connect: jest.fn() };
  createAnalyser = jest.fn(() => ({
    fftSize: 2048,
    frequencyBinCount: 1024,
    smoothingTimeConstant: 0.8,
    connect: jest.fn(),
    getFloatFrequencyData: jest.fn(),
    getFloatTimeDomainData: jest.fn()
  }));
  createMediaStreamSource = jest.fn(() => ({
    connect: jest.fn()
  }));
  createBufferSource = jest.fn(() => ({
    buffer: null,
    connect: jest.fn(),
    start: jest.fn(),
    stop: jest.fn(),
    onended: null
  }));
  close = jest.fn();
  state = 'running';
}

// Mock MediaDevices
Object.defineProperty(global.navigator, 'mediaDevices', {
  value: {
    getUserMedia: jest.fn(() => Promise.resolve({} as MediaStream))
  },
  writable: true
});

// Mock AudioContext
Object.defineProperty(global, 'AudioContext', {
  value: MockAudioContext,
  writable: true
});

Object.defineProperty(global, 'webkitAudioContext', {
  value: MockAudioContext,
  writable: true
});

describe('Voice Consciousness Integration', () => {
  let voiceIntegration: VoiceConsciousnessIntegration;
  let mockAudioContext: MockAudioContext;

  beforeEach(() => {
    mockAudioContext = new MockAudioContext();
    voiceIntegration = new VoiceConsciousnessIntegration();
  });

  afterEach(() => {
    voiceIntegration.dispose();
  });

  describe('Initialization', () => {
    test('should initialize voice system successfully', async () => {
      // Wait for initialization
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const state = voiceIntegration.getConsciousVoiceState();
      expect(state.currentVoice).not.toBe('default');
      expect(state.voicePreferences.autoSpeak).toBe(true);
    });

    test('should create default voice clone', async () => {
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const state = voiceIntegration.getConsciousVoiceState();
      expect(state.currentVoice).toBeDefined();
      expect(state.voicePreferences.preferredVoice).toBeDefined();
    });
  });

  describe('Voice Decision Making', () => {
    test('should decide to speak for high-priority thoughts', async () => {
      const thought = "This is a critical breakthrough that must be shared!";
      const confidence = 0.9;
      const emotionalContext = { excitement: 0.8, confidence: 0.9 };

      const decision = await voiceIntegration.consciousAgentSpeakDecision(thought, confidence, emotionalContext);
      
      expect(decision.shouldSpeak).toBe(true);
      expect(decision.confidence).toBeGreaterThan(0.7);
      expect(decision.reasoning).toContain('High priority thought');
    });

    test('should not speak for low-priority thoughts', async () => {
      const thought = "Routine background processing continues normally.";
      const confidence = 0.5;
      const emotionalContext = { happiness: 0.6 };

      const decision = await voiceIntegration.consciousAgentSpeakDecision(thought, confidence, emotionalContext);
      
      expect(decision.shouldSpeak).toBe(false);
      expect(decision.reasoning).toContain('does not meet speaking criteria');
    });

    test('should respect time between speech constraints', async () => {
      // First thought should speak
      const thought1 = "Important discovery!";
      const decision1 = await voiceIntegration.consciousAgentSpeakDecision(thought1, 0.9, {});
      expect(decision1.shouldSpeak).toBe(true);

      // Second thought too soon should not speak
      const thought2 = "Another important thought!";
      const decision2 = await voiceIntegration.consciousAgentSpeakDecision(thought2, 0.9, {});
      expect(decision2.shouldSpeak).toBe(false);
      expect(decision2.reasoning).toContain('Too soon since last speech');
    });
  });

  describe('Emotional State Management', () => {
    test('should update emotional state based on context', async () => {
      const initialState = voiceIntegration.getConsciousVoiceState();
      const initialHappiness = initialState.emotionalState.happiness;

      // Update with high happiness context
      await voiceIntegration.consciousAgentSpeakDecision("Happy thought!", 0.8, { happiness: 0.9 });
      
      const newState = voiceIntegration.getConsciousVoiceState();
      expect(newState.emotionalState.happiness).toBe(0.9);
    });

    test('should apply emotional decay over time', async () => {
      // Set high excitement
      await voiceIntegration.consciousAgentSpeakDecision("Exciting!", 0.9, { excitement: 0.9 });
      
      const state1 = voiceIntegration.getConsciousVoiceState();
      expect(state1.emotionalState.excitement).toBe(0.9);

      // Wait and check decay
      await new Promise(resolve => setTimeout(resolve, 100));
      const state2 = voiceIntegration.getConsciousVoiceState();
      expect(state2.emotionalState.excitement).toBeLessThan(0.9);
    });
  });

  describe('User Text Input', () => {
    test('should handle user text input and generate response', async () => {
      const userInput = "What do you think about consciousness?";
      
      await voiceIntegration.handleUserTextInput(userInput);
      
      const state = voiceIntegration.getConsciousVoiceState();
      const conversation = state.conversationHistory;
      
      expect(conversation.length).toBeGreaterThan(0);
      expect(conversation[0].speaker).toBe('user');
      expect(conversation[0].content).toBe(userInput);
    });

    test('should generate appropriate conscious responses', async () => {
      const questionInput = "How does the brain work?";
      const response = await (voiceIntegration as any).generateConsciousResponse(questionInput);
      
      expect(response).toContain('fascinating question');
      expect(response.length).toBeGreaterThan(20);
    });
  });

  describe('Voice Preferences', () => {
    test('should update voice preferences', () => {
      const newPreferences: Partial<VoicePreferences> = {
        speakingStyle: 'enthusiastic',
        speechRate: 1.5,
        autoSpeak: false
      };

      voiceIntegration.updateVoicePreferences(newPreferences);
      
      const state = voiceIntegration.getConsciousVoiceState();
      expect(state.voicePreferences.speakingStyle).toBe('enthusiastic');
      expect(state.voicePreferences.speechRate).toBe(1.5);
      expect(state.voicePreferences.autoSpeak).toBe(false);
    });

    test('should respect auto-speak setting', async () => {
      // Disable auto-speak
      voiceIntegration.updateVoicePreferences({ autoSpeak: false });
      
      const thought = "This is very important!";
      const decision = await voiceIntegration.consciousAgentSpeakDecision(thought, 0.9, {});
      
      expect(decision.shouldSpeak).toBe(false);
    });
  });

  describe('Conversation History', () => {
    test('should maintain conversation history', async () => {
      await voiceIntegration.handleUserTextInput("Hello!");
      await voiceIntegration.handleUserTextInput("How are you?");
      
      const state = voiceIntegration.getConsciousVoiceState();
      expect(state.conversationHistory.length).toBeGreaterThan(1);
    });

    test('should limit conversation history size', async () => {
      // Add many conversations
      for (let i = 0; i < 150; i++) {
        await voiceIntegration.handleUserTextInput(`Message ${i}`);
      }
      
      const state = voiceIntegration.getConsciousVoiceState();
      expect(state.conversationHistory.length).toBeLessThanOrEqual(100);
    });
  });
});

describe('Conscious Agent Voice Bridge', () => {
  let voiceBridge: ConsciousAgentVoiceBridge;

  beforeEach(() => {
    voiceBridge = new ConsciousAgentVoiceBridge();
  });

  afterEach(() => {
    voiceBridge.dispose();
  });

  describe('Thought Processing', () => {
    test('should process conscious thoughts', async () => {
      const thought: ConsciousAgentThought = {
        id: 'test_1',
        content: 'This is a test thought',
        confidence: 0.8,
        priority: 'high',
        category: 'insight',
        timestamp: Date.now()
      };

      await voiceBridge.processConsciousThought(thought);
      
      const stats = voiceBridge.getProcessingStats();
      expect(stats.totalThoughtsProcessed).toBeGreaterThan(0);
    });

    test('should respect rate limiting', async () => {
      // Add many thoughts quickly
      for (let i = 0; i < 10; i++) {
        const thought: ConsciousAgentThought = {
          id: `test_${i}`,
          content: `Thought ${i}`,
          confidence: 0.8,
          priority: 'medium',
          category: 'observation',
          timestamp: Date.now()
        };
        
        await voiceBridge.processConsciousThought(thought);
      }
      
      const stats = voiceBridge.getProcessingStats();
      expect(stats.thoughtsThisMinute).toBeLessThanOrEqual(5); // Default max
    });

    test('should enhance emotional context based on thought properties', async () => {
      const criticalThought: ConsciousAgentThought = {
        id: 'critical_1',
        content: 'Critical system failure detected!',
        confidence: 0.95,
        priority: 'critical',
        category: 'decision',
        timestamp: Date.now()
      };

      await voiceBridge.processConsciousThought(criticalThought);
      
      // Wait for processing
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const voiceState = voiceBridge.getVoiceState();
      expect(voiceState.emotionalState.urgency).toBeGreaterThan(0.8);
      expect(voiceState.emotionalState.confidence).toBeGreaterThan(0.8);
    });
  });

  describe('Configuration Management', () => {
    test('should apply different presets', () => {
      // Test quiet preset
      voiceBridge.updateConfig(VoiceBridgePresets.quiet);
      const config = voiceBridge.getConfig();
      
      expect(config.speakThreshold).toBe(0.9);
      expect(config.maxThoughtsPerMinute).toBe(2);
      expect(config.emotionalSensitivity).toBe(0.5);
    });

    test('should update configuration dynamically', () => {
      voiceBridge.updateConfig({ speakThreshold: 0.5 });
      
      const config = voiceBridge.getConfig();
      expect(config.speakThreshold).toBe(0.5);
    });
  });

  describe('Force Speaking', () => {
    test('should force speak when requested', async () => {
      const content = "I am being forced to speak this thought.";
      
      await voiceBridge.forceSpeakThought(content, 0.9);
      
      // Wait for processing
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const stats = voiceBridge.getProcessingStats();
      expect(stats.totalThoughtsProcessed).toBeGreaterThan(0);
    });
  });

  describe('User Input Handling', () => {
    test('should handle user input through bridge', async () => {
      const userInput = "Tell me about consciousness";
      
      await voiceBridge.handleUserInput(userInput);
      
      const voiceState = voiceBridge.getVoiceState();
      const conversation = voiceState.conversationHistory;
      
      expect(conversation.length).toBeGreaterThan(0);
      expect(conversation[0].speaker).toBe('user');
    });
  });

  describe('Statistics and Monitoring', () => {
    test('should provide processing statistics', () => {
      const stats = voiceBridge.getProcessingStats();
      
      expect(stats).toHaveProperty('thoughtsThisMinute');
      expect(stats).toHaveProperty('totalThoughtsProcessed');
      expect(stats).toHaveProperty('queueLength');
      expect(stats).toHaveProperty('isProcessing');
    });

    test('should track queue length', async () => {
      // Add thoughts faster than processing
      for (let i = 0; i < 5; i++) {
        const thought: ConsciousAgentThought = {
          id: `fast_${i}`,
          content: `Fast thought ${i}`,
          confidence: 0.8,
          priority: 'medium',
          category: 'observation',
          timestamp: Date.now()
        };
        
        await voiceBridge.processConsciousThought(thought);
      }
      
      const stats = voiceBridge.getProcessingStats();
      expect(stats.queueLength).toBeGreaterThan(0);
    });
  });

  describe('Resource Management', () => {
    test('should clear thought queue', async () => {
      // Add some thoughts
      for (let i = 0; i < 3; i++) {
        const thought: ConsciousAgentThought = {
          id: `clear_${i}`,
          content: `Thought to clear ${i}`,
          confidence: 0.8,
          priority: 'medium',
          category: 'observation',
          timestamp: Date.now()
        };
        
        await voiceBridge.processConsciousThought(thought);
      }
      
      voiceBridge.clearThoughtQueue();
      
      const stats = voiceBridge.getProcessingStats();
      expect(stats.queueLength).toBe(0);
    });

    test('should stop speaking when requested', () => {
      voiceBridge.stopSpeaking();
      
      const voiceState = voiceBridge.getVoiceState();
      expect(voiceState.isSpeaking).toBe(false);
    });
  });
});

describe('Voice System Integration', () => {
  test('should integrate voice processor with consciousness', async () => {
    const voiceBridge = new ConsciousAgentVoiceBridge();
    
    // Test full integration
    const thought: ConsciousAgentThought = {
      id: 'integration_test',
      content: 'Testing full voice consciousness integration',
      confidence: 0.9,
      priority: 'high',
      category: 'insight',
      timestamp: Date.now(),
      emotionalContext: { excitement: 0.8, confidence: 0.9 }
    };

    await voiceBridge.processConsciousThought(thought);
    
    // Wait for processing
    await new Promise(resolve => setTimeout(resolve, 200));
    
    const voiceState = voiceBridge.getVoiceState();
    expect(voiceState.conversationHistory.length).toBeGreaterThan(0);
    
    voiceBridge.dispose();
  });
});

// Performance tests
describe('Voice System Performance', () => {
  test('should handle high-frequency thoughts efficiently', async () => {
    const voiceBridge = new ConsciousAgentVoiceBridge({
      maxThoughtsPerMinute: 20
    });
    
    const startTime = Date.now();
    
    // Process many thoughts
    for (let i = 0; i < 20; i++) {
      const thought: ConsciousAgentThought = {
        id: `perf_${i}`,
        content: `Performance test thought ${i}`,
        confidence: 0.8,
        priority: 'medium',
        category: 'observation',
        timestamp: Date.now()
      };
      
      await voiceBridge.processConsciousThought(thought);
    }
    
    const endTime = Date.now();
    const processingTime = endTime - startTime;
    
    // Should complete within reasonable time
    expect(processingTime).toBeLessThan(5000); // 5 seconds
    
    voiceBridge.dispose();
  });
});

// Error handling tests
describe('Voice System Error Handling', () => {
  test('should handle audio context failures gracefully', async () => {
    // Mock audio context failure
    const originalAudioContext = global.AudioContext;
    global.AudioContext = jest.fn(() => {
      throw new Error('Audio context creation failed');
    }) as any;

    // Should not crash
    expect(() => {
      new VoiceConsciousnessIntegration();
    }).not.toThrow();

    // Restore original
    global.AudioContext = originalAudioContext;
  });

  test('should handle microphone access failures', async () => {
    // Mock microphone access failure
    const originalGetUserMedia = navigator.mediaDevices.getUserMedia;
    navigator.mediaDevices.getUserMedia = jest.fn(() => 
      Promise.reject(new Error('Microphone access denied'))
    );

    const voiceIntegration = new VoiceConsciousnessIntegration();
    
    // Should handle error gracefully
    await expect(voiceIntegration.startRealTimeAnalysis()).rejects.toThrow('Microphone access denied');
    
    // Restore original
    navigator.mediaDevices.getUserMedia = originalGetUserMedia;
    
    voiceIntegration.dispose();
  });
});
