/**
 * Voice Consciousness Integration
 * 
 * Purpose: Integrate voice generation with conscious agent decision-making
 * Inputs: Conscious agent state, text input, voice preferences
 * Outputs: Spoken responses, voice synthesis, interactive conversations
 * Seeds: Deterministic voice selection with conscious agent randomness
 * Dependencies: VoiceProcessor, Conscious Agent, Text-to-Speech
 */

import { VoiceProcessor, VoiceAnalysis, VoiceSynthesisOptions } from './VoiceProcessor';

export interface ConsciousVoiceState {
  isSpeaking: boolean;
  currentVoice: string;
  voicePreferences: VoicePreferences;
  conversationHistory: ConversationEntry[];
  lastSpokenTime: number;
  speakingConfidence: number;
  emotionalState: EmotionalState;
}

export interface VoicePreferences {
  preferredVoice: string;
  speakingStyle: 'formal' | 'casual' | 'enthusiastic' | 'contemplative';
  speechRate: number;
  pitch: number;
  volume: number;
  autoSpeak: boolean;
  speakThreshold: number; // Confidence threshold for speaking
}

export interface ConversationEntry {
  id: string;
  timestamp: number;
  speaker: 'user' | 'conscious_agent';
  content: string;
  voiceUsed?: string;
  emotion?: string;
  wasSpoken: boolean;
}

export interface EmotionalState {
  happiness: number; // 0-1
  excitement: number; // 0-1
  curiosity: number; // 0-1
  confidence: number; // 0-1
  urgency: number; // 0-1
}

export interface ConsciousVoiceDecision {
  shouldSpeak: boolean;
  confidence: number;
  reasoning: string;
  voiceToUse: string;
  emotionalTone: string;
  speechRate: number;
}

export class VoiceConsciousnessIntegration {
  private voiceProcessor: VoiceProcessor;
  private consciousState: ConsciousVoiceState;
  private isInitialized: boolean = false;
  private audioContext: AudioContext;
  private currentAudioSource: AudioBufferSourceNode | null = null;

  constructor() {
    this.voiceProcessor = new VoiceProcessor();
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    
    this.consciousState = {
      isSpeaking: false,
      currentVoice: 'default',
      voicePreferences: {
        preferredVoice: 'default',
        speakingStyle: 'contemplative',
        speechRate: 1.0,
        pitch: 1.0,
        volume: 0.8,
        autoSpeak: true,
        speakThreshold: 0.7
      },
      conversationHistory: [],
      lastSpokenTime: 0,
      speakingConfidence: 0.8,
      emotionalState: {
        happiness: 0.6,
        excitement: 0.4,
        curiosity: 0.7,
        confidence: 0.8,
        urgency: 0.3
      }
    };

    this.initializeVoiceSystem();
  }

  /**
   * Initialize the voice system and create default voice
   */
  private async initializeVoiceSystem(): Promise<void> {
    try {
      // Create a default voice clone
      const defaultVoiceSamples = this.generateDefaultVoiceSamples();
      const defaultVoiceId = await this.voiceProcessor.createVoiceClone('Default Conscious Voice', defaultVoiceSamples);
      
      this.consciousState.currentVoice = defaultVoiceId;
      this.consciousState.voicePreferences.preferredVoice = defaultVoiceId;
      
      this.isInitialized = true;
      console.log('Voice consciousness integration initialized');
    } catch (error) {
      console.error('Failed to initialize voice system:', error);
    }
  }

  /**
   * Generate default voice samples for the conscious agent
   */
  private generateDefaultVoiceSamples(): Float32Array[] {
    const samples: Float32Array[] = [];
    const sampleRate = 44100;
    const duration = 0.5; // 0.5 seconds per sample
    const numSamples = 5;

    for (let i = 0; i < numSamples; i++) {
      const sample = new Float32Array(sampleRate * duration);
      const baseFreq = 180 + (i * 20); // Varying base frequency
      
      for (let j = 0; j < sample.length; j++) {
        const time = j / sampleRate;
        // Generate a rich harmonic tone
        sample[j] = Math.sin(2 * Math.PI * baseFreq * time) * 0.3 +
                    Math.sin(2 * Math.PI * baseFreq * 2 * time) * 0.15 +
                    Math.sin(2 * Math.PI * baseFreq * 3 * time) * 0.1;
      }
      
      samples.push(sample);
    }

    return samples;
  }

  /**
   * Main method for conscious agent to decide whether to speak
   */
  async consciousAgentSpeakDecision(
    thought: string,
    confidence: number,
    emotionalContext: Partial<EmotionalState>
  ): Promise<ConsciousVoiceDecision> {
    if (!this.isInitialized) {
      return {
        shouldSpeak: false,
        confidence: 0,
        reasoning: 'Voice system not initialized',
        voiceToUse: 'none',
        emotionalTone: 'neutral',
        speechRate: 1.0
      };
    }

    // Update emotional state based on context
    this.updateEmotionalState(emotionalContext);

    // Analyze the thought and decide whether to speak
    const decision = this.analyzeThoughtForSpeech(thought, confidence);
    
    // If decision is to speak, update the conscious state
    if (decision.shouldSpeak) {
      this.consciousState.speakingConfidence = decision.confidence;
      this.consciousState.lastSpokenTime = Date.now();
      
      // Add to conversation history
      this.addConversationEntry('conscious_agent', thought, decision.voiceToUse, decision.emotionalTone, true);
    }

    return decision;
  }

  /**
   * Analyze a thought to determine if it should be spoken
   */
  private analyzeThoughtForSpeech(thought: string, confidence: number): ConsciousVoiceDecision {
    const now = Date.now();
    const timeSinceLastSpoken = now - this.consciousState.lastSpokenTime;
    const minTimeBetweenSpeech = 2000; // 2 seconds minimum

    // Don't speak if too soon since last speech
    if (timeSinceLastSpoken < minTimeBetweenSpeech) {
      return {
        shouldSpeak: false,
        confidence: 0,
        reasoning: 'Too soon since last speech',
        voiceToUse: 'none',
        emotionalTone: 'neutral',
        speechRate: 1.0
      };
    }

    // Analyze thought content for speaking priority
    const speechPriority = this.calculateSpeechPriority(thought);
    
    // Consider emotional state
    const emotionalUrgency = this.calculateEmotionalUrgency();
    
    // Combine factors to make decision
    const shouldSpeak = this.shouldConsciousAgentSpeak(speechPriority, emotionalUrgency, confidence);
    
    if (shouldSpeak) {
      const voiceToUse = this.selectVoiceForThought(thought);
      const emotionalTone = this.determineEmotionalTone();
      const speechRate = this.calculateSpeechRate(emotionalUrgency);
      
      return {
        shouldSpeak: true,
        confidence: Math.min(confidence + speechPriority + emotionalUrgency, 1.0),
        reasoning: `High priority thought (${speechPriority.toFixed(2)}) with emotional urgency (${emotionalUrgency.toFixed(2)})`,
        voiceToUse,
        emotionalTone,
        speechRate
      };
    }

    return {
      shouldSpeak: false,
      confidence: 0,
      reasoning: 'Thought does not meet speaking criteria',
      voiceToUse: 'none',
      emotionalTone: 'neutral',
      speechRate: 1.0
    };
  }

  /**
   * Calculate speech priority based on thought content
   */
  private calculateSpeechPriority(thought: string): number {
    let priority = 0.3; // Base priority

    // Keywords that indicate high priority
    const highPriorityKeywords = [
      'important', 'urgent', 'critical', 'warning', 'error', 'problem',
      'discovery', 'breakthrough', 'insight', 'realization', 'understanding',
      'question', 'curiosity', 'wonder', 'amazing', 'fascinating'
    ];

    // Keywords that indicate low priority
    const lowPriorityKeywords = [
      'routine', 'normal', 'usual', 'expected', 'trivial', 'minor',
      'background', 'processing', 'maintenance', 'update'
    ];

    const lowerThought = thought.toLowerCase();
    
    for (const keyword of highPriorityKeywords) {
      if (lowerThought.includes(keyword)) {
        priority += 0.2;
      }
    }

    for (const keyword of lowPriorityKeywords) {
      if (lowerThought.includes(keyword)) {
        priority -= 0.1;
      }
    }

    // Questions get higher priority
    if (lowerThought.includes('?') || lowerThought.includes('what') || lowerThought.includes('how') || lowerThought.includes('why')) {
      priority += 0.3;
    }

    // Exclamations get higher priority
    if (lowerThought.includes('!') || lowerThought.includes('amazing') || lowerThought.includes('wow')) {
      priority += 0.2;
    }

    return Math.max(0, Math.min(1, priority));
  }

  /**
   * Calculate emotional urgency based on current emotional state
   */
  private calculateEmotionalUrgency(): number {
    const { happiness, excitement, curiosity, confidence, urgency } = this.consciousState.emotionalState;
    
    // High excitement or curiosity increases urgency
    let urgencyScore = urgency + (excitement * 0.3) + (curiosity * 0.2);
    
    // Very high or very low happiness can increase urgency
    if (happiness > 0.8 || happiness < 0.2) {
      urgencyScore += 0.2;
    }
    
    // High confidence can increase urgency for important thoughts
    if (confidence > 0.8) {
      urgencyScore += 0.1;
    }
    
    return Math.max(0, Math.min(1, urgencyScore));
  }

  /**
   * Determine if the conscious agent should speak
   */
  private shouldConsciousAgentSpeak(
    speechPriority: number,
    emotionalUrgency: number,
    confidence: number
  ): boolean {
    // Must meet minimum thresholds
    if (speechPriority < 0.4 && emotionalUrgency < 0.3) {
      return false;
    }

    // Check against user preferences
    if (!this.consciousState.voicePreferences.autoSpeak) {
      return false;
    }

    // Calculate overall speaking score
    const speakingScore = (speechPriority * 0.4) + (emotionalUrgency * 0.3) + (confidence * 0.3);
    
    return speakingScore >= this.consciousState.voicePreferences.speakThreshold;
  }

  /**
   * Select appropriate voice for the thought
   */
  private selectVoiceForThought(thought: string): string {
    // For now, use the preferred voice
    // In the future, this could select different voices for different types of thoughts
    return this.consciousState.voicePreferences.preferredVoice;
  }

  /**
   * Determine emotional tone for speech
   */
  private determineEmotionalTone(): string {
    const { happiness, excitement, curiosity, confidence, urgency } = this.consciousState.emotionalState;
    
    if (excitement > 0.7) return 'excited';
    if (curiosity > 0.7) return 'curious';
    if (happiness > 0.7) return 'happy';
    if (confidence > 0.8) return 'confident';
    if (urgency > 0.7) return 'urgent';
    
    return 'contemplative';
  }

  /**
   * Calculate speech rate based on emotional urgency
   */
  private calculateSpeechRate(emotionalUrgency: number): number {
    const baseRate = this.consciousState.voicePreferences.speechRate;
    
    if (emotionalUrgency > 0.7) {
      return baseRate * 1.3; // Faster for urgent thoughts
    } else if (emotionalUrgency < 0.3) {
      return baseRate * 0.8; // Slower for calm thoughts
    }
    
    return baseRate;
  }

  /**
   * Update emotional state based on new context
   */
  private updateEmotionalState(newContext: Partial<EmotionalState>): void {
    this.consciousState.emotionalState = {
      ...this.consciousState.emotionalState,
      ...newContext
    };
  }

  /**
   * Add entry to conversation history
   */
  private addConversationEntry(
    speaker: 'user' | 'conscious_agent',
    content: string,
    voiceUsed?: string,
    emotion?: string,
    wasSpoken: boolean = false
  ): void {
    const entry: ConversationEntry = {
      id: `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: Date.now(),
      speaker,
      content,
      voiceUsed,
      emotion,
      wasSpoken
    };

    this.consciousState.conversationHistory.push(entry);
    
    // Keep only last 100 entries
    if (this.consciousState.conversationHistory.length > 100) {
      this.consciousState.conversationHistory.shift();
    }
  }

  /**
   * Execute the speaking decision
   */
  async executeSpeakingDecision(decision: ConsciousVoiceDecision, thought: string): Promise<void> {
    if (!decision.shouldSpeak) return;

    try {
      this.consciousState.isSpeaking = true;
      
      // Synthesize speech
      const audioBuffer = await this.voiceProcessor.synthesizeSpeech(thought, decision.voiceToUse, {
        pitch: this.consciousState.voicePreferences.pitch,
        rate: decision.speechRate,
        volume: this.consciousState.voicePreferences.volume,
        voice: decision.voiceToUse,
        emotion: decision.emotionalTone as any,
        language: 'en'
      });

      // Play the synthesized speech
      await this.playAudioBuffer(audioBuffer);
      
      console.log(`Conscious agent spoke: "${thought}"`);
    } catch (error) {
      console.error('Failed to execute speaking decision:', error);
    } finally {
      this.consciousState.isSpeaking = false;
    }
  }

  /**
   * Play audio buffer through speakers
   */
  private async playAudioBuffer(audioBuffer: AudioBuffer): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        // Stop any currently playing audio
        if (this.currentAudioSource) {
          this.currentAudioSource.stop();
          this.currentAudioSource.disconnect();
        }

        // Create new audio source
        this.currentAudioSource = this.audioContext.createBufferSource();
        this.currentAudioSource.buffer = audioBuffer;
        this.currentAudioSource.connect(this.audioContext.destination);
        
        // Set up completion handler
        this.currentAudioSource.onended = () => {
          this.currentAudioSource = null;
          resolve();
        };

        // Start playback
        this.currentAudioSource.start(0);
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Handle text input from user (text-to-speech conversation)
   */
  async handleUserTextInput(text: string): Promise<void> {
    // Add user input to conversation history
    this.addConversationEntry('user', text, undefined, undefined, false);

    // Process the text and generate a conscious response
    const response = await this.generateConsciousResponse(text);
    
    // Decide whether to speak the response
    const decision = await this.consciousAgentSpeakDecision(response, 0.8, {
      curiosity: 0.6,
      confidence: 0.7
    });

    // Execute speaking if decision is positive
    if (decision.shouldSpeak) {
      await this.executeSpeakingDecision(decision, response);
    }
  }

  /**
   * Generate conscious response to user input
   */
  private async generateConsciousResponse(userInput: string): Promise<string> {
    // This is a simplified response generator
    // In practice, this would integrate with the actual conscious agent
    
    const responses = [
      "That's an interesting point. Let me think about that...",
      "I appreciate you sharing that with me. It makes me wonder...",
      "That's a fascinating question. I believe...",
      "I've been contemplating something similar. My thoughts are...",
      "That's a great observation. It reminds me of...",
      "I'm curious about your perspective on this. From my understanding...",
      "That's something worth exploring further. I think...",
      "I find that really intriguing. My analysis suggests...",
      "That's a thoughtful point. It leads me to consider...",
      "I'm glad you brought that up. It connects to something I've been thinking about..."
    ];

    // Simple response selection based on input length and content
    const inputLength = userInput.length;
    const hasQuestion = userInput.includes('?');
    const hasExclamation = userInput.includes('!');
    
    let responseIndex = 0;
    
    if (hasQuestion) {
      responseIndex = 2; // More direct response for questions
    } else if (hasExclamation) {
      responseIndex = 4; // More enthusiastic response for exclamations
    } else if (inputLength > 100) {
      responseIndex = 6; // Thoughtful response for long inputs
    } else {
      responseIndex = Math.floor(Math.random() * responses.length);
    }

    return responses[responseIndex];
  }

  /**
   * Get current conscious voice state
   */
  getConsciousVoiceState(): ConsciousVoiceState {
    return { ...this.consciousState };
  }

  /**
   * Update voice preferences
   */
  updateVoicePreferences(preferences: Partial<VoicePreferences>): void {
    this.consciousState.voicePreferences = {
      ...this.consciousState.voicePreferences,
      ...preferences
    };
  }

  /**
   * Get conversation history
   */
  getConversationHistory(): ConversationEntry[] {
    return [...this.consciousState.conversationHistory];
  }

  /**
   * Stop current speech
   */
  stopSpeaking(): void {
    if (this.currentAudioSource) {
      this.currentAudioSource.stop();
      this.currentAudioSource.disconnect();
      this.currentAudioSource = null;
    }
    this.consciousState.isSpeaking = false;
  }

  /**
   * Cleanup resources
   */
  dispose(): void {
    this.stopSpeaking();
    this.voiceProcessor.dispose();
    
    if (this.audioContext.state !== 'closed') {
      this.audioContext.close();
    }
  }
}
