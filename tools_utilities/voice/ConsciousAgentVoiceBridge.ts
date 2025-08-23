/**
 * Conscious Agent Voice Bridge
 * 
 * Purpose: Bridge between existing conscious agent and voice generation system
 * Inputs: Conscious agent thoughts, state changes, user interactions
 * Outputs: Voice synthesis decisions, spoken responses, voice state updates
 * Seeds: Integration with existing conscious agent architecture
 * Dependencies: VoiceConsciousnessIntegration, Conscious Agent System
 */

import { VoiceConsciousnessIntegration, ConsciousVoiceDecision, EmotionalState } from './VoiceConsciousnessIntegration';

export interface ConsciousAgentThought {
  id: string;
  content: string;
  confidence: number;
  priority: 'low' | 'medium' | 'high' | 'critical';
  category: 'reflection' | 'question' | 'insight' | 'observation' | 'decision';
  timestamp: number;
  emotionalContext?: Partial<EmotionalState>;
}

export interface VoiceIntegrationConfig {
  enableAutoSpeak: boolean;
  speakThreshold: number;
  maxThoughtsPerMinute: number;
  voicePreference: string;
  emotionalSensitivity: number;
}

export class ConsciousAgentVoiceBridge {
  private voiceIntegration: VoiceConsciousnessIntegration;
  private config: VoiceIntegrationConfig;
  private thoughtQueue: ConsciousAgentThought[] = [];
  private lastThoughtTime: number = 0;
  private isProcessing: boolean = false;
  private thoughtCount: number = 0;
  private lastMinuteReset: number = Date.now();

  constructor(config?: Partial<VoiceIntegrationConfig>) {
    this.voiceIntegration = new VoiceConsciousnessIntegration();
    
    this.config = {
      enableAutoSpeak: true,
      speakThreshold: 0.7,
      maxThoughtsPerMinute: 5,
      voicePreference: 'default',
      emotionalSensitivity: 0.8,
      ...config
    };

    this.initializeBridge();
  }

  /**
   * Initialize the voice bridge
   */
  private async initializeBridge(): Promise<void> {
    try {
      // Wait for voice integration to be ready
      let attempts = 0;
      const maxAttempts = 50; // 5 seconds with 100ms intervals
      
      while (attempts < maxAttempts) {
        const state = this.voiceIntegration.getConsciousVoiceState();
        if (state.currentVoice !== 'default') {
          break;
        }
        await new Promise(resolve => setTimeout(resolve, 100));
        attempts++;
      }

      console.log('Conscious Agent Voice Bridge initialized');
    } catch (error) {
      console.error('Failed to initialize voice bridge:', error);
    }
  }

  /**
   * Process a thought from the conscious agent
   */
  async processConsciousThought(thought: ConsciousAgentThought): Promise<void> {
    // Check rate limiting
    if (!this.shouldProcessThought(thought)) {
      return;
    }

    // Add to queue
    this.thoughtQueue.push(thought);
    this.thoughtCount++;
    this.lastThoughtTime = Date.now();

    // Process queue if not already processing
    if (!this.isProcessing) {
      this.processThoughtQueue();
    }
  }

  /**
   * Check if thought should be processed based on rate limiting
   */
  private shouldProcessThought(thought: ConsciousAgentThought): boolean {
    const now = Date.now();
    
    // Reset counter every minute
    if (now - this.lastMinuteReset > 60000) {
      this.thoughtCount = 0;
      this.lastMinuteReset = now;
    }

    // Check rate limit
    if (this.thoughtCount >= this.config.maxThoughtsPerMinute) {
      return false;
    }

    // Check minimum time between thoughts
    if (now - this.lastThoughtTime < 1000) { // 1 second minimum
      return false;
    }

    return true;
  }

  /**
   * Process the thought queue
   */
  private async processThoughtQueue(): Promise<void> {
    if (this.isProcessing || this.thoughtQueue.length === 0) {
      return;
    }

    this.isProcessing = true;

    try {
      while (this.thoughtQueue.length > 0) {
        const thought = this.thoughtQueue.shift();
        if (!thought) continue;

        await this.processSingleThought(thought);
        
        // Small delay between thoughts
        await new Promise(resolve => setTimeout(resolve, 200));
      }
    } catch (error) {
      console.error('Error processing thought queue:', error);
    } finally {
      this.isProcessing = false;
    }
  }

  /**
   * Process a single thought
   */
  private async processSingleThought(thought: ConsciousAgentThought): Promise<void> {
    try {
      // Enhance emotional context based on thought category and priority
      const enhancedEmotionalContext = this.enhanceEmotionalContext(thought);
      
      // Get voice decision
      const decision = await this.voiceIntegration.consciousAgentSpeakDecision(
        thought.content,
        thought.confidence,
        enhancedEmotionalContext
      );

      // Log the decision
      console.log(`Thought "${thought.content.substring(0, 50)}..." - Decision: ${decision.shouldSpeak ? 'SPEAK' : 'SILENT'} (${decision.confidence.toFixed(2)})`);

      // Execute speaking if decision is positive
      if (decision.shouldSpeak) {
        await this.voiceIntegration.executeSpeakingDecision(decision, thought.content);
      }

      // Update emotional state based on thought processing
      this.updateEmotionalStateFromThought(thought, decision);

    } catch (error) {
      console.error('Failed to process thought:', thought.id, error);
    }
  }

  /**
   * Enhance emotional context based on thought properties
   */
  private enhanceEmotionalContext(thought: ConsciousAgentThought): Partial<EmotionalState> {
    const baseContext = thought.emotionalContext || {};
    const enhanced: Partial<EmotionalState> = { ...baseContext };

    // Enhance based on priority
    switch (thought.priority) {
      case 'critical':
        enhanced.urgency = Math.max(enhanced.urgency || 0, 0.9);
        enhanced.confidence = Math.max(enhanced.confidence || 0, 0.8);
        break;
      case 'high':
        enhanced.urgency = Math.max(enhanced.urgency || 0, 0.7);
        enhanced.curiosity = Math.max(enhanced.curiosity || 0, 0.8);
        break;
      case 'medium':
        enhanced.curiosity = Math.max(enhanced.curiosity || 0, 0.6);
        break;
      case 'low':
        enhanced.happiness = Math.max(enhanced.happiness || 0, 0.5);
        break;
    }

    // Enhance based on category
    switch (thought.category) {
      case 'question':
        enhanced.curiosity = Math.max(enhanced.curiosity || 0, 0.8);
        break;
      case 'insight':
        enhanced.excitement = Math.max(enhanced.excitement || 0, 0.7);
        enhanced.confidence = Math.max(enhanced.confidence || 0, 0.8);
        break;
      case 'reflection':
        enhanced.happiness = Math.max(enhanced.happiness || 0, 0.6);
        break;
      case 'decision':
        enhanced.confidence = Math.max(enhanced.confidence || 0, 0.7);
        break;
    }

    return enhanced;
  }

  /**
   * Update emotional state based on thought processing
   */
  private updateEmotionalStateFromThought(thought: ConsciousAgentThought, decision: ConsciousVoiceDecision): void {
    const currentState = this.voiceIntegration.getConsciousVoiceState();
    const emotionalState = currentState.emotionalState;

    // Adjust emotions based on whether the thought was spoken
    if (decision.shouldSpeak) {
      // Speaking increases confidence and happiness
      emotionalState.confidence = Math.min(1, emotionalState.confidence + 0.1);
      emotionalState.happiness = Math.min(1, emotionalState.happiness + 0.05);
      
      // High confidence thoughts increase excitement
      if (thought.confidence > 0.8) {
        emotionalState.excitement = Math.min(1, emotionalState.excitement + 0.1);
      }
    } else {
      // Silent thoughts might increase curiosity
      if (thought.category === 'question') {
        emotionalState.curiosity = Math.min(1, emotionalState.curiosity + 0.05);
      }
    }

    // Decay emotions over time
    this.applyEmotionalDecay(emotionalState);
  }

  /**
   * Apply emotional decay to prevent emotions from staying too high
   */
  private applyEmotionalDecay(emotionalState: EmotionalState): void {
    const decayRate = 0.02;
    
    emotionalState.excitement = Math.max(0.1, emotionalState.excitement - decayRate);
    emotionalState.urgency = Math.max(0.1, emotionalState.urgency - decayRate);
    emotionalState.curiosity = Math.max(0.2, emotionalState.curiosity - decayRate * 0.5);
  }

  /**
   * Force the conscious agent to speak a specific thought
   */
  async forceSpeakThought(content: string, confidence: number = 0.9): Promise<void> {
    const thought: ConsciousAgentThought = {
      id: `forced_${Date.now()}`,
      content,
      confidence,
      priority: 'high',
      category: 'reflection',
      timestamp: Date.now(),
      emotionalContext: {
        confidence: confidence,
        urgency: 0.8
      }
    };

    await this.processSingleThought(thought);
  }

  /**
   * Handle user text input through the bridge
   */
  async handleUserInput(text: string): Promise<void> {
    await this.voiceIntegration.handleUserTextInput(text);
  }

  /**
   * Get current voice state
   */
  getVoiceState() {
    return this.voiceIntegration.getConsciousVoiceState();
  }

  /**
   * Update voice configuration
   */
  updateConfig(newConfig: Partial<VoiceIntegrationConfig>): void {
    this.config = { ...this.config, ...newConfig };
    
    // Update voice integration preferences
    this.voiceIntegration.updateVoicePreferences({
      speakThreshold: this.config.speakThreshold,
      autoSpeak: this.config.enableAutoSpeak
    });
  }

  /**
   * Get current configuration
   */
  getConfig(): VoiceIntegrationConfig {
    return { ...this.config };
  }

  /**
   * Get thought processing statistics
   */
  getProcessingStats() {
    const now = Date.now();
    const timeSinceReset = now - this.lastMinuteReset;
    const thoughtsThisMinute = timeSinceReset < 60000 ? this.thoughtCount : 0;
    
    return {
      thoughtsThisMinute,
      totalThoughtsProcessed: this.thoughtCount,
      queueLength: this.thoughtQueue.length,
      isProcessing: this.isProcessing,
      lastThoughtTime: this.lastThoughtTime,
      timeSinceLastThought: now - this.lastThoughtTime
    };
  }

  /**
   * Clear the thought queue
   */
  clearThoughtQueue(): void {
    this.thoughtQueue = [];
    console.log('Thought queue cleared');
  }

  /**
   * Stop all current speech
   */
  stopSpeaking(): void {
    this.voiceIntegration.stopSpeaking();
  }

  /**
   * Cleanup resources
   */
  dispose(): void {
    this.voiceIntegration.dispose();
    this.thoughtQueue = [];
    this.isProcessing = false;
  }
}

/**
 * Factory function to create a voice bridge with default configuration
 */
export function createConsciousAgentVoiceBridge(
  config?: Partial<VoiceIntegrationConfig>
): ConsciousAgentVoiceBridge {
  return new ConsciousAgentVoiceBridge(config);
}

/**
 * Default configurations for different use cases
 */
export const VoiceBridgePresets = {
  // Quiet mode - only speaks very important thoughts
  quiet: {
    enableAutoSpeak: true,
    speakThreshold: 0.9,
    maxThoughtsPerMinute: 2,
    emotionalSensitivity: 0.5
  },
  
  // Normal mode - balanced speaking
  normal: {
    enableAutoSpeak: true,
    speakThreshold: 0.7,
    maxThoughtsPerMinute: 5,
    emotionalSensitivity: 0.8
  },
  
  // Chatty mode - speaks more frequently
  chatty: {
    enableAutoSpeak: true,
    speakThreshold: 0.5,
    maxThoughtsPerMinute: 8,
    emotionalSensitivity: 1.0
  },
  
  // Silent mode - no automatic speaking
  silent: {
    enableAutoSpeak: false,
    speakThreshold: 1.0,
    maxThoughtsPerMinute: 0,
    emotionalSensitivity: 0.0
  }
};
