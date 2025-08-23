/**
 * Voice Interaction Interface Component
 * 
 * Purpose: Provide text-to-speech interface and voice controls for conscious agent
 * Inputs: User text input, voice preferences, conscious agent state
 * Outputs: Text input handling, voice synthesis, conversation display
 * Seeds: Component state management with React hooks
 * Dependencies: React, VoiceConsciousnessIntegration, Tailwind CSS
 */

import React, { useState, useEffect, useRef } from 'react';
import { VoiceConsciousnessIntegration, ConsciousVoiceState, VoicePreferences } from '../voice/VoiceConsciousnessIntegration';

interface VoiceInteractionInterfaceProps {
  onVoiceStateChange?: (state: ConsciousVoiceState) => void;
  onConversationUpdate?: (conversation: any[]) => void;
  className?: string;
}

export const VoiceInteractionInterface: React.FC<VoiceInteractionInterfaceProps> = ({
  onVoiceStateChange,
  onConversationUpdate,
  className = ''
}) => {
  const [voiceIntegration, setVoiceIntegration] = useState<VoiceConsciousnessIntegration | null>(null);
  const [consciousState, setConsciousState] = useState<ConsciousVoiceState | null>(null);
  const [userInput, setUserInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [showAdvancedControls, setShowAdvancedControls] = useState(false);
  const [voicePreferences, setVoicePreferences] = useState<VoicePreferences>({
    preferredVoice: 'default',
    speakingStyle: 'contemplative',
    speechRate: 1.0,
    pitch: 1.0,
    volume: 0.8,
    autoSpeak: true,
    speakThreshold: 0.7
  });

  const textInputRef = useRef<HTMLTextAreaElement>(null);
  const conversationEndRef = useRef<HTMLDivElement>(null);

  // Initialize voice integration
  useEffect(() => {
    const integration = new VoiceConsciousnessIntegration();
    setVoiceIntegration(integration);

    // Set up periodic state updates
    const interval = setInterval(() => {
      const state = integration.getConsciousVoiceState();
      setConsciousState(state);
      onVoiceStateChange?.(state);
    }, 1000);

    return () => {
      clearInterval(interval);
      integration.dispose();
    };
  }, [onVoiceStateChange]);

  // Auto-scroll conversation to bottom
  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [consciousState?.conversationHistory]);

  // Handle user text input submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userInput.trim() || !voiceIntegration || isProcessing) return;

    setIsProcessing(true);
    try {
      await voiceIntegration.handleUserTextInput(userInput.trim());
      setUserInput('');
    } catch (error) {
      console.error('Failed to process user input:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle voice preference changes
  const handleVoicePreferenceChange = (key: keyof VoicePreferences, value: any) => {
    const newPreferences = { ...voicePreferences, [key]: value };
    setVoicePreferences(newPreferences);
    
    if (voiceIntegration) {
      voiceIntegration.updateVoicePreferences(newPreferences);
    }
  };

  // Stop current speech
  const handleStopSpeaking = () => {
    voiceIntegration?.stopSpeaking();
  };

  // Force conscious agent to speak a thought
  const handleForceSpeak = async () => {
    if (!voiceIntegration) return;
    
    const thought = "I'm being asked to speak directly. Let me share what's on my mind...";
    const decision = await voiceIntegration.consciousAgentSpeakDecision(thought, 0.9, {
      confidence: 0.9,
      curiosity: 0.8
    });
    
    if (decision.shouldSpeak) {
      await voiceIntegration.executeSpeakingDecision(decision, thought);
    }
  };

  // Format timestamp for display
  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  // Get speaker display name
  const getSpeakerDisplayName = (speaker: string) => {
    return speaker === 'conscious_agent' ? '🤖 Conscious Agent' : '👤 You';
  };

  // Get emotion emoji
  const getEmotionEmoji = (emotion?: string) => {
    const emotionMap: Record<string, string> = {
      'excited': '😃',
      'curious': '🤔',
      'happy': '😊',
      'confident': '😌',
      'urgent': '⚡',
      'contemplative': '🧘',
      'neutral': '😐'
    };
    return emotionMap[emotion || 'neutral'] || '😐';
  };

  if (!consciousState) {
    return (
      <div className={`flex items-center justify-center p-8 ${className}`}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Initializing voice consciousness...</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg shadow-lg p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-800">🎤 Voice Consciousness Interface</h2>
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${consciousState.isSpeaking ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`}></div>
          <span className="text-sm text-gray-600">
            {consciousState.isSpeaking ? 'Speaking...' : 'Listening'}
          </span>
        </div>
      </div>

      {/* Voice Status */}
      <div className="bg-gray-50 rounded-lg p-4 mb-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{consciousState.speakingConfidence.toFixed(2)}</div>
            <div className="text-sm text-gray-600">Confidence</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{consciousState.emotionalState.happiness.toFixed(2)}</div>
            <div className="text-sm text-gray-600">Happiness</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">{consciousState.emotionalState.curiosity.toFixed(2)}</div>
            <div className="text-sm text-gray-600">Curiosity</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">{consciousState.emotionalState.confidence.toFixed(2)}</div>
            <div className="text-sm text-gray-600">Confidence</div>
          </div>
        </div>
      </div>

      {/* Text Input Form */}
      <form onSubmit={handleSubmit} className="mb-6">
        <div className="flex space-x-2">
          <textarea
            ref={textInputRef}
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder="Type your message here... (Press Enter to send, Shift+Enter for new line)"
            className="flex-1 p-3 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            rows={3}
            disabled={isProcessing}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
          />
          <div className="flex flex-col space-y-2">
            <button
              type="submit"
              disabled={isProcessing || !userInput.trim()}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isProcessing ? '⏳' : '💬'}
            </button>
            <button
              type="button"
              onClick={handleStopSpeaking}
              disabled={!consciousState.isSpeaking}
              className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              🔇
            </button>
            <button
              type="button"
              onClick={handleForceSpeak}
              className="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
            >
              🎯
            </button>
          </div>
        </div>
      </form>

      {/* Advanced Controls Toggle */}
      <div className="mb-4">
        <button
          onClick={() => setShowAdvancedControls(!showAdvancedControls)}
          className="text-blue-600 hover:text-blue-800 text-sm font-medium"
        >
          {showAdvancedControls ? '▼' : '▶'} Advanced Voice Controls
        </button>
      </div>

      {/* Advanced Voice Controls */}
      {showAdvancedControls && (
        <div className="bg-gray-50 rounded-lg p-4 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Speaking Style</label>
              <select
                value={voicePreferences.speakingStyle}
                onChange={(e) => handleVoicePreferenceChange('speakingStyle', e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md"
              >
                <option value="contemplative">Contemplative</option>
                <option value="formal">Formal</option>
                <option value="casual">Casual</option>
                <option value="enthusiastic">Enthusiastic</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Speech Rate</label>
              <input
                type="range"
                min="0.5"
                max="2.0"
                step="0.1"
                value={voicePreferences.speechRate}
                onChange={(e) => handleVoicePreferenceChange('speechRate', parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-center text-sm text-gray-600">{voicePreferences.speechRate.toFixed(1)}x</div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Auto-Speak Threshold</label>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.1"
                value={voicePreferences.speakThreshold}
                onChange={(e) => handleVoicePreferenceChange('speakThreshold', parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-center text-sm text-gray-600">{(voicePreferences.speakThreshold * 100).toFixed(0)}%</div>
            </div>
          </div>
          
          <div className="mt-4 flex items-center">
            <input
              type="checkbox"
              id="autoSpeak"
              checked={voicePreferences.autoSpeak}
              onChange={(e) => handleVoicePreferenceChange('autoSpeak', e.target.checked)}
              className="mr-2"
            />
            <label htmlFor="autoSpeak" className="text-sm text-gray-700">
              Enable automatic speaking for high-priority thoughts
            </label>
          </div>
        </div>
      )}

      {/* Conversation History */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Conversation History</h3>
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {consciousState.conversationHistory.length === 0 ? (
            <p className="text-gray-500 text-center py-8">No conversation yet. Start by typing a message above!</p>
          ) : (
            consciousState.conversationHistory.map((entry) => (
              <div
                key={entry.id}
                className={`p-3 rounded-lg ${
                  entry.speaker === 'conscious_agent' 
                    ? 'bg-blue-100 border-l-4 border-blue-500' 
                    : 'bg-green-100 border-l-4 border-green-500'
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="font-medium text-gray-800">
                      {getSpeakerDisplayName(entry.speaker)}
                    </span>
                    {entry.emotion && (
                      <span className="text-lg">{getEmotionEmoji(entry.emotion)}</span>
                    )}
                    {entry.wasSpoken && (
                      <span className="text-xs bg-blue-200 text-blue-800 px-2 py-1 rounded-full">
                        🎤 Spoken
                      </span>
                    )}
                  </div>
                  <span className="text-xs text-gray-500">{formatTimestamp(entry.timestamp)}</span>
                </div>
                <p className="text-gray-700">{entry.content}</p>
              </div>
            ))
          )}
          <div ref={conversationEndRef} />
        </div>
      </div>

      {/* Footer Stats */}
      <div className="mt-6 text-center text-sm text-gray-500">
        <p>Voice System Status: {consciousState.isSpeaking ? '🟢 Active' : '🔴 Idle'}</p>
        <p>Total Conversations: {consciousState.conversationHistory.length}</p>
        <p>Last Spoken: {consciousState.lastSpokenTime ? formatTimestamp(consciousState.lastSpokenTime) : 'Never'}</p>
      </div>
    </div>
  );
};
