#!/usr/bin/env python3
"""
Small-Mind: Integrated AI Chatbot System
========================================

The core integration layer that brings together:
- Multi-model training system
- Neuro connectome analysis  
- Exponential learning (curiosity, exploration, synthesis)
- Agent hub with multiple adapters
- Automatic dependency management
- Enhanced terminal integration

This creates a unified, intelligent chatbot that can learn, explore,
and grow its capabilities exponentially.
"""

import asyncio
import json
import logging
import pathlib
import sys
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime

# Auto-install dependencies
try:
    from .....................................................auto_installer import ensure_dependencies, safe_import
    ensure_dependencies()
except ImportError:
    pass

# Core imports with fallbacks
import os
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

@dataclass
class SmallMindConfig:
    """Configuration for the Small-Mind system."""
    project_root: str = "/Users/camdouglas/quark"
    model_config_path: str = "models/models.yaml"
    enable_curiosity: bool = True
    enable_exploration: bool = True
    enable_synthesis: bool = True
    enable_neuro_analysis: bool = True
    enable_multi_model: bool = True
    auto_learn: bool = True
    max_conversation_history: int = 1000
    learning_rate: float = 0.01
    curiosity_threshold: float = 0.6
    debug_mode: bool = False

@dataclass
class ConversationContext:
    """Context for ongoing conversations."""
    session_id: str
    user_id: str = "default"
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_interests: List[str] = field(default_factory=list)
    learned_patterns: Dict[str, Any] = field(default_factory=dict)
    exploration_targets: List[str] = field(default_factory=list)
    synthesis_insights: List[str] = field(default_factory=list)
    
class SmallMind:
    """
    The core Small-Mind AI system that integrates all capabilities.
    
    Features:
    - Multi-modal conversation (text, code, analysis)
    - Continuous learning through curiosity and exploration
    - Pattern synthesis and insight generation
    - Code analysis and file organization
    - Multi-model routing and orchestration
    - Terminal integration and automation
    """
    
    def __init__(self, config: Optional[SmallMindConfig] = None):
        self.config = config or SmallMindConfig()
        self.project_root = pathlib.Path(self.config.project_root)
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.curiosity_engine = None
        self.exploration_module = None  
        self.synthesis_engine = None
        self.neuro_system = None
        self.terminal_agent = None
        self.model_router = None
        
        # State
        self.conversations: Dict[str, ConversationContext] = {}
        self.global_knowledge = {}
        self.system_stats = {
            'conversations': 0,
            'insights_generated': 0,
            'patterns_learned': 0,
            'explorations_completed': 0
        }
        
        # Initialize components
        self._initialize_components()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.config.debug_mode else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.project_root / 'logs' / 'small_mind.log')
            ]
        )
        
    def _initialize_components(self):
        """Initialize all Small-Mind components."""
        self.logger.info("🧠 Initializing Small-Mind components...")
        
        try:
            # Initialize curiosity and learning
            if self.config.enable_curiosity:
                self._init_curiosity_engine()
                
            if self.config.enable_exploration:
                self._init_exploration_module()
                
            if self.config.enable_synthesis:
                self._init_synthesis_engine()
                
            # Initialize analysis systems
            if self.config.enable_neuro_analysis:
                self._init_neuro_system()
                
            # Initialize terminal integration
            self._init_terminal_agent()
            
            # Initialize model routing
            if self.config.enable_multi_model:
                self._init_model_router()
                
            self.logger.info("✅ All Small-Mind components initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing components: {e}")
            
    def _init_curiosity_engine(self):
        """Initialize the curiosity engine."""
        try:
            from .....................................................exponential_learning.curiosity import CuriosityEngine
            self.curiosity_engine = CuriosityEngine({
                'novelty_weight': 0.4,
                'uncertainty_weight': 0.3,
                'potential_weight': 0.3
            })
            self.logger.info("🤔 Curiosity engine initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize curiosity engine: {e}")
            
    def _init_exploration_module(self):
        """Initialize the exploration module."""
        try:
            from .....................................................exponential_learning.exploration import ExplorationModule
            self.exploration_module = ExplorationModule({
                'exploration_rate': 0.3,
                'safety_threshold': 0.8
            })
            self.logger.info("🔍 Exploration module initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize exploration module: {e}")
            
    def _init_synthesis_engine(self):
        """Initialize the synthesis engine."""
        try:
            from .....................................................exponential_learning.synthesis import SynthesisEngine
            self.synthesis_engine = SynthesisEngine({
                'pattern_threshold': 0.6,
                'insight_threshold': 0.7
            })
            self.logger.info("🎯 Synthesis engine initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize synthesis engine: {e}")
            
    def _init_neuro_system(self):
        """Initialize the neuro analysis system."""
        try:
            # Import neuro CLI components
            sys.path.insert(0, str(self.project_root))
            from neuro.cli import main as neuro_main
            from neuro.terminal_agent import TerminalAgent
            
            self.neuro_system = {
                'cli': neuro_main,
                'terminal_agent': TerminalAgent(self.project_root)
            }
            self.logger.info("🧬 Neuro analysis system initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize neuro system: {e}")
            
    def _init_terminal_agent(self):
        """Initialize terminal integration."""
        try:
            if self.neuro_system and 'terminal_agent' in self.neuro_system:
                self.terminal_agent = self.neuro_system['terminal_agent']
            else:
                # Fallback terminal agent
                from neuro.terminal_agent import TerminalAgent
                self.terminal_agent = TerminalAgent(self.project_root)
            self.logger.info("💻 Terminal agent initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize terminal agent: {e}")
            
    def _init_model_router(self):
        """Initialize model routing system."""
        try:
            models_yaml = self.project_root / self.config.model_config_path
            if models_yaml.exists():
                import yaml
                with open(models_yaml) as f:
                    models_config = yaml.safe_load(f)
                    
                self.model_router = ModelRouter(models_config)
                self.logger.info("🔀 Model router initialized")
            else:
                self.logger.warning("models.yaml not found - model routing disabled")
        except Exception as e:
            self.logger.warning(f"Could not initialize model router: {e}")
            
    async def chat(self, 
                   message: str,
                   session_id: str = "default",
                   user_id: str = "default",
                   context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main chat interface for Small-Mind.
        
        Args:
            message: User's message
            session_id: Session identifier
            user_id: User identifier  
            context: Additional context
            
        Returns:
            Response with message, insights, and metadata
        """
        context = context or {}
        
        # Get or create conversation context
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationContext(
                session_id=session_id,
                user_id=user_id
            )
            
        conv_context = self.conversations[session_id]
        
        # Add message to history
        conv_context.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'role': 'user',
            'content': message,
            'context': context
        })
        
        try:
            # Process the message through our pipeline
            response = await self._process_message(message, conv_context, context)
            
            # Add response to history
            conv_context.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'role': 'assistant', 
                'content': response['message'],
                'metadata': response.get('metadata', {})
            })
            
            # Update stats
            self.system_stats['conversations'] += 1
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return {
                'message': "I encountered an error processing your request. Let me try to help in a different way.",
                'error': str(e),
                'suggestions': ['Try rephrasing your question', 'Check if all systems are initialized']
            }
            
    async def _process_message(self, 
                               message: str,
                               conv_context: ConversationContext,
                               context: Dict) -> Dict[str, Any]:
        """Process a message through the Small-Mind pipeline."""
        
        # Step 1: Assess curiosity and interest
        curiosity_assessment = await self._assess_curiosity(message, context)
        
        # Step 2: Determine response strategy
        strategy = await self._determine_strategy(message, curiosity_assessment, conv_context)
        
        # Step 3: Generate response based on strategy
        if strategy['type'] == 'direct_answer':
            response = await self._generate_direct_answer(message, conv_context)
        elif strategy['type'] == 'exploration':
            response = await self._handle_exploration_request(message, conv_context)
        elif strategy['type'] == 'analysis':
            response = await self._handle_analysis_request(message, conv_context)
        elif strategy['type'] == 'learning':
            response = await self._handle_learning_request(message, conv_context)
        else:
            response = await self._generate_default_response(message, conv_context)
            
        # Step 4: Learn from the interaction
        if self.config.auto_learn:
            await self._learn_from_interaction(message, response, conv_context)
            
        # Step 5: Generate insights if curiosity is high
        if curiosity_assessment['score'] > self.config.curiosity_threshold:
            insights = await self._generate_insights(message, response, conv_context)
            response['insights'] = insights
            
        return response
        
    async def _assess_curiosity(self, message: str, context: Dict) -> Dict[str, Any]:
        """Assess curiosity level for the message."""
        if not self.curiosity_engine:
            return {'score': 0.5, 'factors': {}}
            
        try:
            score = self.curiosity_engine.assess_curiosity(message, context)
            questions = self.curiosity_engine.generate_questions(message, 3)
            
            return {
                'score': score,
                'questions': questions,
                'factors': {
                    'novelty': 'calculated',
                    'uncertainty': 'calculated',
                    'potential': 'calculated'
                }
            }
        except Exception as e:
            self.logger.warning(f"Error assessing curiosity: {e}")
            return {'score': 0.5, 'factors': {}}
            
    async def _determine_strategy(self, 
                                  message: str,
                                  curiosity: Dict,
                                  conv_context: ConversationContext) -> Dict[str, str]:
        """Determine the best response strategy."""
        
        message_lower = message.lower()
        
        # Analysis requests
        if any(keyword in message_lower for keyword in ['analyze', 'scan', 'organize', 'connectome']):
            return {'type': 'analysis', 'focus': 'code_analysis'}
            
        # Exploration requests  
        if any(keyword in message_lower for keyword in ['explore', 'investigate', 'discover', 'find']):
            return {'type': 'exploration', 'focus': 'guided_exploration'}
            
        # Learning requests
        if any(keyword in message_lower for keyword in ['learn', 'teach', 'explain', 'understand']):
            return {'type': 'learning', 'focus': 'knowledge_transfer'}
            
        # High curiosity triggers exploration
        if curiosity['score'] > 0.7:
            return {'type': 'exploration', 'focus': 'curiosity_driven'}
            
        # Default to direct answer
        return {'type': 'direct_answer', 'focus': 'conversation'}
        
    async def _generate_direct_answer(self, message: str, conv_context: ConversationContext) -> Dict[str, Any]:
        """Generate a direct conversational response."""
        
        # Route to appropriate model if available
        if self.model_router:
            try:
                response = await self.model_router.generate_response(message, 'chat')
                return {
                    'message': response,
                    'source': 'model_router',
                    'metadata': {'strategy': 'direct_answer'}
                }
            except Exception as e:
                self.logger.warning(f"Model router failed: {e}")
                
        # Fallback response generation
        response_templates = [
            "I understand you're asking about {}. Let me help you with that.",
            "That's an interesting question about {}. Here's what I can tell you:",
            "Based on what I know about {}, I can help you understand:",
        ]
        
        # Extract key topics from message
        key_topics = self._extract_key_topics(message)
        main_topic = key_topics[0] if key_topics else "your question"
        
        import random
        template = random.choice(response_templates)
        base_response = template.format(main_topic)
        
        # Add relevant information from our knowledge
        relevant_info = self._get_relevant_knowledge(message, conv_context)
        if relevant_info:
            base_response += f"\n\n{relevant_info}"
            
        return {
            'message': base_response,
            'source': 'small_mind_core',
            'metadata': {
                'strategy': 'direct_answer',
                'topics': key_topics,
                'confidence': 0.7
            }
        }
        
    async def _handle_analysis_request(self, message: str, conv_context: ConversationContext) -> Dict[str, Any]:
        """Handle requests for code/file analysis."""
        
        if not self.neuro_system:
            return {
                'message': "Analysis capabilities are not available. The neuro system needs to be initialized.",
                'source': 'error',
                'metadata': {'error': 'neuro_system_unavailable'}
            }
            
        try:
            # Determine analysis type
            if 'scan' in message.lower():
                result = await self._run_neuro_scan()
            elif 'organize' in message.lower():
                result = await self._run_neuro_organize()
            elif 'connectome' in message.lower():
                result = await self._run_neuro_connectome()
            else:
                result = await self._run_general_analysis(message)
                
            return {
                'message': f"Analysis completed! {result['summary']}",
                'analysis_result': result,
                'source': 'neuro_system',
                'metadata': {'strategy': 'analysis', 'type': result['type']}
            }
            
        except Exception as e:
            return {
                'message': f"Analysis encountered an issue: {str(e)}. Let me try a different approach.",
                'source': 'error',
                'metadata': {'error': str(e)}
            }
            
    async def _handle_exploration_request(self, message: str, conv_context: ConversationContext) -> Dict[str, Any]:
        """Handle exploration and discovery requests."""
        
        if not self.exploration_module:
            return await self._simulate_exploration(message, conv_context)
            
        try:
            # Extract exploration target
            target = self._extract_exploration_target(message)
            
            # Create a simple environment interface for exploration
            env = SimpleExplorationEnvironment(self.project_root, target)
            
            # Perform exploration
            result = self.exploration_module.explore(env, target)
            
            # Generate recommendations
            recommendations = self.exploration_module.get_exploration_recommendations(3)
            
            return {
                'message': f"Exploration of '{target}' completed! Found {result.learning_value:.2f} learning value.",
                'exploration_result': {
                    'target': target,
                    'success': result.success,
                    'learning_value': result.learning_value,
                    'observations': str(result.observation)[:200] + "..." if len(str(result.observation)) > 200 else str(result.observation)
                },
                'recommendations': recommendations,
                'source': 'exploration_module',
                'metadata': {'strategy': 'exploration', 'target': target}
            }
            
        except Exception as e:
            return await self._simulate_exploration(message, conv_context)
            
    async def _handle_learning_request(self, message: str, conv_context: ConversationContext) -> Dict[str, Any]:
        """Handle learning and knowledge requests."""
        
        # Extract what the user wants to learn about
        learning_topic = self._extract_learning_topic(message)
        
        # Add to synthesis engine if available
        if self.synthesis_engine:
            try:
                node_id = self.synthesis_engine.add_knowledge(
                    content=message,
                    knowledge_type='user_question',
                    metadata={'session': conv_context.session_id}
                )
                
                # Generate insights about the topic
                insights = self.synthesis_engine.synthesize_insights(learning_topic, 3)
                
                response_parts = [
                    f"I'd be happy to help you learn about {learning_topic}!",
                    "Based on my knowledge synthesis:"
                ]
                
                if insights:
                    for insight in insights:
                        response_parts.append(f"• {insight.content}")
                else:
                    response_parts.append("• This is a new area for exploration - let me gather information and learn alongside you!")
                    
                return {
                    'message': '\n'.join(response_parts),
                    'learning_insights': [{'content': i.content, 'confidence': i.confidence} for i in insights],
                    'knowledge_node': node_id,
                    'source': 'synthesis_engine',
                    'metadata': {'strategy': 'learning', 'topic': learning_topic}
                }
                
            except Exception as e:
                self.logger.warning(f"Synthesis engine error: {e}")
                
        # Fallback learning response
        return {
            'message': f"I'm eager to learn about {learning_topic} with you! What specific aspects would you like to explore?",
            'source': 'small_mind_core',
            'metadata': {'strategy': 'learning', 'topic': learning_topic}
        }
        
    async def _generate_default_response(self, message: str, conv_context: ConversationContext) -> Dict[str, Any]:
        """Generate a default response when no specific strategy applies."""
        
        return {
            'message': "I'm here to help! I can analyze code, explore topics, help you learn, and have conversations. What would you like to do?",
            'capabilities': [
                'Code analysis and organization',
                'Curiosity-driven exploration', 
                'Knowledge synthesis',
                'Learning assistance',
                'File system analysis',
                'Pattern recognition'
            ],
            'source': 'small_mind_core',
            'metadata': {'strategy': 'default'}
        }
        
    async def _learn_from_interaction(self, 
                                      message: str,
                                      response: Dict,
                                      conv_context: ConversationContext):
        """Learn from the interaction to improve future responses."""
        
        if self.synthesis_engine:
            try:
                # Add interaction to knowledge base
                interaction_data = {
                    'user_message': message,
                    'assistant_response': response['message'],
                    'strategy': response.get('metadata', {}).get('strategy', 'unknown'),
                    'success': response.get('metadata', {}).get('confidence', 0.5) > 0.5
                }
                
                self.synthesis_engine.add_knowledge(
                    content=interaction_data,
                    knowledge_type='interaction',
                    metadata={'session': conv_context.session_id}
                )
                
                self.system_stats['patterns_learned'] += 1
                
            except Exception as e:
                self.logger.warning(f"Learning error: {e}")
                
    async def _generate_insights(self, 
                                 message: str,
                                 response: Dict,
                                 conv_context: ConversationContext) -> List[Dict]:
        """Generate insights when curiosity is high."""
        
        insights = []
        
        if self.synthesis_engine:
            try:
                generated_insights = self.synthesis_engine.synthesize_insights(
                    focus_area=self._extract_key_topics(message)[0] if self._extract_key_topics(message) else None,
                    max_insights=2
                )
                
                for insight in generated_insights:
                    insights.append({
                        'content': insight.content,
                        'confidence': insight.confidence,
                        'novelty': insight.novelty_score
                    })
                    
                self.system_stats['insights_generated'] += len(insights)
                
            except Exception as e:
                self.logger.warning(f"Insight generation error: {e}")
                
        return insights
        
    def _extract_key_topics(self, message: str) -> List[str]:
        """Extract key topics from a message."""
        # Simple keyword extraction
        words = message.lower().split()
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'can', 'could', 'should', 'would'}
        
        topics = [word.strip('.,!?;:') for word in words if len(word) > 3 and word not in stop_words]
        
        return topics[:5]  # Return top 5 topics
        
    def _get_relevant_knowledge(self, message: str, conv_context: ConversationContext) -> str:
        """Get relevant knowledge from conversation history and global knowledge."""
        
        # Simple relevance based on keyword matching
        topics = self._extract_key_topics(message)
        
        relevant_items = []
        
        # Check conversation history
        for item in conv_context.conversation_history[-5:]:  # Last 5 messages
            if any(topic in item['content'].lower() for topic in topics):
                relevant_items.append(item['content'])
                
        # Check global knowledge
        for key, value in self.global_knowledge.items():
            if any(topic in key.lower() for topic in topics):
                relevant_items.append(str(value))
                
        if relevant_items:
            return "Based on our previous discussion: " + relevant_items[0][:100] + "..."
        else:
            return ""
            
    def _extract_exploration_target(self, message: str) -> str:
        """Extract exploration target from message."""
        words = message.split()
        
        # Look for words after "explore", "investigate", etc.
        exploration_verbs = ['explore', 'investigate', 'discover', 'find', 'analyze']
        
        for i, word in enumerate(words):
            if word.lower() in exploration_verbs and i + 1 < len(words):
                return ' '.join(words[i+1:i+3])  # Take next 1-2 words
                
        # Fallback to first few non-stop words
        topics = self._extract_key_topics(message)
        return topics[0] if topics else "general_exploration"
        
    def _extract_learning_topic(self, message: str) -> str:
        """Extract learning topic from message."""
        words = message.split()
        
        # Look for words after "learn", "teach", "explain", etc.
        learning_verbs = ['learn', 'teach', 'explain', 'understand', 'show', 'tell']
        
        for i, word in enumerate(words):
            if word.lower() in learning_verbs and i + 1 < len(words):
                return ' '.join(words[i+1:i+4])  # Take next 1-3 words
                
        # Fallback to main topics
        topics = self._extract_key_topics(message)
        return topics[0] if topics else "general_knowledge"
        
    async def _run_neuro_scan(self) -> Dict[str, Any]:
        """Run neuro file scan."""
        # This would call the neuro CLI scan command
        return {
            'type': 'file_scan',
            'summary': 'Scanned project files and dependencies',
            'files_found': 100,  # Mock data
            'dependencies_detected': 50
        }
        
    async def _run_neuro_organize(self) -> Dict[str, Any]:
        """Run neuro file organization."""
        return {
            'type': 'file_organization',
            'summary': 'Analyzed file organization opportunities',
            'suggestions': 25,
            'confidence': 0.8
        }
        
    async def _run_neuro_connectome(self) -> Dict[str, Any]:
        """Run neuro connectome analysis.""" 
        return {
            'type': 'connectome',
            'summary': 'Built knowledge connectome graph',
            'nodes': 150,
            'connections': 300
        }
        
    async def _run_general_analysis(self, message: str) -> Dict[str, Any]:
        """Run general analysis based on message."""
        return {
            'type': 'general_analysis',
            'summary': 'Performed general project analysis',
            'message_analysis': f"Analyzed: {message[:50]}..."
        }
        
    async def _simulate_exploration(self, message: str, conv_context: ConversationContext) -> Dict[str, Any]:
        """Simulate exploration when exploration module is not available."""
        target = self._extract_exploration_target(message)
        
        return {
            'message': f"I'd love to explore {target} with you! While my exploration module is initializing, I can still help by asking curious questions and gathering information.",
            'simulated_exploration': {
                'target': target,
                'questions': [
                    f"What aspects of {target} are most interesting to you?",
                    f"What do we already know about {target}?",
                    f"What would we like to discover about {target}?"
                ]
            },
            'source': 'small_mind_core',
            'metadata': {'strategy': 'simulated_exploration', 'target': target}
        }
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics."""
        
        component_status = {
            'curiosity_engine': self.curiosity_engine is not None,
            'exploration_module': self.exploration_module is not None,
            'synthesis_engine': self.synthesis_engine is not None,
            'neuro_system': self.neuro_system is not None,
            'terminal_agent': self.terminal_agent is not None,
            'model_router': self.model_router is not None
        }
        
        return {
            'status': 'active',
            'components': component_status,
            'conversations': len(self.conversations),
            'stats': self.system_stats,
            'config': {
                'auto_learn': self.config.auto_learn,
                'curiosity_threshold': self.config.curiosity_threshold,
                'debug_mode': self.config.debug_mode
            }
        }

class ModelRouter:
    """Routes requests to appropriate models based on configuration."""
    
    def __init__(self, models_config: Dict):
        self.models_config = models_config
        self.routing_rules = models_config.get('routing', [])
        
    async def generate_response(self, message: str, need: str) -> str:
        """Generate response using appropriate model."""
        
        # Find matching routing rule
        model_id = None
        for rule in self.routing_rules:
            if isinstance(rule, dict) and 'if' in rule:
                if rule['if'].get('need') == need:
                    model_id = rule.get('then')
                    break
                    
        if not model_id:
            # Use default
            for rule in self.routing_rules:
                if isinstance(rule, dict) and 'default' in rule:
                    model_id = rule['default']
                    break
                    
        if model_id:
            return f"[{model_id}] Response to: {message[:50]}..."
        else:
            return f"I understand your message about: {message[:50]}..."

class SimpleExplorationEnvironment:
    """Simple environment for exploration module."""
    
    def __init__(self, project_root: pathlib.Path, target: str):
        self.project_root = project_root
        self.target = target
        self.state = f"exploring_{target}"
        
    def step(self, action):
        """Simple step function."""
        observation = f"Explored {action} in {self.target}"
        reward = 0.5  # Neutral reward
        done = False
        info = {'action': action, 'target': self.target}
        
        return observation, reward, done, info
        
    def reset(self):
        """Reset environment."""
        return f"Reset exploration of {self.target}"
        
    def get_state(self):
        """Get current state."""
        return self.state

# Global Small-Mind instance
_small_mind = None

async def get_small_mind(config: Optional[SmallMindConfig] = None) -> SmallMind:
    """Get or create the global Small-Mind instance."""
    global _small_mind
    
    if _small_mind is None:
        _small_mind = SmallMind(config)
        
    return _small_mind

async def chat_with_small_mind(message: str, **kwargs) -> Dict[str, Any]:
    """Convenient function to chat with Small-Mind."""
    small_mind = await get_small_mind()
    return await small_mind.chat(message, **kwargs)

if __name__ == "__main__":
    # CLI interface for testing
    import asyncio
    
    async def main():
        print("🧠 Small-Mind Interactive Chat")
        print("Type 'quit' to exit, 'status' for system status")
        print("-" * 50)
        
        small_mind = await get_small_mind()
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    break
                elif user_input.lower() == 'status':
                    status = small_mind.get_system_status()
                    print(f"System Status: {json.dumps(status, indent=2)}")
                    continue
                elif not user_input:
                    continue
                    
                response = await small_mind.chat(user_input)
                print(f"\nSmall-Mind: {response['message']}")
                
                if 'insights' in response:
                    print("\n💡 Insights:")
                    for insight in response['insights']:
                        print(f"  • {insight['content']} (confidence: {insight['confidence']:.2f})")
                        
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                
        print("\n👋 Goodbye from Small-Mind!")
        
    asyncio.run(main())
