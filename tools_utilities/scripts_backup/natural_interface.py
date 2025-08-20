#!/usr/bin/env python3
"""
Natural Interface for Small-Mind Agent Hub

This module provides a completely command-free experience where users can just
ask questions and get intelligent, well-thought-out answers automatically.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)

class NaturalInterface:
    """Natural language interface that eliminates all commands."""
    
    def __init__(self):
        self.setup_logging()
        self.load_system()
    
    def setup_logging(self):
        """Setup logging for the natural interface."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def load_system(self):
        """Load the intelligent routing system."""
        try:
            from planner import auto_route_request
            from registry import ModelRegistry
            from router import choose_model
            from runner import run_model
            from intelligent_feedback import create_feedback_collector
            from cloud_training import create_training_manager
            
            self.auto_route = auto_route_request
            self.registry = ModelRegistry()
            self.choose_model = choose_model
            self.run_model = run_model
            
            # Initialize feedback collection
            self.feedback_collector = create_feedback_collector()
            
            logger.info("Natural interface system loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to load system components: {e}")
            sys.exit(1)
    
    def process_natural_input(self, user_input: str) -> str:
        """
        Process natural language input and return intelligent response.
        
        Args:
            user_input: User's natural language question/request
            
        Returns:
            Intelligent, well-thought-out response
        """
        try:
            # Auto-detect intent and route
            auto_result = self.auto_route(user_input)
            
            # Choose the best model
            model_id = self.choose_model(auto_result['needs'], self.registry.routing, self.registry)
            model = self.registry.get(model_id)
            
            # Execute with the selected model
            result = self.run_model(model, user_input, allow_shell=False, sudo_ok=False)
            
            # Format response intelligently
            formatted_response = self._format_intelligent_response(
                result, auto_result, user_input
            )
            
            # Collect feedback automatically
            self._collect_automatic_feedback(result, user_input, model_id)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return self._generate_error_response(user_input, str(e))
    
    def _format_intelligent_response(self, result: Dict[str, Any], 
                                   auto_result: Dict[str, Any], 
                                   user_input: str) -> str:
        """Format the response intelligently based on intent and content."""
        
        # Extract response content
        stdout = result.get("result", {}).get("stdout", "")
        stderr = result.get("result", {}).get("stderr", "")
        success = result.get("result", {}).get("rc", 1) == 0
        
        # Get routing information
        intent = auto_result['intent']['primary_intent']
        action = auto_result['routing']['action']
        response_config = auto_result['response_config']
        
        # Build intelligent response
        response_parts = []
        
        # Add context header
        response_parts.append(self._generate_context_header(intent, action, user_input))
        
        # Add main content
        if stdout.strip():
            response_parts.append(self._format_main_content(stdout, response_config))
        else:
            response_parts.append("I understand your request, but I need more information to provide a helpful response.")
        
        # Add error information if any
        if stderr.strip():
            response_parts.append(self._format_error_info(stderr))
        
        # Add follow-up suggestions
        response_parts.append(self._generate_follow_up_suggestions(intent, user_input))
        
        # Add quality indicator
        if hasattr(self, 'feedback_collector'):
            quality_report = self.feedback_collector.get_quality_report(result)
            response_parts.append(self._format_quality_indicator(quality_report))
        
        return "\n\n".join(response_parts)
    
    def _generate_context_header(self, intent: str, action: str, user_input: str) -> str:
        """Generate a context-aware header for the response."""
        headers = {
            "question": "🤔 **Understanding Your Question**",
            "command": "⚡ **Executing Your Request**",
            "analysis": "🔍 **Analyzing Your Request**",
            "creation": "🎨 **Creating What You Need**",
            "optimization": "🚀 **Optimizing Your Solution**"
        }
        
        header = headers.get(intent, "💡 **Processing Your Request**")
        
        # Add context about what we're doing
        context_lines = [header]
        
        if intent == "question":
            context_lines.append(f"I'll help you understand: *{user_input}*")
        elif intent == "command":
            context_lines.append(f"I'll execute: *{user_input}*")
        elif intent == "analysis":
            context_lines.append(f"I'll analyze: *{user_input}*")
        elif intent == "creation":
            context_lines.append(f"I'll create: *{user_input}*")
        elif intent == "optimization":
            context_lines.append(f"I'll optimize: *{user_input}*")
        
        return "\n".join(context_lines)
    
    def _format_main_content(self, content: str, response_config: Dict[str, Any]) -> str:
        """Format the main content based on response configuration."""
        format_type = response_config.get("format", "text")
        
        if format_type == "structured":
            return f"📋 **Structured Solution**\n\n{content}"
        elif format_type == "detailed":
            return f"🔍 **Detailed Analysis**\n\n{content}"
        elif format_type == "creative":
            return f"🎨 **Creative Solution**\n\n{content}"
        elif format_type == "step_by_step":
            return f"📝 **Step-by-Step Process**\n\n{content}"
        else:
            return f"💡 **Answer**\n\n{content}"
    
    def _format_error_info(self, stderr: str) -> str:
        """Format error information helpfully."""
        return f"⚠️ **Note**\n\nI encountered some issues:\n```\n{stderr}\n```\n\nThis might help you understand what happened."
    
    def _generate_follow_up_suggestions(self, intent: str, user_input: str) -> str:
        """Generate intelligent follow-up suggestions."""
        suggestions = []
        
        if intent == "question":
            suggestions.extend([
                "Would you like me to explain this in more detail?",
                "Should I show you some examples?",
                "Would you like me to help you implement this?"
            ])
        elif intent == "command":
            suggestions.extend([
                "Would you like me to explain what I just did?",
                "Should I show you how to customize this?",
                "Would you like me to help you with the next steps?"
            ])
        elif intent == "analysis":
            suggestions.extend([
                "Would you like me to dive deeper into any specific aspect?",
                "Should I provide recommendations for improvement?",
                "Would you like me to help you implement the suggestions?"
            ])
        elif intent == "creation":
            suggestions.extend([
                "Would you like me to explain how this works?",
                "Should I show you how to customize it?",
                "Would you like me to help you test it?"
            ])
        elif intent == "optimization":
            suggestions.extend([
                "Would you like me to explain the optimization strategy?",
                "Should I show you how to measure the improvements?",
                "Would you like me to help you implement the changes?"
            ])
        
        if suggestions:
            return f"💭 **Follow-up Questions**\n\n" + "\n".join(f"• {s}" for s in suggestions[:2])
        
        return ""
    
    def _format_quality_indicator(self, quality_report: Dict[str, Any]) -> str:
        """Format quality indicator for transparency."""
        overall_score = quality_report.get("overall_score", 0)
        estimated_rating = quality_report.get("estimated_rating", 3)
        
        # Convert score to emoji
        if overall_score >= 0.8:
            quality_emoji = "🌟"
            quality_text = "Excellent"
        elif overall_score >= 0.6:
            quality_emoji = "✨"
            quality_text = "Good"
        elif overall_score >= 0.4:
            quality_emoji = "👍"
            quality_text = "Fair"
        else:
            quality_emoji = "🤔"
            quality_text = "Basic"
        
        return f"{quality_emoji} **Response Quality: {quality_text}** ({overall_score:.1f}/1.0)"
    
    def _collect_automatic_feedback(self, result: Dict[str, Any], 
                                   user_input: str, model_id: str):
        """Collect feedback automatically for continuous improvement."""
        try:
            if hasattr(self, 'feedback_collector'):
                # Simulate execution metrics
                execution_metrics = {
                    "execution_time": 2.5,  # This would come from actual timing
                    "resource_usage": {"memory_mb": 150, "cpu_percent": 25}
                }
                
                self.feedback_collector.collect_execution_feedback(
                    run_result=result,
                    user_prompt=user_input,
                    model_id=model_id,
                    execution_metrics=execution_metrics
                )
        except Exception as e:
            logger.warning(f"Failed to collect feedback: {e}")
    
    def _generate_error_response(self, user_input: str, error: str) -> str:
        """Generate a helpful error response."""
        return f"""❌ **I encountered an issue**

I'm sorry, but I couldn't process your request: *{user_input}*

**What happened:** {error}

**What you can do:**
• Try rephrasing your question
• Ask for something simpler
• Check if the system is running properly

**I'm here to help** - just ask again in a different way!"""
    
    def interactive_mode(self):
        """Run in interactive mode for natural conversation."""
        print("🧠 **Small-Mind Natural Interface**")
        print("=" * 50)
        print("Just ask questions naturally - no commands needed!")
        print("Type 'quit' or 'exit' to end the conversation.")
        print("=" * 50)
        print()
        
        while True:
            try:
                # Get user input
                user_input = input("💭 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\n👋 Goodbye! Thanks for using Small-Mind!")
                    break
                
                if not user_input:
                    continue
                
                # Process the input
                print("\n🤖 Small-Mind: Processing your request...")
                response = self.process_natural_input(user_input)
                
                # Display response
                print(f"\n{response}")
                print("\n" + "-" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using Small-Mind!")
                break
            except Exception as e:
                print(f"\n❌ An unexpected error occurred: {e}")
                print("Please try again or ask for help.")

def main():
    """Main entry point for the natural interface."""
    if len(sys.argv) > 1:
        # Command line mode - process single input
        interface = NaturalInterface()
        user_input = " ".join(sys.argv[1:])
        response = interface.process_natural_input(user_input)
        print(response)
    else:
        # Interactive mode
        interface = NaturalInterface()
        interface.interactive_mode()

if __name__ == "__main__":
    main()
