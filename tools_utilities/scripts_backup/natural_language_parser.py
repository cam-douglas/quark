#!/usr/bin/env python3
"""
Natural Language Command Parser
Advanced NLP system for mapping user requests to specific commands.
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

@dataclass
class ParsedIntent:
    """Represents a parsed user intent."""
    command_ids: List[str]
    confidence: float
    intent_type: str  # query, execute, help, configure
    parameters: Dict[str, Any]
    requires_confirmation: bool = False
    safety_warning: str = ""

class NaturalLanguageParser:
    """Advanced natural language parser for command interpretation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sentence_model = None
        self.nlp = None
        self.intent_patterns = self._load_intent_patterns()
        self.command_embeddings = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models."""
        try:
            # Load spaCy model for NER and POS tagging
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model not found. Using basic pattern matching.")
            self.nlp = None
        
        try:
            # Load sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            self.logger.warning(f"Could not load sentence transformer: {e}")
            self.sentence_model = None
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load intent recognition patterns."""
        return {
            "execute": [
                r"\b(run|execute|start|launch|perform|do)\b",
                r"\b(can you|please|could you)\s+(run|execute|start)\b",
                r"\b(simulate|train|optimize|deploy|install)\b"
            ],
            "query": [
                r"\b(what|how|where|when|why|which)\b",
                r"\b(show|list|display|find|search)\b",
                r"\b(tell me|explain|describe)\b"
            ],
            "help": [
                r"\b(help|assistance|guide|tutorial)\b",
                r"\b(how do i|how to|what are the steps)\b",
                r"\b(documentation|manual|instructions)\b"
            ],
            "configure": [
                r"\b(configure|setup|install|initialize)\b",
                r"\b(settings|config|preferences)\b",
                r"\b(enable|disable|turn on|turn off)\b"
            ],
            "status": [
                r"\b(status|state|condition|health)\b",
                r"\b(check|verify|validate|test)\b",
                r"\b(is.+running|is.+working|is.+available)\b"
            ]
        }
    
    def parse_user_input(self, user_input: str, command_database) -> ParsedIntent:
        """Parse user input and return structured intent."""
        user_input = user_input.strip().lower()
        
        # Determine intent type
        intent_type = self._classify_intent(user_input)
        
        # Extract entities and parameters
        entities = self._extract_entities(user_input)
        parameters = self._extract_parameters(user_input, entities)
        
        # Find matching commands
        matching_commands = self._find_matching_commands(user_input, entities, command_database)
        
        # Calculate confidence
        confidence = self._calculate_confidence(user_input, matching_commands, intent_type)
        
        # Safety assessment
        requires_confirmation, safety_warning = self._assess_safety(matching_commands, parameters)
        
        return ParsedIntent(
            command_ids=[cmd.id for cmd in matching_commands],
            confidence=confidence,
            intent_type=intent_type,
            parameters=parameters,
            requires_confirmation=requires_confirmation,
            safety_warning=safety_warning
        )
    
    def _classify_intent(self, user_input: str) -> str:
        """Classify the user's intent."""
        scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                try:
                    if re.search(pattern, user_input, re.IGNORECASE):
                        score += 1
                except re.error:
                    # Skip invalid regex patterns
                    continue
            scores[intent] = score
        
        # Return intent with highest score, default to 'query'
        if not scores or max(scores.values()) == 0:
            return "query"
        
        return max(scores, key=scores.get)
    
    def _extract_entities(self, user_input: str) -> Dict[str, List[str]]:
        """Extract named entities from user input."""
        entities = {
            "technologies": [],
            "actions": [],
            "numbers": [],
            "files": [],
            "models": []
        }
        
        # Technology keywords
        tech_keywords = [
            "neural", "brain", "simulation", "aws", "cloud", "docker", "jupyter",
            "python", "cuda", "gpu", "cpu", "memory", "model", "training",
            "optimization", "deployment", "visualization", "analysis"
        ]
        
        # Model names
        model_keywords = [
            "deepseek", "qwen", "mixtao", "llama", "transformer", "moe",
            "bert", "gpt", "claude", "gemini"
        ]
        
        # Action verbs
        action_keywords = [
            "train", "deploy", "optimize", "simulate", "analyze", "visualize",
            "install", "configure", "start", "stop", "run", "execute"
        ]
        
        words = user_input.lower().split()
        
        for word in words:
            if word in tech_keywords:
                entities["technologies"].append(word)
            elif word in model_keywords:
                entities["models"].append(word)
            elif word in action_keywords:
                entities["actions"].append(word)
            elif word.isdigit():
                entities["numbers"].append(word)
            elif "." in word and "/" in word:
                entities["files"].append(word)
        
        # Use spaCy if available
        if self.nlp:
            doc = self.nlp(user_input)
            for ent in doc.ents:
                if ent.label_ in ["PRODUCT", "ORG", "PERSON"]:
                    entities["technologies"].append(ent.text.lower())
                elif ent.label_ in ["CARDINAL", "QUANTITY"]:
                    entities["numbers"].append(ent.text)
        
        return entities
    
    def _extract_parameters(self, user_input: str, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Extract command parameters from user input."""
        parameters = {}
        
        # Extract common parameter patterns
        patterns = {
            "duration": r"(?:for\s+)?(\d+)\s*(?:seconds?|minutes?|hours?|ms|milliseconds?)",
            "steps": r"(\d+)\s*steps?",
            "epochs": r"(\d+)\s*epochs?",
            "neurons": r"(\d+)\s*neurons?",
            "batch_size": r"batch.?size\s*[:=]?\s*(\d+)",
            "learning_rate": r"learning.?rate\s*[:=]?\s*([\d.e-]+)",
            "model_name": r"model\s*[:=]?\s*['\"]?([a-zA-Z0-9\-_.]+)['\"]?",
            "file_path": r"(?:file|path)\s*[:=]?\s*['\"]?([a-zA-Z0-9\-_./]+)['\"]?",
            "region": r"region\s*[:=]?\s*([a-zA-Z0-9\-]+)",
            "instance_type": r"instance.?type\s*[:=]?\s*([a-zA-Z0-9\.]+)"
        }
        
        for param, pattern in patterns.items():
            try:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    parameters[param] = match.group(1)
            except re.error:
                # Skip invalid regex patterns
                continue
        
        # Extract boolean flags
        boolean_patterns = {
            "force": r"\b(?:force|forced|override)\b",
            "verbose": r"\b(?:verbose|detailed|debug)\b",
            "dry_run": r"\b(?:dry.?run|simulate|preview)\b",
            "gpu": r"\b(?:gpu|cuda|nvidia)\b",
            "cpu": r"\b(?:cpu|processor)\b"
        }
        
        for param, pattern in boolean_patterns.items():
            try:
                if re.search(pattern, user_input, re.IGNORECASE):
                    parameters[param] = True
            except re.error:
                # Skip invalid regex patterns
                continue
        
        # Add extracted entities as parameters
        if entities["numbers"]:
            parameters["numbers"] = entities["numbers"]
        if entities["models"]:
            parameters["models"] = entities["models"]
        if entities["technologies"]:
            parameters["technologies"] = entities["technologies"]
        
        return parameters
    
    def _find_matching_commands(self, user_input: str, entities: Dict[str, List[str]], command_database) -> List[Any]:
        """Find commands that match the user input."""
        # Start with keyword-based search
        potential_commands = command_database.search_commands(user_input)
        
        # If no results, try entity-based search
        if not potential_commands and entities["technologies"]:
            for tech in entities["technologies"]:
                potential_commands.extend(command_database.search_commands(tech))
        
        # If no results, try action-based search
        if not potential_commands and entities["actions"]:
            for action in entities["actions"]:
                potential_commands.extend(command_database.search_commands(action))
        
        # Remove duplicates
        seen_ids = set()
        unique_commands = []
        for cmd in potential_commands:
            if cmd.id not in seen_ids:
                unique_commands.append(cmd)
                seen_ids.add(cmd.id)
        
        # Rank by relevance using semantic similarity if available
        if self.sentence_model and unique_commands:
            ranked_commands = self._rank_by_semantic_similarity(user_input, unique_commands)
            return ranked_commands[:5]  # Return top 5 matches
        
        # Fallback: rank by keyword overlap
        return self._rank_by_keyword_overlap(user_input, unique_commands)[:5]
    
    def _rank_by_semantic_similarity(self, user_input: str, commands: List[Any]) -> List[Any]:
        """Rank commands by semantic similarity to user input."""
        if not self.sentence_model:
            return commands
        
        try:
            # Encode user input
            user_embedding = self.sentence_model.encode([user_input])
            
            # Encode command descriptions
            command_texts = []
            for cmd in commands:
                text = f"{cmd.name} {cmd.description} {' '.join(cmd.keywords)}"
                command_texts.append(text)
            
            command_embeddings = self.sentence_model.encode(command_texts)
            
            # Calculate similarities
            similarities = cosine_similarity(user_embedding, command_embeddings)[0]
            
            # Sort commands by similarity
            ranked_indices = np.argsort(similarities)[::-1]
            return [commands[i] for i in ranked_indices]
        
        except Exception as e:
            self.logger.warning(f"Error in semantic ranking: {e}")
            return commands
    
    def _rank_by_keyword_overlap(self, user_input: str, commands: List[Any]) -> List[Any]:
        """Rank commands by keyword overlap."""
        user_words = set(user_input.lower().split())
        
        def calculate_overlap(cmd):
            cmd_words = set(cmd.name.lower().split() + 
                          cmd.description.lower().split() + 
                          cmd.keywords)
            overlap = len(user_words.intersection(cmd_words))
            return overlap
        
        return sorted(commands, key=calculate_overlap, reverse=True)
    
    def _calculate_confidence(self, user_input: str, matching_commands: List[Any], intent_type: str) -> float:
        """Calculate confidence score for the parse result."""
        base_confidence = 0.5
        
        # Boost confidence based on number of matches
        if matching_commands:
            base_confidence += 0.2
            if len(matching_commands) == 1:
                base_confidence += 0.2  # Higher confidence for single match
        
        # Boost confidence based on intent clarity
        intent_indicators = 0
        try:
            intent_indicators = len([pattern for patterns in self.intent_patterns[intent_type] 
                                   for pattern in patterns 
                                   if re.search(pattern, user_input, re.IGNORECASE)])
        except re.error:
            # Skip regex errors
            pass
        base_confidence += min(intent_indicators * 0.1, 0.3)
        
        # Boost confidence for specific terms
        specific_terms = ["execute", "run", "start", "deploy", "train", "optimize"]
        if any(term in user_input.lower() for term in specific_terms):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _assess_safety(self, matching_commands: List[Any], parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Assess safety implications of the parsed command."""
        requires_confirmation = False
        safety_warning = ""
        
        # Check for potentially dangerous commands
        dangerous_operations = ["sudo", "rm", "delete", "destroy", "terminate", "kill"]
        destructive_flags = ["force", "override", "destroy"]
        
        for cmd in matching_commands:
            # Check if command requires elevated privileges
            if cmd.requires_sudo or cmd.requires_shell:
                requires_confirmation = True
                safety_warning = f"⚠️  Command '{cmd.name}' requires elevated privileges"
            
            # Check if command is marked as unsafe
            if not cmd.safe_mode:
                requires_confirmation = True
                safety_warning = f"⚠️  Command '{cmd.name}' may modify system state"
            
            # Check for dangerous keywords
            cmd_text = f"{cmd.name} {cmd.description}".lower()
            if any(op in cmd_text for op in dangerous_operations):
                requires_confirmation = True
                safety_warning = f"⚠️  Command '{cmd.name}' performs potentially destructive operations"
        
        # Check parameters for destructive flags
        if any(flag in parameters for flag in destructive_flags):
            requires_confirmation = True
            if not safety_warning:
                safety_warning = "⚠️  Destructive parameters detected"
        
        return requires_confirmation, safety_warning
    
    def generate_disambiguation_prompt(self, parsed_intent: ParsedIntent, command_database) -> str:
        """Generate a prompt to help user choose between multiple commands."""
        if len(parsed_intent.command_ids) <= 1:
            return ""
        
        prompt = "🤔 I found multiple commands that might match your request:\n\n"
        
        for i, cmd_id in enumerate(parsed_intent.command_ids[:5], 1):
            cmd = command_database.get_command(cmd_id)
            if cmd:
                prompt += f"{i}. {cmd.name}\n"
                prompt += f"   {cmd.description}\n"
                prompt += f"   Category: {cmd.category} | Complexity: {cmd.complexity}\n"
                if cmd.examples:
                    prompt += f"   Example: {cmd.examples[0]}\n"
                prompt += "\n"
        
        prompt += "Please specify which command you'd like to use (1-5), or provide more details."
        return prompt
    
    def suggest_similar_commands(self, user_input: str, command_database) -> List[str]:
        """Suggest similar commands when no exact match is found."""
        # Extract key terms from user input
        key_terms = self._extract_key_terms(user_input)
        suggestions = []
        
        for term in key_terms:
            similar_commands = command_database.search_commands(term)
            for cmd in similar_commands[:3]:  # Top 3 per term
                suggestion = f"• {cmd.name}: {cmd.description}"
                if suggestion not in suggestions:
                    suggestions.append(suggestion)
        
        return suggestions[:10]  # Maximum 10 suggestions
    
    def _extract_key_terms(self, user_input: str) -> List[str]:
        """Extract key terms from user input for suggestions."""
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "how", "what", "when", "where", "why", "can", "could", "would", "should", "do", "does", "did", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "get", "got", "make", "made"}
        
        words = user_input.lower().split()
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms[:5]  # Return top 5 key terms

class CommandCompletionEngine:
    """Auto-completion engine for commands."""
    
    def __init__(self, command_database):
        self.command_database = command_database
        self.completion_cache = {}
    
    def get_completions(self, partial_input: str, limit: int = 10) -> List[str]:
        """Get command completions for partial input."""
        partial_input = partial_input.lower().strip()
        
        if partial_input in self.completion_cache:
            return self.completion_cache[partial_input][:limit]
        
        completions = []
        
        # Get all commands
        all_commands = self.command_database.search_commands("")
        
        for cmd in all_commands:
            # Check if command name starts with partial input
            if cmd.name.lower().startswith(partial_input):
                completions.append(cmd.name)
            
            # Check if any keyword starts with partial input
            for keyword in cmd.keywords:
                if keyword.lower().startswith(partial_input):
                    completions.append(f"{cmd.name} ({keyword})")
        
        # Remove duplicates and sort
        completions = list(set(completions))
        completions.sort()
        
        # Cache results
        self.completion_cache[partial_input] = completions
        
        return completions[:limit]
    
    def get_parameter_suggestions(self, command_id: str, parameter_name: str) -> List[str]:
        """Get parameter value suggestions for a command."""
        cmd = self.command_database.get_command(command_id)
        if not cmd:
            return []
        
        # Common parameter suggestions
        suggestions = {
            "model": ["deepseek-v2", "qwen1.5-moe", "mix-tao-moe"],
            "region": ["us-east-1", "us-west-2", "eu-west-1"],
            "instance_type": ["t3.micro", "g4dn.xlarge", "c5.2xlarge"],
            "duration": ["1000", "2000", "5000"],
            "epochs": ["10", "50", "100", "200"],
            "batch_size": ["16", "32", "64", "128"]
        }
        
        return suggestions.get(parameter_name, [])

if __name__ == "__main__":
    # Test the parser
    from command_database import CommandDatabase
    
    db = CommandDatabase()
    parser = NaturalLanguageParser()
    
    test_inputs = [
        "train a neural network with 100 epochs",
        "deploy to AWS using GPU instance",
        "show me brain simulation commands",
        "help with model optimization",
        "what are the available neuroscience tools?"
    ]
    
    print("🧠 Natural Language Parser Test")
    print("=" * 50)
    
    for test_input in test_inputs:
        print(f"\nInput: '{test_input}'")
        intent = parser.parse_user_input(test_input, db)
        
        print(f"Intent: {intent.intent_type}")
        print(f"Confidence: {intent.confidence:.2f}")
        print(f"Commands found: {len(intent.command_ids)}")
        print(f"Parameters: {intent.parameters}")
        
        if intent.requires_confirmation:
            print(f"⚠️  Safety warning: {intent.safety_warning}")
    
    db.close()
