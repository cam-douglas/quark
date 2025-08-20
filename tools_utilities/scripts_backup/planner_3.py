import re
from typing import Dict, Any, List, Set, Tuple
import logging

logger = logging.getLogger(__name__)

# Enhanced capability patterns for intelligent routing
CAPABILITY_PATTERNS = {
    "shell": [
        r"\b(install|brew|pip|sudo|apt|yum|chmod|chown|mkdir|rm|cp|mv|ln|tar|gzip|ssh|scp)\b",
        r"\b(system|command|terminal|bash|zsh|shell|exec|run|execute)\b",
        r"\b(process|kill|ps|top|htop|iotop|netstat|lsof)\b"
    ],
    "fs": [
        r"\b(file|directory|folder|path|ls|find|grep|cat|head|tail|less|more)\b",
        r"\b(read|write|create|delete|modify|rename|copy|move|link)\b",
        r"\b(permission|ownership|size|date|metadata|inode)\b"
    ],
    "python": [
        r"\b(python|pip|conda|venv|virtualenv|requirements|setup\.py|pyproject\.toml)\b",
        r"\b(import|from|class|def|function|module|package|library)\b",
        r"\b(numpy|pandas|matplotlib|scikit|tensorflow|pytorch|jupyter)\b"
    ],
    "planning": [
        r"\b(plan|strategy|approach|method|steps|sequence|order|timeline|roadmap)\b",
        r"\b(break down|decompose|organize|structure|outline|framework)\b",
        r"\b(goal|objective|target|milestone|deadline|schedule)\b"
    ],
    "reasoning": [
        r"\b(explain|reason|analyze|understand|comprehend|interpret|clarify)\b",
        r"\b(why|how|what if|because|therefore|however|although)\b",
        r"\b(logic|deduction|inference|conclusion|hypothesis|theory)\b"
    ],
    "retrieval": [
        r"\b(search|find|lookup|query|retrieve|fetch|get|obtain|access)\b",
        r"\b(database|index|catalog|archive|repository|storage|memory)\b",
        r"\b(document|text|content|information|data|knowledge)\b"
    ],
    "summarize": [
        r"\b(summarize|summary|abstract|overview|brief|concise|condense)\b",
        r"\b(extract|highlight|key points|main ideas|gist|essence)\b",
        r"\b(shorten|abbreviate|simplify|distill|capture)\b"
    ],
    "classify": [
        r"\b(classify|categorize|group|sort|organize|label|tag|identify)\b",
        r"\b(type|category|class|kind|sort|variety|classification)\b",
        r"\b(pattern|similarity|difference|distinction|characteristic)\b"
    ],
    "orchestrate": [
        r"\b(coordinate|orchestrate|manage|oversee|supervise|direct|guide)\b",
        r"\b(workflow|pipeline|sequence|coordination|management|leadership)\b",
        r"\b(team|collaboration|cooperation|integration|synchronization)\b"
    ],
    "code": [
        r"\b(code|program|script|algorithm|function|method|class|module)\b",
        r"\b(debug|test|compile|build|deploy|version|git|repository)\b",
        r"\b(syntax|semantics|logic|structure|architecture|design)\b"
    ],
    "chat": [
        r"\b(chat|conversation|discussion|talk|speak|communicate|interact)\b",
        r"\b(hello|hi|greeting|question|answer|help|assist|support)\b",
        r"\b(opinion|thought|idea|suggestion|advice|recommendation)\b"
    ]
}

# Intent patterns for automatic command detection
INTENT_PATTERNS = {
    "question": [
        r"\b(what|how|why|when|where|who|which|can|could|would|should|is|are|do|does)\b",
        r"\b\?\s*$",  # Ends with question mark
        r"\b(help|explain|tell me|show me|give me)\b"
    ],
    "command": [
        r"\b(do|make|create|build|set|get|run|start|stop|install|configure|setup)\b",
        r"\b(go|move|change|switch|open|close|save|load|export|import)\b"
    ],
    "analysis": [
        r"\b(analyze|examine|investigate|review|assess|evaluate|check|verify)\b",
        r"\b(look at|study|research|explore|inspect|audit|test)\b"
    ],
    "creation": [
        r"\b(write|create|build|develop|design|compose|generate|produce)\b",
        r"\b(make|construct|assemble|put together|form|establish)\b"
    ],
    "optimization": [
        r"\b(optimize|improve|enhance|boost|speed up|make faster|efficient)\b",
        r"\b(refactor|restructure|reorganize|streamline|simplify)\b"
    ]
}

# Response strategy patterns
RESPONSE_STRATEGIES = {
    "question": "explanatory",
    "command": "actionable",
    "analysis": "analytical",
    "creation": "creative",
    "optimization": "iterative"
}

# Priority weights for capability matching
CAPABILITY_WEIGHTS = {
    "shell": 10,      # High priority for system operations
    "fs": 8,          # High priority for file operations
    "python": 7,      # High priority for Python tasks
    "planning": 6,    # Medium-high for planning tasks
    "reasoning": 5,   # Medium for reasoning tasks
    "retrieval": 4,   # Medium for retrieval tasks
    "summarize": 3,   # Lower for summarization
    "classify": 3,    # Lower for classification
    "orchestrate": 6, # Medium-high for orchestration
    "code": 7,        # High for code tasks
    "chat": 1         # Lowest priority for general chat
}

def infer_needs(prompt: str, tools: List[str] = None) -> Dict[str, Any]:
    """
    Infer capabilities needed from prompt text using pattern matching and analysis.
    
    Args:
        prompt: The user's prompt text
        tools: Explicitly requested tools (optional)
    
    Returns:
        Dict with inferred needs and confidence scores
    """
    prompt_lower = prompt.lower()
    needs = set()
    confidence_scores = {}
    
    # Check explicit tool requests first
    if tools:
        for tool in tools:
            if tool in CAPABILITY_PATTERNS:
                needs.add(tool)
                confidence_scores[tool] = 1.0  # High confidence for explicit requests
    
    # Pattern-based inference
    for capability, patterns in CAPABILITY_PATTERNS.items():
        if capability in needs:  # Skip if already explicitly requested
            continue
            
        score = 0.0
        for pattern in patterns:
            matches = re.findall(pattern, prompt_lower)
            if matches:
                # Score based on number of matches and pattern complexity
                score += len(matches) * 0.3
                # Bonus for longer/more specific patterns
                score += len(pattern) * 0.01
        
        if score > 0.5:  # Threshold for inclusion
            needs.add(capability)
            confidence_scores[capability] = min(score, 1.0)
    
    # Context-aware adjustments
    if "install" in prompt_lower or "setup" in prompt_lower:
        needs.add("shell")
        confidence_scores["shell"] = max(confidence_scores.get("shell", 0), 0.9)
    
    if "error" in prompt_lower or "debug" in prompt_lower:
        needs.add("reasoning")
        confidence_scores["reasoning"] = max(confidence_scores.get("reasoning", 0), 0.8)
    
    if "compare" in prompt_lower or "difference" in prompt_lower:
        needs.add("reasoning")
        confidence_scores["reasoning"] = max(confidence_scores.get("reasoning", 0), 0.7)
    
    # Fallback to chat if no specific needs detected
    if not needs:
        needs.add("chat")
        confidence_scores["chat"] = 0.5
    
    # Sort needs by priority
    sorted_needs = sorted(needs, key=lambda x: CAPABILITY_WEIGHTS.get(x, 0), reverse=True)
    
    logger.info(f"Inferred needs: {sorted_needs} with confidences: {confidence_scores}")
    
    return {
        "need": sorted_needs,
        "confidence": confidence_scores,
        "primary_need": sorted_needs[0] if sorted_needs else "chat",
        "complexity": _assess_complexity(prompt_lower, needs)
    }

def detect_intent(prompt: str) -> Dict[str, Any]:
    """
    Automatically detect user intent without requiring explicit commands.
    
    Args:
        prompt: User's natural language input
        
    Returns:
        Dict with detected intent and response strategy
    """
    prompt_lower = prompt.lower()
    intent_scores = {}
    
    # Score each intent type
    for intent, patterns in INTENT_PATTERNS.items():
        score = 0.0
        for pattern in patterns:
            matches = re.findall(pattern, prompt_lower)
            if matches:
                score += len(matches) * 0.4
                # Bonus for question marks
                if "?" in prompt:
                    score += 0.3
                # Bonus for imperative verbs
                if any(verb in prompt_lower for verb in ["do", "make", "create", "build"]):
                    score += 0.2
        
        if score > 0.0:
            intent_scores[intent] = score
    
    # Determine primary intent
    if intent_scores:
        primary_intent = max(intent_scores, key=intent_scores.get)
        response_strategy = RESPONSE_STRATEGIES.get(primary_intent, "general")
    else:
        primary_intent = "general"
        response_strategy = "general"
    
    # Detect urgency and tone
    urgency = _detect_urgency(prompt_lower)
    tone = _detect_tone(prompt_lower)
    
    return {
        "primary_intent": primary_intent,
        "intent_scores": intent_scores,
        "response_strategy": response_strategy,
        "urgency": urgency,
        "tone": tone,
        "confidence": max(intent_scores.values()) if intent_scores else 0.0
    }

def _detect_urgency(prompt: str) -> str:
    """Detect urgency level in the prompt."""
    urgent_patterns = [
        r"\b(urgent|asap|immediately|now|quick|fast|hurry|emergency|critical)\b",
        r"\b(deadline|due|time|schedule|meeting|appointment)\b"
    ]
    
    for pattern in urgent_patterns:
        if re.search(pattern, prompt):
            return "high"
    
    if any(word in prompt for word in ["soon", "later", "when", "schedule"]):
        return "medium"
    
    return "low"

def _detect_tone(prompt: str) -> str:
    """Detect the tone of the prompt."""
    tone_patterns = {
        "formal": [r"\b(please|kindly|would you|could you|request|require)\b"],
        "casual": [r"\b(hey|hi|yo|what's up|cool|awesome|great)\b"],
        "technical": [r"\b(implement|algorithm|optimize|performance|efficiency|scalability)\b"],
        "friendly": [r"\b(thanks|thank you|appreciate|help|assist|support)\b"]
    }
    
    for tone, patterns in tone_patterns.items():
        for pattern in patterns:
            if re.search(pattern, prompt):
                return tone
    
    return "neutral"

def auto_route_request(prompt: str) -> Dict[str, Any]:
    """
    Automatically route a request without explicit commands.
    This is the main function for intelligent agent selection.
    
    Args:
        prompt: User's natural language input
        
    Returns:
        Dict with routing decision and response strategy
    """
    # Detect intent and needs
    intent = detect_intent(prompt)
    needs = infer_needs(prompt)
    
    # Combine intent and needs for optimal routing
    routing_decision = _make_routing_decision(intent, needs)
    
    # Determine response format and style
    response_config = _determine_response_config(intent, needs, routing_decision)
    
    return {
        "routing": routing_decision,
        "intent": intent,
        "needs": needs,
        "response_config": response_config,
        "auto_command": _generate_auto_command(intent, needs)
    }

def _make_routing_decision(intent: Dict[str, Any], needs: Dict[str, Any]) -> Dict[str, Any]:
    """Make intelligent routing decision based on intent and needs."""
    primary_intent = intent["primary_intent"]
    primary_need = needs["primary_need"]
    complexity = needs["complexity"]
    
    # Intent-based routing logic
    if primary_intent == "question":
        if complexity == "high":
            return {"action": "plan", "model_type": "reasoning", "priority": "high"}
        else:
            return {"action": "ask", "model_type": "chat", "priority": "medium"}
    
    elif primary_intent == "command":
        if "shell" in needs["need"]:
            return {"action": "run", "model_type": "shell", "priority": "high"}
        else:
            return {"action": "execute", "model_type": "action", "priority": "medium"}
    
    elif primary_intent == "analysis":
        return {"action": "analyze", "model_type": "reasoning", "priority": "high"}
    
    elif primary_intent == "creation":
        return {"action": "create", "model_type": "creative", "priority": "medium"}
    
    elif primary_intent == "optimization":
        return {"action": "optimize", "model_type": "planning", "priority": "high"}
    
    else:
        # Default routing based on needs
        return {"action": "ask", "model_type": primary_need, "priority": "medium"}

def _determine_response_config(intent: Dict[str, Any], needs: Dict[str, Any], routing: Dict[str, Any]) -> Dict[str, Any]:
    """Determine how to format and present the response."""
    strategy = intent["response_strategy"]
    urgency = intent["urgency"]
    tone = intent["tone"]
    
    config = {
        "format": "text",
        "detail_level": "standard",
        "include_examples": True,
        "include_code": "code" in needs["need"],
        "include_planning": "planning" in needs["need"],
        "urgency_handling": urgency,
        "tone_adjustment": tone
    }
    
    # Adjust based on strategy
    if strategy == "explanatory":
        config["detail_level"] = "high"
        config["include_examples"] = True
    elif strategy == "actionable":
        config["format"] = "structured"
        config["include_planning"] = True
    elif strategy == "analytical":
        config["format"] = "detailed"
        config["include_code"] = True
    elif strategy == "creative":
        config["format"] = "creative"
        config["include_examples"] = True
    elif strategy == "iterative":
        config["format"] = "step_by_step"
        config["include_planning"] = True
    
    return config

def _generate_auto_command(intent: Dict[str, Any], needs: Dict[str, Any]) -> str:
    """Generate the appropriate command automatically."""
    primary_intent = intent["primary_intent"]
    
    if primary_intent == "question":
        return "ask"
    elif primary_intent == "command":
        return "run"
    elif primary_intent == "analysis":
        return "analyze"
    elif primary_intent == "creation":
        return "create"
    elif primary_intent == "optimization":
        return "optimize"
    else:
        return "ask"

def _assess_complexity(prompt: str, needs: Set[str]) -> str:
    """Assess the complexity of the request based on needs and prompt length."""
    if len(needs) > 3:
        return "high"
    elif len(needs) > 1:
        return "medium"
    else:
        return "low"

def validate_needs(needs: Dict[str, Any]) -> bool:
    """Validate that the inferred needs are reasonable and complete."""
    if not needs.get("need"):
        return False
    
    # Check for conflicting needs
    conflicting_pairs = [
        ({"shell", "fs"}, {"chat"}),  # Shell/FS operations vs general chat
        ({"planning", "orchestrate"}, {"chat"})  # Planning vs general chat
    ]
    
    need_set = set(needs["need"])
    for pair in conflicting_pairs:
        if pair[0].intersection(need_set) and pair[1].intersection(need_set):
            logger.warning(f"Potential conflicting needs detected: {pair[0]} vs {pair[1]}")
    
    return True

def suggest_alternative_needs(original_needs: Dict[str, Any]) -> List[str]:
    """Suggest alternative capabilities that might be useful."""
    suggestions = []
    primary = original_needs.get("primary_need", "chat")
    
    # Suggest complementary capabilities
    complementary_map = {
        "shell": ["fs", "python"],
        "fs": ["shell", "python"],
        "python": ["shell", "fs"],
        "planning": ["reasoning", "orchestrate"],
        "reasoning": ["planning", "retrieval"],
        "retrieval": ["summarize", "classify"],
        "code": ["python", "reasoning"],
        "chat": ["reasoning", "retrieval"]
    }
    
    if primary in complementary_map:
        suggestions.extend(complementary_map[primary])
    
    return list(set(suggestions))  # Remove duplicates
