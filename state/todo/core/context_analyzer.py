"""
Context Analyzer Module
=======================
Analyzes natural language commands and current work context.
"""

import re
from typing import Dict, Tuple, Optional, List
from pathlib import Path


class ContextAnalyzer:
    """Analyzes commands and work context to determine intent."""
    
    def __init__(self):
        # Command patterns and their mappings
        # Order matters! More specific patterns must come before general ones
        self.patterns = {
            # Brain patterns (most specific - must come first)
            r'\bbrain\s+orchestrate\b': ('brain', 'orchestrate'),
            r'\bbrain\s+startup\b': ('brain', 'startup'),
            r'\bbrain\s+status\b': ('brain', 'status'),
            r'\bbrain\s+analyze\b': ('brain', 'analyze'),
            r'\bbrain\s+test\b': ('brain', 'test'),
            r'\bbrain\s+profile\b': ('brain', 'profile'),
            r'\bbrain\s+visualize\b': ('brain', 'visualize'),
            r'\bbrain\s+list\b': ('brain', 'list'),
            r'\b(simulate|sim)\s+(cerebellum|cortex|hippocampus|basal[_\s]?ganglia|brainstem|thalamus|amygdala)\b': ('brain', 'simulate'),
            r'\b(simulate|sim)\s+(morphogen|alphagenome|e8|consciousness)\b': ('brain', 'simulate'),
            r'\b(simulate|sim)\s+(cognitive|memory|motor|sensory|learning)\b': ('brain', 'simulate'),
            
            # Task-related patterns
            r'\b(plan|create|new)\s+(task|work|feature)\b': ('task', 'plan'),
            r'\b(generate|make|create)\s+(from|using)\s+roadmap\b': ('task', 'generate'),
            r'\b(work|execute|do|implement)\s+(on|task)\b': ('task', 'execute'),
            r'\b(track|progress|status)\b': ('task', 'track'),
            r'\b(list|show)\s+(tasks?|work|todos?)\b': ('task', 'list'),
            r'\b(review|completed|done)\b': ('task', 'review'),
            
            # Validation-related patterns
            r'\b(validate|check|verify)\s+(foundation|cerebellum|stage\s*\d+)\b': ('validate', 'verify'),
            r'\b(validate|check)\s+(current|changes|my\s+work)\b': ('validate', 'quick'),
            r'\b(validation|validate)\s+(guide|sprint|interactive)\b': ('validate', 'sprint'),
            r'\b(metrics|kpis?|measurements?)\b': ('validate', 'metrics'),
            r'\b(dashboard|report|visualization)\b': ('validate', 'dashboard'),
            
            # General simulation patterns
            r'\b(run|start)\s+brain\s+simulation\b': ('brain', 'simulate'),
            r'\b(simulation)\s+(status|check)\b': ('simulation', 'status'),
            r'\b(analyze)\s+(simulation|results)\b': ('simulation', 'analyze'),
            
            # Training patterns
            r'\b(train|training)\s+(model|network)\b': ('training', 'train'),
            r'\b(train|training)\s+(gcp|cloud)\b': ('training', 'train'),
            r'\b(training)\s+(status|check)\b': ('training', 'status'),
            r'\b(stop|halt)\s+training\b': ('training', 'stop'),
            r'\b(resume)\s+training\b': ('training', 'resume'),
            r'\b(training)\s+(metrics|loss)\b': ('training', 'metrics'),
            
            # Deployment patterns
            r'\b(deploy)\s+(to\s+)?(gcp|cloud|docker|local)\b': ('deployment', 'deploy'),
            r'\b(deployment)\s+(status|check)\b': ('deployment', 'status'),
            r'\b(rollback)\s+(deployment|deploy)\b': ('deployment', 'rollback'),
            r'\b(deployment)\s+(logs|output)\b': ('deployment', 'logs'),
            
            # Documentation patterns
            r'\b(generate|create)\s+(docs?|documentation)\b': ('documentation', 'generate'),
            r'\b(update)\s+(readme|docs?)\b': ('documentation', 'update'),
            r'\b(generate)\s+(api)\s+(docs?)\b': ('documentation', 'api'),
            r'\b(check)\s+(docs?|documentation)\b': ('documentation', 'check'),
            
            # Benchmarking patterns
            r'\b(benchmark|bench)\s+(performance|memory|inference|training)\b': ('benchmarking', 'benchmark'),
            r'\b(compare)\s+(metrics|benchmarks?)\b': ('benchmarking', 'compare'),
            r'\b(profile)\s+(cpu|gpu|performance)\b': ('benchmarking', 'profile'),
            r'\b(benchmark)\s+(report|results)\b': ('benchmarking', 'report'),
            
            # State system patterns
            r'\b(state|guardian|agent)\s+(status|check)\b': ('state', 'status'),
            r'\b(evolve|advance|continue)\b': ('state', 'evolve'),
            
            # Testing patterns
            r'\b(test|pytest|unit\s+test)\b': ('test', 'run'),
            r'\b(coverage|test\s+coverage)\b': ('test', 'coverage'),
            
            # Git patterns
            r'\b(commit|save|checkpoint)\b': ('git', 'commit'),
            r'\b(git\s+)?status\b': ('git', 'status'),
            r'\b(diff|changes)\b': ('git', 'diff'),
            
            # General patterns
            r'\bwhat\'?s?\s+(next|pending|todo)\b': ('workflow', 'next'),
            r'\b(help|how|what\s+can)\b': ('help', 'show'),
            r'\b(clean|cleanup)\b': ('maintenance', 'clean'),
        }
        
        # Priority keywords for disambiguation
        self.priority_keywords = {
            'validate': ['validate', 'validation', 'verify', 'check'],
            'task': ['task', 'plan', 'execute', 'work', 'implement'],
            'brain': ['brain', 'cerebellum', 'cortex', 'hippocampus', 'neural', 'morphogen'],
            'simulation': ['simulate', 'simulation', 'environment', 'physics'],
            'training': ['train', 'training', 'model', 'epoch', 'checkpoint'],
            'deployment': ['deploy', 'deployment', 'rollback', 'docker', 'gcp'],
            'documentation': ['docs', 'documentation', 'readme', 'api'],
            'benchmarking': ['benchmark', 'profile', 'performance', 'metrics'],
            'test': ['test', 'pytest', 'coverage'],
            'state': ['state', 'guardian', 'agent', 'evolve'],
        }
    
    def analyze_command(self, command: str) -> Tuple[str, str, Dict]:
        """
        Analyze a natural language command.
        Returns: (system, action, parameters)
        """
        command_lower = command.lower()
        
        # Check for exact matches first
        for pattern, (system, action) in self.patterns.items():
            if re.search(pattern, command_lower):
                # Extract parameters
                params = self._extract_parameters(command_lower, system, action)
                return system, action, params
        
        # If no match, try to determine by keywords
        system = self._determine_system_by_keywords(command_lower)
        action = 'help'  # Default to help if unclear
        
        return system, action, {}
    
    def _extract_parameters(self, command: str, system: str, action: str) -> Dict:
        """Extract parameters from command based on system and action."""
        params = {}
        
        if system == 'validate' and action == 'verify':
            # Extract domain/stage
            stage_match = re.search(r'stage\s*(\d+)', command)
            if stage_match:
                params['stage'] = int(stage_match.group(1))
            
            domain_match = re.search(r'(foundation|cerebellum|developmental|brainstem)', command)
            if domain_match:
                params['domain'] = domain_match.group(1)
        
        elif system == 'task':
            # Extract task ID if present
            id_match = re.search(r'task[_-]?(\w+)', command)
            if id_match:
                params['task_id'] = id_match.group(1)
            
            # Extract category
            cat_match = re.search(r'(foundation|cerebellum|developmental|brainstem)', command)
            if cat_match:
                params['category'] = cat_match.group(1)
        
        elif system == 'brain':
            if action == 'simulate':
                # Extract brain component to simulate
                comp_match = re.search(r'(cerebellum|cortex|hippocampus|basal[_\s]?ganglia|brainstem|'
                                     r'thalamus|amygdala|morphogen|alphagenome|e8|consciousness|'
                                     r'cognitive|memory|motor|sensory|learning|full)', command)
                if comp_match:
                    component = comp_match.group(1).replace(' ', '_')
                    params['component'] = component
            
            elif action == 'orchestrate':
                # Extract orchestration operation
                op_match = re.search(r'orchestrate\s+(status|store_knowledge|plan_action|process_sensory|learn_skill)', command)
                if op_match:
                    params['operation'] = op_match.group(1)
            
            elif action == 'startup':
                # Extract startup mode
                mode_match = re.search(r'startup\s+(minimal|cognitive_only|full)', command)
                if mode_match:
                    params['mode'] = mode_match.group(1)
            
            else:
                # Extract component for other brain actions
                comp_match = re.search(r'(cerebellum|cortex|hippocampus|basal[_\s]?ganglia|brainstem|'
                                     r'morphogen|alphagenome|e8)', command)
                if comp_match:
                    params['component'] = comp_match.group(1).replace(' ', '_')
        
        elif system == 'simulation' and action == 'simulate':
            # Extract non-brain component to simulate
            comp_match = re.search(r'(physics|environment|world)', command)
            if comp_match:
                params['component'] = comp_match.group(1)
        
        elif system == 'training' and action == 'train':
            # Extract stage
            stage_match = re.search(r'stage\s*(\d+)', command)
            if stage_match:
                params['stage'] = int(stage_match.group(1))
            
            # Extract mode (gcp/local)
            mode_match = re.search(r'(gcp|cloud|local)', command)
            if mode_match:
                params['mode'] = mode_match.group(1)
        
        elif system == 'deployment' and action == 'deploy':
            # Extract platform
            platform_match = re.search(r'(gcp|docker|local|cloud)', command)
            if platform_match:
                params['platform'] = platform_match.group(1)
        
        elif system == 'benchmarking' and action == 'benchmark':
            # Extract benchmark type
            type_match = re.search(r'(performance|memory|inference|training)', command)
            if type_match:
                params['type'] = type_match.group(1)
        
        elif system == 'benchmarking' and action == 'profile':
            # Extract profile type
            type_match = re.search(r'(cpu|gpu)', command)
            if type_match:
                params['type'] = type_match.group(1)
        
        elif system == 'documentation' and action == 'generate':
            # Extract doc type
            type_match = re.search(r'(api|validation|roadmap|architecture|all)', command)
            if type_match:
                params['type'] = type_match.group(1)
        
        return params
    
    def _determine_system_by_keywords(self, command: str) -> str:
        """Determine which system to use based on keyword frequency."""
        scores = {}
        
        for system, keywords in self.priority_keywords.items():
            score = sum(1 for keyword in keywords if keyword in command)
            if score > 0:
                scores[system] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        return 'workflow'  # Default to workflow
    
    def get_current_context(self) -> Dict:
        """Get current work context from git and file system."""
        import subprocess
        
        context = {
            'git_branch': 'main',
            'has_changes': False,
            'current_task': None,
            'recent_files': []
        }
        
        try:
            # Get git branch
            result = subprocess.run(['git', 'branch', '--show-current'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                context['git_branch'] = result.stdout.strip()
            
            # Check for changes
            result = subprocess.run(['git', 'status', '--porcelain'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                context['has_changes'] = bool(result.stdout.strip())
            
            # Get recently modified files
            result = subprocess.run(['git', 'diff', '--name-only', 'HEAD'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                context['recent_files'] = result.stdout.strip().split('\n')[:5]
        except:
            pass
        
        return context
    
    def suggest_next_action(self, context: Dict) -> List[str]:
        """Suggest next actions based on current context."""
        suggestions = []
        
        if context.get('has_changes'):
            suggestions.append("validate current changes")
            suggestions.append("run tests")
            suggestions.append("commit changes")
        
        if context.get('current_task'):
            suggestions.append(f"continue task {context['current_task']}")
        else:
            suggestions.append("list pending tasks")
            suggestions.append("plan new task")
        
        suggestions.append("check validation status")
        
        return suggestions
