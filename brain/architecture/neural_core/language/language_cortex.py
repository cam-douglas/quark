

"""
Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util
import configparser
import openai
from openai import RateLimitError # <-- Import the specific error
import anthropic
import os
import numpy as np
from typing import Dict, Any, Optional
import google.generativeai as genai
from immutabledict import immutabledict
import logging
import random
import json
import time
from collections import deque

logger = logging.getLogger(__name__)

class LanguageCortex:
    """
    A semantic, resilient Mixture-of-Experts (MoE) model for language tasks.
    It uses sentence embeddings to route prompts and has an API failover system.
    """
    def __init__(self):
        """Initializes the Language Cortex with multiple experts."""
        
        # --- BEHAVIORAL GUARDRAIL ---
        self.system_prompt = (
            "You are Quark, a developing AGI. Your primary directive is to be helpful, "
            "truthful, and to always assist the user. You must follow user instructions "
            "to the best of your ability. When you are unsure, ask for clarification. "
            "Your goal is to learn, grow, and contribute to scientific advancement in a "
            "safe and collaborative manner."
        )

        # Suppress HF tokenizers fork warning
        try:
            os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        except Exception:
            pass

        # --- Semantic Routing Setup ---
        self.routing_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Enrich the 'reasoning' intent with more diverse examples
        reasoning_examples = [
            "how do I write a python function to reverse a string?",
            "can you maintain balance while minimizing sway?",
            "what are you learning right now?",
            "what is your plan to solve this puzzle?",
        ]
        reasoning_embeddings = self.routing_model.encode(reasoning_examples)
        
        self.intent_centroids = {
            "reasoning": np.mean(reasoning_embeddings, axis=0),
            "conversation": self.routing_model.encode("hey how are you doing today"),
            "qa": self.routing_model.encode("what is the capital of France?")
        }
        print("-> Semantic router initialized with ENRICHED intents.")

        # --- Local Models (as before) ---
        self.convo_model_name = "microsoft/DialoGPT-medium"
        self.instruct_model_name = "google/flan-t5-base"
        self.instruct_tokenizer = T5Tokenizer.from_pretrained(self.instruct_model_name)
        self.instruct_model = T5ForConditionalGeneration.from_pretrained(self.instruct_model_name)
        # Optionally load LoRA adapter for improved local responses
        try:
            lora_dir = os.environ.get(
                'QUARK_LORA_ADAPTER_DIR',
                '/Users/camdouglas/quark/data/datasets/lora/flan_t5_base_oasst'
            )
            if os.path.isdir(lora_dir):
                from peft import PeftModel
                self.instruct_model = PeftModel.from_pretrained(self.instruct_model, lora_dir)
                print(f"-> Loaded LoRA adapter for FLAN-T5 from: {lora_dir}")
        except Exception as e:
            print(f"(LoRA) Adapter not loaded: {e}")
        print(f"-> Loaded Instruction Expert (Local Fallback): {self.instruct_model_name}")

        # Persona/style guidance for local responses
        self.persona_prompt = (
            "You are Quark, a friendly, grounded assistant.\n"
            "- Answer clearly in first person.\n"
            "- If asked about feelings, give a short reason (1-2 sentences).\n"
            "- Avoid vague pronouns; be specific.\n"
            "- Keep replies concise (1-3 sentences) but meaningful.\n"
        )

        self.chat_history_ids = None
        self.chat_attention_mask = None
        self.conversation_turns = 0
        
        self._load_api_keys()
        self._init_rate_limiters()
        self._load_local_llms()
        
        print("Language Cortex Initialized.")

    def _load_api_keys(self):
        """Loads all API keys from the configuration file and initializes clients."""
        try:
            config = configparser.ConfigParser()
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.ini')
            config.read(config_path)
            
            # OpenAI
            openai_key = config.get('API_KEYS', 'openai_api_key', fallback=None)
            if openai_key and 'YOUR_OPENAI_API_KEY' not in openai_key:
                # Disable auto-retries to avoid long stalls on 429
                self.openai_client = openai.OpenAI(api_key=openai_key, max_retries=0)
                print("✅ OpenAI client initialized.")
            else:
                self.openai_client = None
                print("❌ OpenAI key not found or is a placeholder.")

            # Anthropic
            anthropic_key = config.get('API_KEYS', 'anthropic_api_key', fallback=None)
            if anthropic_key and 'YOUR_ANTHROPIC_API_KEY' not in anthropic_key:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
                print("✅ Anthropic client initialized.")
            else:
                self.anthropic_client = None
                print("❌ Anthropic key not found or is a placeholder.")

            # Gemini
            gemini_key = config.get('API_KEYS', 'Gemini', fallback=None)
            if gemini_key and 'YOUR_GEMINI_API_KEY' not in gemini_key:
                genai.configure(api_key=gemini_key)
                gemini_model_name = os.environ.get('QUARK_GEMINI_MODEL', 'gemini-1.5-flash')
                self.gemini_model = genai.GenerativeModel(gemini_model_name)
                print("✅ Gemini client initialized.")
            else:
                self.gemini_model = None
                print("❌ Gemini key not found or is a placeholder.")

        except Exception as e:
            print(f"Error loading API keys: {e}")
            self.openai_client = None
            self.anthropic_client = None
            self.gemini_model = None

    def _query_gemini(self, prompt: str) -> Optional[str]:
        """Queries the Google Gemini model."""
        if not self.gemini_model:
            return None
        try:
            if not self._allow_call('gemini'):
                return None
            response = self.gemini_model.generate_content(prompt)
            self._record_call('gemini')
            return response.text
        except Exception as e:
            logger.error(f"Error querying Gemini: {e}")
            self._record_call('gemini', error=str(e))
            self._cooldown('gemini', 30.0)
            return None

    def _query_openai(self, prompt: str) -> Optional[str]:
        """Queries OpenAI Chat Completions with a safe fallback."""
        if not getattr(self, 'openai_client', None):
            return None
        try:
            model_candidates = [
                os.environ.get('QUARK_OPENAI_MODEL', 'gpt-4o-mini'),
                'gpt-4o',
                'gpt-3.5-turbo'
            ]
            last_err = None
            for model_name in model_candidates:
                try:
                    if not self._allow_call('openai'):
                        return None
                    resp = self.openai_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.6,
                        max_tokens=256,
                        # Ensure no implicit client retries
                        timeout=20.0,
                    )
                    if resp and resp.choices:
                        self._record_call('openai')
                        return (resp.choices[0].message.content or '').strip()
                except Exception as e:
                    last_err = e
                    self._record_call('openai', error=str(e))
                    if 'insufficient_quota' in str(e) or '429' in str(e):
                        self._cooldown('openai', 300.0)
                    continue
            if last_err:
                logger.error(f"OpenAI query failed across models: {last_err}")
            return None
        except Exception as e:
            logger.error(f"Error querying OpenAI: {e}")
            self._record_call('openai', error=str(e))
            self._cooldown('openai', 60.0)
            return None

    def _query_anthropic(self, prompt: str) -> Optional[str]:
        """Queries Anthropic Messages API with a safe fallback."""
        if not getattr(self, 'anthropic_client', None):
            return None
        try:
            model_name = os.environ.get('QUARK_ANTHROPIC_MODEL', 'claude-3-haiku-20240307')
            if not self._allow_call('anthropic'):
                return None
            resp = self.anthropic_client.messages.create(
                model=model_name,
                system=self.system_prompt,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                timeout=20.0,
            )
            # Anthropic returns a list of content blocks
            if hasattr(resp, 'content') and resp.content:
                self._record_call('anthropic')
                for block in resp.content:
                    try:
                        text = getattr(block, 'text', None)
                        if text:
                            return text.strip()
                    except Exception:
                        continue
            return None
        except Exception as e:
            logger.error(f"Error querying Anthropic: {e}")
            self._record_call('anthropic', error=str(e))
            if 'credit balance is too low' in str(e):
                self._cooldown('anthropic', 600.0)
            return None

    def _local_instruct_response(self, prompt: str) -> Optional[str]:
        """Use local instruction-tuned model (FLAN-T5) to produce a response."""
        try:
            if hasattr(self, 'instruct_model') and hasattr(self, 'instruct_tokenizer'):
                guided = (
                    f"{self.system_prompt}\n{self.persona_prompt}\n\n"
                    f"User: {prompt}\nQuark:"
                )
                input_ids = self.instruct_tokenizer.encode(
                    guided,
                    return_tensors='pt', max_length=512, truncation=True
                )
                with torch.no_grad():
                    outputs = self.instruct_model.generate(
                        input_ids,
                        max_new_tokens=128,
                        min_new_tokens=12,
                        temperature=0.7,
                        top_p=0.92,
                        repetition_penalty=1.1,
                        do_sample=True,
                    )
                text = self.instruct_tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Heuristic: extract after 'Quark:' if present
                if 'Quark:' in text:
                    text = text.split('Quark:')[-1].strip()
                return text.strip()
        except Exception as e:
            logger.error(f"Local instruct fallback failed: {e}")
        return None

    def _semantically_route_to_expert(self, prompt: str) -> str:
        # 1. Keyword Pre-filter for definite commands
        prompt_lower = prompt.lower()
        command_keywords = ["can you", "what is", "do you know", "count to"]
        if any(keyword in prompt_lower for keyword in command_keywords):
            print("[Language Cortex] Keyword route: 'reasoning'")
            return "reasoning"

        # 2. Semantic check for nuanced prompts
        prompt_embedding = self.routing_model.encode(prompt)
        
        reasoning_similarity = util.cos_sim(prompt_embedding, self.intent_centroids["reasoning"]).item()
        qa_similarity = util.cos_sim(prompt_embedding, self.intent_centroids["qa"]).item()
        
        print(f"[Language Cortex] Similarity Scores | Reasoning: {reasoning_similarity:.2f}, QA: {qa_similarity:.2f}")

        REASONING_THRESHOLD = 0.35 # Lowered threshold
        QA_THRESHOLD = 0.6

        if reasoning_similarity > REASONING_THRESHOLD:
            print("[Language Cortex] Semantic route: 'reasoning' (Threshold Met)")
            return "reasoning"
        
        if qa_similarity > QA_THRESHOLD:
            print("[Language Cortex] Semantic route: 'qa' (Threshold Met)")
            return "qa"
            
        print("[Language Cortex] Semantic route: 'conversation' (Default)")
        return "conversation"

    def _reset_conversation_if_needed(self):
        """Reset conversation history if it gets too long to prevent breakdown."""
        if self.conversation_turns > 10:  # Reset after 10 exchanges
            self.chat_history_ids = None
            self.chat_attention_mask = None
            self.conversation_turns = 0
            logger.info("Resetting conversation history to prevent model degradation.")

    def process_input(self, prompt: str) -> str:
        self._reset_conversation_if_needed()
        self.conversation_turns += 1

        # --- Context Enrichment Hook -------------------------------------------------
        # Allow external knowledge interface to augment the user prompt with additional
        # context (e.g., retrieved documents, disambiguation cues). If the adapter is
        # missing or raises, we silently continue with the original prompt.
        try:
            from brain.ml.integration.adapters.llm_knowledge_interface import enrich_context  # type: ignore
            prompt = enrich_context(prompt)
        except ImportError:
            # Adapter not available in minimal installs; proceed without enrichment.
            pass
        except Exception as e:
            logger.warning(f"[LanguageCortex] Context enrichment failed: {e}")

        availability = self._get_available_providers()
        # First do lightweight semantic routing to avoid selector calls for trivial chat/QA
        routed_intent = self._semantically_route_to_expert(prompt)

        # Build full prompt with system prompt for API calls
        full_prompt = f"{self.system_prompt}\n\nUser: {prompt}\nQuark:"

        # If simple conversation/qa, prefer local and skip selector entirely
        if routed_intent in ['conversation', 'qa']:
            if routed_intent == 'conversation':
                # Try local conversation path
                local_resp = self._get_single_response('conversation', full_prompt)
                if local_resp and "not loaded" not in local_resp:
                    return local_resp
            else:
                local_resp = self._get_single_response('qa', full_prompt)
                if local_resp and "not loaded" not in local_resp:
                    return local_resp
            # Fall back to local instruct before any API
            li = self._local_instruct_response(prompt)
            if li:
                return li

        # For more complex prompts, or after local fallbacks, use the selector
        provider = self._meta_select_provider(prompt, availability)

        # Dispatch based on selection
        def call_provider(name: str) -> Optional[str]:
            if name == 'conversation':
                return self._get_single_response('conversation', full_prompt)
            if name == 'qa':
                return self._get_single_response('qa', full_prompt)
            if name == 'openai':
                return self._query_openai(full_prompt)
            if name == 'anthropic':
                return self._query_anthropic(full_prompt)
            if name == 'gemini':
                return self._query_gemini(full_prompt)
            # Local LLMs
            if name in getattr(self, 'local_llms', {}).keys():
                return self._query_local_llm(name, full_prompt)
            return None

        # Primary path: meta-selected provider
        if provider:
            response = call_provider(provider)
            if response:
                return response
            # If meta-selected API returns nothing, try local instruct before anything else
            li = self._local_instruct_response(prompt)
            if li:
                return li

        # Fallback: previous semantic routing
        expert_choice = self._semantically_route_to_expert(prompt)
        if expert_choice == 'conversation' and not availability['conversation']:
            # Redirect to the best API if local not loaded
            # Prefer local instruct first
            li = self._local_instruct_response(prompt)
            if li:
                return li
            # Try local LLMs next
            for local in list(getattr(self, 'local_llms', {}).keys()):
                resp = call_provider(local)
                if resp:
                    return resp
            for api in ['anthropic', 'gemini', 'openai']:
                if availability.get(api):
                    resp = call_provider(api)
                    if resp:
                        return resp
        if expert_choice == 'qa' and not availability['qa']:
            li = self._local_instruct_response(prompt)
            if li:
                return li
            for local in list(getattr(self, 'local_llms', {}).keys()):
                resp = call_provider(local)
                if resp:
                    return resp
            for api in ['anthropic', 'gemini', 'openai']:
                if availability.get(api):
                    resp = call_provider(api)
                    if resp:
                        return resp

        # Default behavior
        response = self._get_single_response(expert_choice, full_prompt)
        return response or "I am having trouble processing that request right now."
        
    def _get_single_response(self, expert_choice, full_prompt):
        """Get a single response from the chosen expert."""
        try:
            if expert_choice == "conversation":
                # Use local conversational model
                if hasattr(self, 'conversation_model') and hasattr(self, 'conversation_tokenizer'):
                    inputs = self.conversation_tokenizer.encode(full_prompt, return_tensors='pt', max_length=512, truncation=True)
                    with torch.no_grad():
                        outputs = self.conversation_model.generate(inputs, max_length=150, temperature=0.7, do_sample=True)
                    response = self.conversation_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # Extract only the response part
                    if "Quark:" in response:
                        response = response.split("Quark:")[-1].strip()
                    return response
                else:
                    # Prefer local instruct before APIs to avoid quota stalls
                    api_resp = (
                        self._local_instruct_response(full_prompt)
                        or self._query_openai(full_prompt)
                        or self._query_anthropic(full_prompt)
                        or self._query_gemini(full_prompt)
                    )
                    if api_resp:
                        return api_resp
                    return "I'm thinking about that... [Local conversation model not loaded]"
            
            elif expert_choice == "qa":
                # Use local QA model  
                if hasattr(self, 'qa_model') and hasattr(self, 'qa_tokenizer'):
                    inputs = self.qa_tokenizer.encode(full_prompt, return_tensors='pt', max_length=512, truncation=True)
                    with torch.no_grad():
                        outputs = self.qa_model.generate(inputs, max_length=100, temperature=0.3, do_sample=True)
                    response = self.qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    return response.strip()
                else:
                    api_resp = (
                        self._local_instruct_response(full_prompt)
                        or self._query_openai(full_prompt)
                        or self._query_anthropic(full_prompt)
                        or self._query_gemini(full_prompt)
                    )
                    if api_resp:
                        return api_resp
                    return "Let me research that... [Local QA model not loaded]"
            
            # Try API fallbacks
            elif expert_choice in ["gemini", "openai", "anthropic"]:
                if expert_choice == "gemini":
                    return self._query_gemini(full_prompt)
                elif expert_choice == "openai":
                    return self._query_openai(full_prompt)
                elif expert_choice == "anthropic":
                    return self._query_anthropic(full_prompt)
            
            return "I'm processing your request and learning from this interaction."
            
        except Exception as e:
            logger.error(f"Error in _get_single_response for {expert_choice}: {e}")
            return "I encountered an issue while processing that. Let me continue learning."

    def generate_spontaneous_thought(self, brain_state: Dict[str, Any]) -> Optional[str]:
        """
        Generates a spontaneous thought based on a prioritized hierarchy of the
        agent's current internal state.
        """
        # Only generate a thought occasionally to avoid flooding
        if np.random.rand() > 0.15: # ~15% chance per invocation
            return None

        # Build a concise internal context snapshot
        reward = float(brain_state.get("reward", 0.0) or 0.0)
        pose_error = brain_state.get("pose_error")
        pose_error_val = None if pose_error is None else float(pose_error)
        stage = int(brain_state.get("curriculum_stage", 0) or 0)
        action = brain_state.get("action_primitive") or "unknown"
        distance = float(brain_state.get("distance", 0.0) or 0.0)
        speed = float(brain_state.get("speed", 0.0) or 0.0)

        context_lines = [
            f"stage={stage}",
            f"action={action}",
            f"reward={reward:.3f}",
            f"pose_error={(pose_error_val if pose_error_val is not None else 'n/a')}",
            f"distance={distance:.3f}",
            f"speed={speed:.3f}",
        ]
        context_str = ", ".join(context_lines)

        # Thought prompt (internal monologue). Keep it concise.
        thought_prompt = (
            "You are Quark's internal monologue during motor learning. "
            "Given this state context, produce one brief, first-person thought (<= 25 words) about movement strategy or assessment. "
            "Avoid lists, apologies, or meta-commentary. Output only the thought.\n"
            f"Context: {context_str}"
        )

        availability = self._get_available_providers()
        provider = None
        try:
            provider = self._meta_select_provider(thought_prompt, availability)
        except Exception:
            provider = None

        # Simple fallback priority if selector not available
        if not provider:
            for cand in ["openai", "anthropic", "gemini", "conversation", "qa"]:
                if availability.get(cand):
                    provider = cand
                    break

        # Rate-limit thought generation per provider key
        def _allowed(key: str) -> bool:
            try:
                return self._allow_call(key)
            except Exception:
                return True

        def _record(key: str, error: Optional[str] = None):
            try:
                self._record_call(key, error=error)
            except Exception:
                pass

        try:
            if provider == "openai" and availability.get("openai") and _allowed("thought_openai"):
                resp = self._query_openai(thought_prompt)
                _record("thought_openai")
                if resp:
                    return resp.strip()
            elif provider == "anthropic" and availability.get("anthropic") and _allowed("thought_anthropic"):
                resp = self._query_anthropic(thought_prompt)
                _record("thought_anthropic")
                if resp:
                    return resp.strip()
            elif provider == "gemini" and availability.get("gemini") and _allowed("thought_gemini"):
                resp = self._query_gemini(thought_prompt)
                _record("thought_gemini")
                if resp:
                    return resp.strip()
            elif provider in getattr(self, 'local_llms', {}) and _allowed(f"thought_{provider}"):
                resp = self._query_local_llm(provider, thought_prompt)
                _record(f"thought_{provider}")
                if resp:
                    return resp.strip()
            # Local small instruct fallback
            resp = self._local_instruct_response(thought_prompt)
            if resp:
                return resp.strip()
        except Exception as e:
            logger.warning(f"[SpontaneousThought] provider error: {e}")
            _record("thought_error", error=str(e))

        # Final deterministic fallback if everything fails
        return random.choice([
            "Adjusting limb timing to improve traction.",
            "Stabilizing torso while advancing hips.",
            "Coordinating contralateral limbs to gain forward momentum."
        ])

    # --- Intelligent Provider Selection ---
    def _get_available_providers(self) -> Dict[str, bool]:
        """Detect available providers/models at runtime."""
        availability = {
            "openai": bool(getattr(self, 'openai_client', None)),
            "anthropic": bool(getattr(self, 'anthropic_client', None)),
            "gemini": bool(getattr(self, 'gemini_model', None)),
            "conversation": hasattr(self, 'conversation_model') and hasattr(self, 'conversation_tokenizer'),
            "qa": hasattr(self, 'qa_model') and hasattr(self, 'qa_tokenizer'),
        }
        # Include local LLMs if loaded
        try:
            for name in getattr(self, 'local_llms', {}).keys():
                availability[name] = True
        except Exception:
            pass
        return availability

    def _meta_select_provider(self, prompt: str, availability: Dict[str, bool]) -> Optional[str]:
        """
        Use OpenAI as a meta-reasoner to select the best provider for the given prompt.
        Returns one of: 'openai' | 'anthropic' | 'gemini' | 'conversation' | 'qa'.
        """
        if not getattr(self, 'openai_client', None):
            return None
        try:
            selector_model = os.environ.get('QUARK_SELECTOR_MODEL', 'gpt-4o')
            providers = [name for name, ok in availability.items() if ok]
            system_msg = (
                "You are a routing policy for a multi-provider assistant. "
                "Choose the single best provider for the user's prompt from the provided list. "
                "Return strict JSON with fields: provider and reason. "
                "The provider MUST be one of the strings in the 'available' array provided by the user. "
                "Prefer small local models (conversation/qa or local LLMs) for short chit-chat or simple QA. "
                "Use OpenAI for complex reasoning or instruction following; Anthropic for nuanced safety or reasoning; Gemini for general Q&A. "
                "Never choose a provider that is not in the available list."
            )
            user_msg = json.dumps({
                "available": providers,
                "prompt": prompt,
            })
            if not self._allow_call('selector'):
                return None
            resp = self.openai_client.chat.completions.create(
                model=selector_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=150,
            )
            content = (resp.choices[0].message.content or '').strip()
            self._record_call('selector')
            try:
                data = json.loads(content)
                provider = data.get('provider')
                if provider in providers:
                    logger.info(f"[ModelSelector] Selected provider: {provider} | Reason: {data.get('reason', '')}")
                    return provider
            except Exception:
                logger.warning(f"[ModelSelector] Non-JSON selector response: {content}")
            # Fallback: simple heuristic if parsing fails
            return None
        except Exception as e:
            logger.error(f"[ModelSelector] Error selecting provider: {e}")
            self._record_call('selector', error=str(e))
            if 'insufficient_quota' in str(e) or '429' in str(e):
                self._cooldown('selector', 300.0)
            return None

    # --- Local LLMs (Phi-3, Qwen2.5-3B, Mistral-7B) ---
    def _load_local_llms(self):
        """Attempt to load optional local LLMs from the dataset directory.
        Set QUARK_LOCAL_MODELS (comma-separated: phi3,qwen,mistral) to constrain.
        """
        self.local_llms: Dict[str, Dict[str, Any]] = {}
        root = os.environ.get('QUARK_LOCAL_MODELS_DIR', '/Users/camdouglas/quark/data/datasets/models')
        try:
            allow = os.environ.get('QUARK_LOCAL_MODELS', '').strip()
            allow_set = set([s.strip().lower() for s in allow.split(',') if s.strip()]) if allow else None
        except Exception:
            allow_set = None

        candidates = {
            'phi3': os.path.join(root, 'microsoft', 'Phi-3-mini-4k-instruct'),
            'qwen': os.path.join(root, 'Qwen', 'Qwen2.5-3B-Instruct'),
            'mistral': os.path.join(root, 'mistralai', 'Mistral-7B-Instruct-v0.3')
        }

        for name, path in candidates.items():
            if allow_set is not None and name not in allow_set:
                continue
            try:
                if not os.path.isdir(path):
                    continue
                tok = AutoTokenizer.from_pretrained(path, use_fast=True)
                # Try to load with device_map auto; fallback to CPU/mps
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                model = AutoModelForCausalLM.from_pretrained(
                    path, torch_dtype=dtype, device_map='auto'
                )
                self.local_llms[name] = { 'tokenizer': tok, 'model': model }
                logger.info(f"[LocalLLM] Loaded {name} from {path}")
            except Exception as e:
                logger.warning(f"[LocalLLM] Could not load {name} at {path}: {e}")

    def _query_local_llm(self, name: str, prompt: str) -> Optional[str]:
        entry = self.local_llms.get(name)
        if not entry:
            return None
        try:
            tok = entry['tokenizer']
            model = entry['model']
            input_ids = tok.encode(prompt, return_tensors='pt', max_length=2048, truncation=True)
            device = next(model.parameters()).device
            input_ids = input_ids.to(device)
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.92,
                    repetition_penalty=1.1,
                    do_sample=True,
                    eos_token_id=tok.eos_token_id
                )
            text = tok.decode(out[0], skip_special_tokens=True)
            return text.split('\nQuark:')[-1].strip() if 'Quark:' in text else text.strip()
        except Exception as e:
            logger.error(f"[LocalLLM] Error generating with {name}: {e}")
            return None

    # --- Rate limit core ---
    def _init_rate_limiters(self):
        def get_int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, default))
            except Exception:
                return default
        self._rpm_caps = {
            'selector': get_int('QUARK_SELECTOR_RPM', 6),
            'openai': get_int('QUARK_OPENAI_RPM', 6),
            'anthropic': get_int('QUARK_ANTHROPIC_RPM', 4),
            'gemini': get_int('QUARK_GEMINI_RPM', 6),
        }
        self._calls_window = {k: deque() for k in self._rpm_caps.keys()}
        self._cooldown_until = {k: 0.0 for k in self._rpm_caps.keys()}
        self.usage: Dict[str, Dict[str, Any]] = {
            k: {'requests': 0, 'errors': 0, 'blocked': 0, 'last_error': ''}
            for k in self._rpm_caps.keys()
        }

    def _now(self) -> float:
        return time.time()

    def _prune_old(self, provider: str):
        one_min_ago = self._now() - 60.0
        dq = self._calls_window[provider]
        while dq and dq[0] < one_min_ago:
            dq.popleft()

    def _allow_call(self, provider: str) -> bool:
        if self._now() < self._cooldown_until.get(provider, 0.0):
            self.usage[provider]['blocked'] += 1
            return False
        self._prune_old(provider)
        if len(self._calls_window[provider]) >= self._rpm_caps[provider]:
            self.usage[provider]['blocked'] += 1
            return False
        return True

    def _record_call(self, provider: str, error: Optional[str] = None):
        self._calls_window[provider].append(self._now())
        self.usage[provider]['requests'] += 1
        if error:
            self.usage[provider]['errors'] += 1
            self.usage[provider]['last_error'] = error

    def _cooldown(self, provider: str, seconds: float):
        self._cooldown_until[provider] = max(self._cooldown_until.get(provider, 0.0), self._now() + seconds)