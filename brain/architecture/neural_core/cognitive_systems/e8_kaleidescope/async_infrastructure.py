"""Async infrastructure and LLM clients for E8 Mind System.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import asyncio
import json
import os
import time
import traceback
from typing import Dict, List, Optional

from .config import EMBED_DIM
from .utils import get_path

# Optional dependencies
try:
    from datetime import datetime, timezone
except ImportError:
    datetime = None
    timezone = None

try:
    import ollama
except ImportError:
    ollama = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

import numpy as np

class Probe:
    """Debug probe for logging system events."""
    def __init__(self, run_id: str):
        self.path = get_path("debug/probe.ndjson", run_id)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = asyncio.Lock()

    async def log(self, **kv):
        """Async log entry."""
        if datetime and timezone:
            kv["ts"] = datetime.now(timezone.utc).isoformat()
        try:
            async with self._lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(kv, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def log_sync(self, **kv):
        """Synchronous log entry."""
        if datetime and timezone:
            kv["ts"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(kv, ensure_ascii=False) + "\n")
        except Exception:
            pass

def set_asyncio_exception_logger(probe: Probe):
    """Set up asyncio exception logging."""
    try:
        loop = asyncio.get_running_loop()
        def _handler(loop, context):
            msg = context.get("message", "")
            exc = context.get("exception")
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) if exc else msg
            try:
                if loop.is_running() and not loop.is_closed():
                    asyncio.create_task(probe.log(ev="loop_exception", message=msg, traceback=tb))
                else:
                    probe.log_sync(ev="loop_exception", message=msg, traceback=tb)
            except Exception:
                probe.log_sync(ev="loop_exception", message=msg, traceback=tb)
        loop.set_exception_handler(_handler)
    except RuntimeError:  # No running loop
        pass

class InstrumentedLock:
    """Lock with performance instrumentation."""
    def __init__(self, name: str = "lock", probe: Optional[Probe] = None):
        self._lock = asyncio.Lock()
        self.name = name
        self.probe = probe
        self._t_acq = 0.0

    async def __aenter__(self):
        t0 = time.time()
        await self._lock.acquire()
        wait = (time.time() - t0) * 1000.0
        if self.probe:
            await self.probe.log(ev="lock_acquire", lock=self.name, wait_ms=round(wait, 2))
        self._t_acq = time.time()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        held = (time.time() - self._t_acq) * 1000.0
        if self.probe:
            await self.probe.log(ev="lock_release", lock=self.name, held_ms=round(held, 2))
        self._lock.release()

class AsyncOpenAIClient:
    """Async OpenAI client wrapper."""
    def __init__(self, api_key: str, console):
        try:
            from openai import AsyncOpenAI, BadRequestError
            self.client = AsyncOpenAI(api_key=api_key)
            self.BadRequestError = BadRequestError
        except ImportError:
            raise ImportError("openai package required for AsyncOpenAIClient")
        self.console = console

    async def chat(self, messages: List[Dict], model: Optional[str] = None,
                   max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        """Send chat completion request."""
        try:
            cc = await self.client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
            )
            if cc.choices:
                return (cc.choices[0].message.content or "").strip()
            return "[LLM ERROR] No choices returned from API."
        except self.BadRequestError as e:
            self.console.log(f"[bold red]OpenAI API Error: {e}[/bold red]")
            return f"[LLM ERROR] {e}"

    async def get_logprobs_and_tokens(self, messages: List[Dict], **kwargs) -> tuple:
        """Get log probabilities (placeholder)."""
        return -99.0, []

    async def embedding(self, text: str, model: Optional[str] = None,
                       dimensions: Optional[int] = None) -> List[float]:
        """Get text embedding."""
        try:
            res = await self.client.embeddings.create(
                input=[text], model=model or "text-embedding-3-small", dimensions=dimensions
            )
            return res.data[0].embedding
        except Exception as e:
            self.console.log(f"[bold red]OpenAI Embedding Error: {e}[/bold red]")
            return np.zeros(EMBED_DIM).tolist()

    async def batch_embedding(self, texts: List[str], model: Optional[str] = None,
                             dimensions: Optional[int] = None) -> List[List[float]]:
        """Get batch embeddings."""
        try:
            res = await self.client.embeddings.create(
                input=texts, model=model or "text-embedding-3-small", dimensions=dimensions
            )
            return [d.embedding for d in res.data]
        except Exception as e:
            self.console.log(f"[bold red]OpenAI Batch Embedding Error: {e}[/bold red]")
            return [np.zeros(EMBED_DIM).tolist() for _ in texts]

class OllamaClient:
    """Async Ollama client wrapper."""
    def __init__(self, ollama_model: str, console):
        if ollama is None:
            raise RuntimeError("Python package 'ollama' not installed. Please `pip install ollama`.")
        self.client = ollama.AsyncClient()
        self.model = ollama_model
        self.console = console

    async def chat(self, messages: List[Dict], **kwargs) -> str:
        """Send chat request to Ollama."""
        try:
            res = await self.client.chat(model=self.model, messages=messages)
            return res["message"]["content"].strip()
        except Exception as e:
            self.console.log(f"[bold red]Ollama Chat Error: {e}[/bold red]")
            return f"[LLM ERROR] Could not connect to Ollama or model '{self.model}' not found."

    async def get_logprobs_and_tokens(self, messages: List[Dict], **kwargs) -> tuple:
        """Get log probabilities (placeholder)."""
        return -99.0, []

    async def embedding(self, text: str, model: Optional[str] = None,
                       dimensions: Optional[int] = None) -> List[float]:
        """Get text embedding from Ollama."""
        try:
            res = await self.client.embeddings(model=model or self.model, prompt=text)
            emb = res["embedding"]
            if dimensions:
                if len(emb) > dimensions:
                    emb = emb[:dimensions]
                elif len(emb) < dimensions:
                    emb = emb + [0.0] * (dimensions - len(emb))
            return emb
        except Exception as e:
            self.console.log(f"[bold red]Ollama Embedding Error: {e}[/bold red]")
            v = np.random.standard_normal(EMBED_DIM).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
            return v.tolist()

    async def batch_embedding(self, texts: List[str], model: Optional[str] = None,
                             dimensions: Optional[int] = None) -> List[List[float]]:
        """Get batch embeddings from Ollama."""
        try:
            tasks = [self.embedding(t, model, dimensions) for t in texts]
            return await asyncio.gather(*tasks)
        except Exception as e:
            self.console.log(f"[bold red]Ollama Batch Embedding Error: {e}[/bold red]")
            out = []
            for _ in texts:
                v = np.random.standard_normal(EMBED_DIM).astype(np.float32)
                v /= (np.linalg.norm(v) + 1e-12)
                out.append(v.tolist())
            return out

class GeminiClient:
    """Async Gemini client wrapper."""
    def __init__(self, api_key: str, console, model_name: str = "gemini-1.5-flash"):
        if genai is None:
            raise RuntimeError("google.generativeai not available")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.console = console

    async def chat(self, messages: List[Dict], **kwargs) -> str:
        """Send chat request to Gemini."""
        try:
            user_text = messages[-1]["content"] if messages else ""
            response = await self.model.generate_content_async(user_text)
            return response.text.strip()
        except Exception as e:
            self.console.log(f"[bold red]Gemini Chat Error: {e}[/bold red]")
            return f"[LLM ERROR] {e}"

    async def get_logprobs_and_tokens(self, messages: List[Dict], **kwargs) -> tuple:
        """Get log probabilities (placeholder)."""
        return -99.0, []

    async def embedding(self, text: str, **kwargs) -> List[float]:
        """Get text embedding (fallback)."""
        v = np.random.standard_normal(EMBED_DIM).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        return v.tolist()

    async def batch_embedding(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Get batch embeddings (fallback)."""
        return [await self.embedding(text) for text in texts]
