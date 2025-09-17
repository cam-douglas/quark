"""Web server and main entry point for E8 Mind System.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import asyncio
import os
import numpy as np
from typing import Dict, Any

from .config import BASE_DIR, SEMANTIC_DOMAIN, EMBED_DIM
from .e8_mind_core import E8Mind
from .async_infrastructure import AsyncOpenAIClient, OllamaClient, GeminiClient
from .utils import get_run_id, UniversalEmbeddingAdapter
from .profiles.loader import load_profile

# Optional dependencies
try:
    from aiohttp import web
    import aiohttp_cors
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None
    aiohttp_cors = None

try:
    import ollama
except ImportError:
    ollama = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Rich console fallback
try:
    from rich.console import Console
except ImportError:
    class Console:
        def __init__(self, record=False):
            pass
        def log(self, *a, **k):
            print(*a)
        def print(self, *a, **k):
            print(*a)
        def rule(self, *a, **k):
            print("-" * 40)
        def export_text(self):
            return ""
        def print_exception(self):
            import traceback
            traceback.print_exc()

console = Console(record=True)

def _collect_config_from_user() -> Dict[str, Any]:
    """Interactive configuration collector."""
    print("\n" + "="*60)
    print("E8 Mind Server Configuration")
    print("="*60)

    cfg = {}
    print("\nAvailable LLM Providers:")
    print("1. OpenAI (GPT-4)")
    print("2. Ollama (Local)")
    print("3. Gemini")

    while True:
        choice = input("\nSelect provider (1-3): ").strip()
        if choice in ['1', '2', '3']:
            cfg['provider_choice'] = choice
            break
        print("Invalid choice. Please select 1, 2, or 3.")

    if choice == '1':
        key = input("OpenAI API Key: ").strip()
        model = input("Model (gpt-4-turbo-preview): ").strip() or "gpt-4-turbo-preview"
        cfg.update({"openai_api_key": key, "openai_model_name": model})
    elif choice == '2':
        model = input("Ollama Model (llama3): ").strip() or "llama3"
        cfg['ollama_model_name'] = model
    elif choice == '3':
        key = input("Gemini API Key: ").strip()
        model = input("Model (gemini-1.5-flash): ").strip() or "gemini-1.5-flash"
        cfg.update({"gemini_api_key": key, "gemini_model_name": model})

    return cfg

# Placeholder API handlers
async def handle_get_graph(request):
    """Handle graph API endpoint."""
    mind = request.app['mind']
    try:
        from .tasks import export_graph
        graph_data = export_graph(mind.memory.graph_db.graph)
        return web.json_response(graph_data)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def handle_get_telemetry(request):
    """Handle telemetry API endpoint."""
    mind = request.app['mind']
    try:
        telemetry = {
            "step": mind.step_num,
            "mood": mind.mood.mood_vector,
            "nodes": len(mind.memory.graph_db.graph.nodes()),
            "narrative": mind.subconscious.narrative
        }
        return web.json_response(telemetry)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def handle_stream_telemetry(request):
    """Handle streaming telemetry endpoint."""
    response = web.StreamResponse()
    response.headers['Content-Type'] = 'text/event-stream'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    await response.prepare(request)

    try:
        while True:
            mind = request.app['mind']
            telemetry = {
                "step": mind.step_num,
                "mood": mind.mood.mood_vector,
                "nodes": len(mind.memory.graph_db.graph.nodes())
            }
            await response.write(f"data: {telemetry}\n\n".encode())
            await asyncio.sleep(1.0)
    except Exception:
        pass

    return response

async def handle_get_blueprint(request):
    """Handle blueprint API endpoint."""
    mind = request.app['mind']
    return web.json_response(mind.blueprint if hasattr(mind, 'blueprint') else {})

async def handle_add_concept(request):
    """Handle add concept API endpoint."""
    mind = request.app['mind']
    try:
        data = await request.json()
        concept_text = data.get('text', '')
        if concept_text:
            # Add concept to memory (simplified)
            embedding = await mind.get_embedding(concept_text)
            node_id = f"api_concept_{mind.step_num}"
            mind.memory.main_vectors[node_id] = embedding
            mind.memory.graph_db.add_node(node_id, type="external_concept", label=concept_text)
            return web.json_response({"success": True, "node_id": node_id})
        return web.json_response({"error": "No text provided"}, status=400)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def handle_trigger_dream(request):
    """Handle dream trigger API endpoint."""
    mind = request.app['mind']
    try:
        dream_result = await mind.dream_engine.dream_step()
        return web.json_response({"dream": dream_result})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def handle_get_qeng_telemetry(request):
    """Handle quantum engine telemetry."""
    return web.json_response({"quantum_state": "placeholder"})

async def handle_get_qeng_ablation(request):
    """Handle quantum engine ablation."""
    return web.json_response({"ablation": "placeholder"})

async def handle_get_qeng_probabilities(request):
    """Handle quantum engine probabilities."""
    return web.json_response({"probabilities": "placeholder"})

async def shutdown_market_feed(app):
    """Shutdown market feed on app shutdown."""
    pass

async def main():
    """Main entry point for E8 Mind Server."""
    if not AIOHTTP_AVAILABLE:
        console.log("[bold red]aiohttp not available. Cannot start web server.[/bold red]")
        return

    run_id = get_run_id()
    provider_native_embed_dim = 1536
    IS_EMBED_PLACEHOLDER = False

    # LLM Provider setup (simplified)
    LLM_PROVIDER = os.getenv("E8_PROVIDER", "stub").lower()

    if LLM_PROVIDER == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set.")
        llm_client = AsyncOpenAIClient(api_key, console)
        model_name = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        embedding_model = "text-embedding-3-small"
    elif LLM_PROVIDER == "ollama":
        model_name = os.getenv("OLLAMA_MODEL", "llama3")
        llm_client = OllamaClient(model_name, console)
        embedding_model = "nomic-embed-text"
    elif LLM_PROVIDER == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set.")
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        llm_client = GeminiClient(api_key, console, model_name)
        embedding_model = "models/embedding-001"
    else:
        # Stub client
        class StubClient:
            def __init__(self, console):
                self.console = console
            async def chat(self, *a, **k):
                return "This is a placeholder response from a stubbed LLM."
            async def embedding(self, *a, **k):
                import numpy as np
                return np.random.randn(provider_native_embed_dim).tolist()
            async def batch_embedding(self, texts, *a, **k):
                import numpy as np
                return [np.random.randn(provider_native_embed_dim).tolist() for _ in texts]

        llm_client = StubClient(console)
        model_name = "stub"
        embedding_model = "stub"
        IS_EMBED_PLACEHOLDER = True

    console.log(f"[INIT] LLM Provider: {LLM_PROVIDER}")
    console.log(f"[INIT] Model: {model_name}")

    # Test embedding dimension
    console.log("[INIT] Probing embedding dimension from provider...")
    _test_vec = await llm_client.embedding("adapter_probe")
    if isinstance(_test_vec, dict) and "embedding" in _test_vec:
        _test_vec = _test_vec["embedding"]
    if isinstance(_test_vec, list) and _test_vec and isinstance(_test_vec[0], (list, np.ndarray)):
        _test_vec = _test_vec[0]

    embed_in_dim = int(len(_test_vec))
    if embed_in_dim > 1:
        provider_native_embed_dim = embed_in_dim
    console.log(f"[INIT] Detected provider embedding dimension: {provider_native_embed_dim}")

    # Setup embedding adapter
    try:
        profile_name = os.getenv("MIND_PROFILE", "default")
        sem, _ = load_profile(profile_name)
        probe_native = np.zeros(provider_native_embed_dim, dtype=np.float32)
        probe_post = sem.post_embed(probe_native)
        adapter_in_dim = int(np.asarray(probe_post, dtype=np.float32).size)
        console.log(f"[INIT] post_embed output dim: {adapter_in_dim}")
    except Exception as e:
        adapter_in_dim = provider_native_embed_dim
        console.log(f"[INIT] post_embed probe failed: {e}. Using provider dim.")

    embed_adapter = UniversalEmbeddingAdapter(adapter_in_dim, EMBED_DIM)
    console.log(f"[INIT] Universal Embedding Adapter created: {adapter_in_dim} -> {EMBED_DIM}")

    # Initialize E8Mind
    mind = E8Mind(
        semantic_domain_val=SEMANTIC_DOMAIN,
        run_id=run_id,
        llm_client_instance=llm_client,
        client_model=model_name,
        embedding_model_name=embedding_model,
        embed_adapter=embed_adapter,
        embed_in_dim=provider_native_embed_dim,
        console=console
    )

    # Setup web server
    app = web.Application()
    app['mind'] = mind
    app['sse_clients'] = set()
    mind.sse_clients = app['sse_clients']

    app.on_shutdown.append(shutdown_market_feed)
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True, expose_headers="*", allow_headers="*"
        )
    })

    # Add routes
    app.router.add_get("/api/graph", handle_get_graph)
    app.router.add_get("/api/telemetry", handle_get_telemetry)
    app.router.add_get("/api/telemetry/stream", handle_stream_telemetry)
    app.router.add_get("/api/blueprint", handle_get_blueprint)
    app.router.add_post("/api/concept", handle_add_concept)
    app.router.add_post("/api/action/dream", handle_trigger_dream)
    app.router.add_get("/api/qeng/telemetry", handle_get_qeng_telemetry)
    app.router.add_get("/api/qeng/ablation", handle_get_qeng_ablation)
    app.router.add_get("/api/qeng/probabilities", handle_get_qeng_probabilities)

    # Static files
    static_path = os.path.join(BASE_DIR, 'static')
    if os.path.exists(static_path):
        app.router.add_static('/', static_path, show_index=True, default_filename='index.html')

    for route in list(app.router.routes()):
        cors.add(route)

    # Start server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 7870)
    await site.start()
    console.log("[bold green]E8 Mind Server running at http://localhost:7870[/bold green]")
    console.log(f"Run ID: {run_id}")

    # Run cognitive cycle
    cycle_task = asyncio.create_task(mind.run_cognitive_cycle())
    try:
        await cycle_task
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.log("\n[bold cyan]Shutting down E8 Mind...[/bold cyan]")
    except Exception as e:
        console.log(f"[bold red]CRITICAL ERROR in main: {e}[/bold red]")
        console.print_exception()
