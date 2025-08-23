from brain_architecture.neural_core.hybrid_language.router import SparseMixtureRouter, SubModel


def _keyword_score(keyword: str):
    return lambda prompt: 1.0 if keyword in prompt else 0.0


def test_router_top_k():
    router = SparseMixtureRouter(k=1)
    router.register(SubModel("model_a", _keyword_score("hello"), backend="slm"))
    router.register(SubModel("model_b", _keyword_score("world"), backend="llm"))

    out = router.route("hello there")
    assert list(out.keys()) == ["model_a"]

    out = router.route("world news")
    assert list(out.keys()) == ["model_b"]
