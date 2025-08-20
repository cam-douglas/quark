from __future__ import annotations
import hashlib, math
from typing import List, Tuple
try:
    from sentence_transformers import SentenceTransformer  # optional
    _HAS_ST = True
except Exception:
    _HAS_ST = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # optional
    _HAS_TF = True
except Exception:
    _HAS_TF = False

class TextEmbedder:
    """Adaptive embedder: uses SentenceTransformers if present, else TF-IDF, else hashing char-trigrams."""
    def __init__(self, model_names: List[str] | None = None):
        self.model = None
        self.vectorizer = None
        self.backend = "hash-trigram"
        if _HAS_ST:
            names = model_names or ["intfloat/e5-small-v2", "all-MiniLM-L6-v2"]
            for n in names:
                try:
                    self.model = SentenceTransformer(n)
                    self.backend = f"st::{n}"
                    break
                except Exception:
                    continue
        if self.model is None and _HAS_TF:
            self.vectorizer = TfidfVectorizer(max_features=8192, ngram_range=(1,2))
            self.backend = "tfidf"
    def fit(self, docs: List[str]):  # only for TF-IDF
        if self.vectorizer:
            self.vectorizer.fit(docs)
    def encode(self, docs: List[str]):
        if self.model:
            return self.model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
        if self.vectorizer:
            mat = self.vectorizer.transform(docs)
            return mat
        # fallback: hashing char-trigrams
        return [self._hash_trigrams(d) for d in docs]
    def _hash_trigrams(self, s: str, dim: int = 2048):
        v = [0.0]*dim
        t = s.lower()
        for i in range(len(t)-2):
            tri = t[i:i+3]
            h = int(hashlib.md5(tri.encode()).hexdigest(), 16) % dim
            v[h] += 1.0
        # L2 normalize
        norm = math.sqrt(sum(x*x for x in v)) or 1.0
        return [x/norm for x in v]
def cosine(a, b) -> float:
    # supports dense lists and sparse sklearn rows
    try:
        # sklearn sparse
        from numpy import ndarray
        if hasattr(a, "multiply"):
            num = a.multiply(b).sum()
            da = (a.multiply(a).sum())**0.5
            db = (b.multiply(b).sum())**0.5
            return float(num/(da*db)) if da and db else 0.0
    except Exception:
        pass
    # dense
    num = sum(x*y for x,y in zip(a,b))
    da  = sum(x*x for x in a) ** 0.5
    db  = sum(x*x for x in b) ** 0.5
    return float(num/(da*db)) if da and db else 0.0
