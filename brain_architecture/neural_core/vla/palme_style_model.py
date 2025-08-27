from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from .vla_interface import VLAInterface


class _TinyVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(32, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.proj(h)


class _TinyTextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 20000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 64)
        self.rnn = nn.GRU(64, 64, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        e = self.embed(tokens)
        _, h = self.rnn(e)
        return h[-1]


class _TinyFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([v, t], dim=-1)
        return self.mlp(x)


class _TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.vel = nn.Linear(32, 3)
        self.gait = nn.Linear(32, 2)  # frequency, amplitude

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"target_velocity": self.vel(x), "gait": self.gait(x)}


class PalmEStyleVLA(VLAInterface):
    """
    Tiny PaLM-E–style VLA: image + text → high-level control intents.
    """

    def __init__(self):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for PalmEStyleVLA")
        self.vision = _TinyVisionEncoder()
        self.text = _TinyTextEncoder()
        self.fuse = _TinyFusion()
        self.head = _TinyHead()

    def _to_tensor(self, image: np.ndarray, tokens: np.ndarray) -> Any:
        import torch
        img = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tok = torch.from_numpy(tokens).unsqueeze(0).long()
        return img, tok

    def _simple_tokenize(self, text_prompt: str, vocab: int = 20000, max_len: int = 32) -> np.ndarray:
        # trivial tokenizer: hash words to ids
        words = text_prompt.strip().split()
        ids = [(hash(w) % (vocab - 2)) + 2 for w in words][:max_len]
        if not ids:
            ids = [1]
        return np.array(ids, dtype=np.int64)

    def infer(self, image: np.ndarray, text_prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255).astype(np.uint8)
        tokens = self._simple_tokenize(text_prompt)

        import torch
        with torch.no_grad():
            t_img, t_tok = self._to_tensor(img, tokens)
            v = self.vision(t_img)
            t = self.text(t_tok)
            z = self.fuse(v, t)
            out = self.head(z)
        tv = out["target_velocity"].squeeze(0).cpu().numpy().tolist()
        gait = out["gait"].squeeze(0).cpu().numpy().tolist()  # [freq, amp]
        return {"target_velocity": tv, "gait": {"frequency": gait[0], "amplitude": gait[1]}}


