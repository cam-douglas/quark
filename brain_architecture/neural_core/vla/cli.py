#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None

from .palme_style_model import PalmEStyleVLA


def load_image(path: Path) -> np.ndarray:
    if Image is None:
        raise RuntimeError("Pillow is required: pip install pillow")
    img = Image.open(path).convert("RGB")
    img = img.resize((256, 256))
    return np.array(img, dtype=np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run VLA on image + text prompt")
    ap.add_argument("image", type=str, help="Path to image file")
    ap.add_argument("prompt", type=str, help="Text prompt")
    args = ap.parse_args()

    image = load_image(Path(args.image))
    vla = PalmEStyleVLA()
    out = vla.infer(image=image, text_prompt=args.prompt)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())


